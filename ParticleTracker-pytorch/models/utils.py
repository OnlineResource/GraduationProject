from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """
    Multi-layer perceptron (i.e. Multi-layer full-connection neural network)
    使用 nn.Conv1d 代替 nn.Linear
    (linear + BN + ReLU) * n + linear
    forward()输入 (N,C,L)
    :param in_channel:
    :param out_channels:
    :param do_in:
    :return:
        nn.Sequential
    """

    def __init__(self, in_channel: int, out_channels: list, do_in=True) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for idx, out_channel in enumerate(out_channels):
            self.layers.append(nn.Conv1d(in_channel, out_channel, kernel_size=(1,), bias=True))
            if idx < len(out_channels) - 1:
                if do_in:
                    self.layers.append(nn.InstanceNorm1d(out_channel))
                self.layers.append(nn.ReLU())  # TODO: 默认 mlp 中使用 ReLU
            in_channel = out_channel
        self.out_channel = out_channels[-1]

    def forward(self, x):
        if x.shape[2] > 0:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            return torch.empty((1, self.out_channel, 0), device=x.device)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, conv_num_in_block, conv_norm='none'):
        super().__init__()
        self.layers = nn.ModuleList()
        for idx in range(conv_num_in_block):
            self.layers.append(nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1)))
            if conv_norm == 'batch':
                self.layers.append(nn.BatchNorm2d(out_channels))
            elif conv_norm == 'instance':
                self.layers.append(nn.InstanceNorm2d(out_channels))
            elif conv_norm == 'none':
                pass
            else:
                raise ValueError(f'conv_norm={conv_norm} in ConvBlock is invalid! use "none", "batch", "instance"')
            self.layers.append(nn.ReLU())  # TODO: in old using LReLU(0.2)
            in_channels = out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SENetBlock(nn.Module):

    def __init__(self, in_channels, dense_div):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, max(in_channels // dense_div, 1)),
            nn.ReLU(),  # TODO: in old using LReLU(0.2)
            nn.Linear(max(in_channels // dense_div, 1), in_channels),
            nn.Sigmoid()
        ])

    def forward(self, x):
        x0 = x
        for layer in self.layers:
            x = layer(x)
        return torch.einsum('nchw,nc->nchw', x0, x)


class UNetEncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dense_div, conv_num_in_block, conv_norm, dropout_rate):
        super().__init__()

        self.conv_block = ConvBlock(in_channels, out_channels, conv_num_in_block, conv_norm)
        self.senet_block = SENetBlock(out_channels, dense_div)
        self.dropout2d = nn.Dropout2d(dropout_rate)
        self.maxpool2d = nn.MaxPool2d((2, 2))

    def forward(self, x):
        # x -- (N, C, H, W)
        x = self.conv_block(x)
        x = self.senet_block(x)
        x = self.dropout2d(x)
        x_pool = self.maxpool2d(x)
        # x -- (N, C_new, H, W)
        # x_pool -- (N, C_new, H/2, W/2)
        return x_pool, x


class UNetDecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dense_div, conv_num_in_block, conv_norm):
        super().__init__()

        self.conv_block = ConvBlock(in_channels, out_channels, conv_num_in_block, conv_norm)
        self.senet_block = SENetBlock(out_channels, dense_div)
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=(2, 2))

    def forward(self, x, x_cat):
        # x -- (N, C1, H, W)
        # x_cat -- (N, C2, H, W)
        x = self.conv_block(x)
        x_ori = self.senet_block(x)
        x = self.up_sample(x_ori)
        # re -- (N, C1_new + C2, H, W)
        return torch.cat((x_cat, x), dim=1), x_ori


class UNetEncoder(nn.Module):
    """
    ParticleDetector Encoder1&2
    """

    def __init__(self, in_channels, num_coders, basic_pow, dense_div, conv_num_in_block, conv_norm, dropout_rate):
        super().__init__()
        self.unet_encoder_list = nn.ModuleList()
        self.cat_channels_list = []
        for idx in range(num_coders):
            out_channels = 2 ** basic_pow
            self.unet_encoder_list.append(
                UNetEncoderBlock(in_channels, out_channels, dense_div, conv_num_in_block, conv_norm, dropout_rate)
            )
            self.cat_channels_list.append(out_channels)
            in_channels = out_channels

        self.out_channels = in_channels
        self.final_cat_channels = in_channels

    def forward(self, x):
        x_cat_list = []
        for unet_encoder in self.unet_encoder_list:
            x, x_cat = unet_encoder(x)
            x_cat_list.append(x_cat)  # encoder输出的未经maxpool的特征图，用于和之后decoder的结果cat
        return x, x_cat_list


class UNetDecoder(nn.Module):
    """
    ParticleDetector Decoder
    """

    def __init__(self, in_channels, num_coders, basic_pow, dense_div, conv_num_in_block, conv_norm, cat_channels_list):
        super().__init__()

        self.unet_decoder_list = nn.ModuleList()
        for idx in range(num_coders):
            out_channels = 2 ** basic_pow
            self.unet_decoder_list.append(
                UNetDecoderBlock(in_channels, out_channels, dense_div, conv_num_in_block, conv_norm)
            )
            if idx == 0:
                self.final_cat_channels = out_channels
            in_channels = out_channels + cat_channels_list.pop()

        self.out_channels = in_channels

    def forward(self, x, x_cat_list):
        x_cat_final2 = None
        for unet_decoder in self.unet_decoder_list:
            x, x_ = unet_decoder(x, x_cat_list.pop())
            if x_cat_final2 is None:
                x_cat_final2 = x_  # 第一个 decoder 在 up_sample 前的输出即为 x_cat_final2
        return x, x_cat_final2


class UNetFinal(nn.Module):  # TODO: 最终可能需要修改此处命名方式，重新训练一下

    def __init__(self, in_channels, basic_pow, final_channels, dense_div, conv_num_in_block, conv_norm):
        super().__init__()
        out_channels = 2 ** basic_pow
        self.conv_block = ConvBlock(in_channels, out_channels, conv_num_in_block, conv_norm)
        self.senet_block = SENetBlock(out_channels, dense_div)
        self.final_conv1 = nn.Conv2d(out_channels, out_channels, (1, 1), padding=(0, 0))
        self.final_conv2 = nn.Conv2d(out_channels, final_channels, (1, 1), padding=(0, 0))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.final_channels = final_channels
        self.max_num_in_cell = int((final_channels - 1) / 3)

    def forward(self, x):
        # x -- (N, C, H, W)
        x = self.conv_block(x)
        x = self.senet_block(x)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        x_score = self.softmax(x[:, :self.max_num_in_cell + 1, :, :])
        x_loc_center = self.sigmoid(x[:, self.max_num_in_cell + 1:self.max_num_in_cell + 3, :, :])
        x_loc_bias = self.tanh(x[:, self.max_num_in_cell + 3:, :, :])
        x_loc = torch.clip(torch.cat([x_loc_center] * 3, dim=1) + x_loc_bias, 0., 1.)
        # x -- (N, final_channels, H, W)
        return torch.cat([x_score, x_loc], dim=1)


class CoordsTranslater(nn.Module):
    """
    ParticleDetector 输出的坐标和特征图转为 两个 list：归一化的绝对坐标（float） 以及 对应坐标上的特征描述子
    len(list) = batch_size
    """

    def __init__(self, cell_size, max_num_in_cell) -> None:
        super().__init__()
        self.cell_size = cell_size
        self.max_num_in_cell = max_num_in_cell
        self.cell_grid = None

    def forward(self, x_loc, x_fea):
        # x_loc -- (N, max_num_in_cell + 3 + max_num_in_cell [ID], H1, W1) -- H1,W1 = full_H,W / cell_size
        # x_fea -- (N, 2 ** basic_pow * 2, H2, W2) -- H2,W2 = full_H,W
        batch_size, loc_channels, cell_h, cell_w = x_loc.shape
        fea_channels = x_fea.shape[1]
        if loc_channels != 1 + 4 * self.max_num_in_cell:
            raise ValueError(f'ParticleDetector output x_loc has {x_loc.shape[1]} channels, '
                             f'but "opt.max_num_in_cell": {self.max_num_in_cell} * 4 + 1 does not match this!')

        if self.cell_grid is None:
            cell_grid_h, cell_grid_w = torch.meshgrid(
                torch.arange(0, cell_h, dtype=torch.long, device=x_loc.device),
                torch.arange(0, cell_w, dtype=torch.long, device=x_loc.device),
                indexing='ij'
            )
            self.cell_grid = torch.cat([cell_grid_h[None], cell_grid_w[None]] * self.max_num_in_cell, dim=0)

        batch_abs_coords = self.cell_size * (
                x_loc[:, self.max_num_in_cell + 1:3 * self.max_num_in_cell + 1, :, :] + self.cell_grid
        )
        batch_in_cell = torch.argmax(x_loc[:, :self.max_num_in_cell + 1, :, :], dim=1).long()
        batch_ids = x_loc[:, -self.max_num_in_cell:, :, :].long()

        # 从此开始，batch_size 存为 list，长度为N
        batch_abs_coords_list = list()
        batch_features_list = list()
        batch_ids_list = list()
        for n in range(batch_size):
            in_cell = batch_in_cell[n]
            abs_coords = torch.empty((0, 2), dtype=torch.float32, device=x_loc.device)
            features = torch.empty((0, fea_channels), dtype=torch.float32, device=x_loc.device)
            ids = torch.empty((0,), dtype=torch.long, device=x_loc.device)
            for num_in_cell in range(1, self.max_num_in_cell + 1):  # 对于含有 num_in_cell 个粒子的 cell 进行遍历
                cell_loc_matrix = torch.eq(in_cell, num_in_cell)

                if torch.sum(cell_loc_matrix) != 0:  # 含有 num_in_cell 个粒子的 cell 的数量不为 0
                    abs_coords_temp = batch_abs_coords[n][:2 * num_in_cell, cell_loc_matrix].permute(1, 0).reshape(-1,
                                                                                                                   2)
                    abs_coords_floor_temp = torch.floor(abs_coords_temp).long()
                    features_temp = x_fea[n][:, abs_coords_floor_temp[:, 0], abs_coords_floor_temp[:, 1]].permute(1, 0)
                    ids_temp = batch_ids[n][:num_in_cell, cell_loc_matrix].permute(1, 0).reshape(-1)

                    # 若 cell 中有 num_in_cell 个粒子，则存储 num_in_cell 份该坐标，特征向量（存储的ID不同，但也是num_in_cell个）
                    abs_coords = torch.cat([abs_coords, abs_coords_temp], dim=0)
                    features = torch.cat([features, features_temp], dim=0)
                    ids = torch.cat([ids, ids_temp], dim=0)

            batch_abs_coords_list.append(abs_coords)
            batch_features_list.append(features)
            batch_ids_list.append(ids)
        return batch_abs_coords_list, batch_features_list, batch_ids_list


class CoordsEncoder(nn.Module):
    """
    输出特征描述子

    """

    def __init__(self, basic_pow, desc_pow) -> None:
        super().__init__()
        mlp_out_channels_list = [2 ** desc_pow, 2 ** desc_pow]
        self.fea_proj = MLP(in_channel=2 ** (basic_pow + 1), out_channels=[2 ** (basic_pow + 1)])
        self.encoder = MLP(in_channel=2 + 2 ** (basic_pow + 1), out_channels=mlp_out_channels_list)

    def forward(self, coords_list, fea_list, img_h, img_w):
        desc_list = []
        for coords, fea in zip(coords_list, fea_list):
            coords = normalize_coords(coords.transpose(0, 1)[None], (img_h, img_w))
            fea = self.fea_proj(fea.transpose(0, 1)[None])
            fea = F.normalize(fea, p=2, dim=1)
            encoder_inputs = torch.cat([coords, fea], dim=1)
            desc_list.append(self.encoder(encoder_inputs))
        return desc_list


def attention(query, key, value, softmax):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = softmax(scores)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivity """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        """
        d_model is just feature_dim
        d_model 必须能够被 num_heads 整除
        """
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads

        # 用 nn.Conv1d 代替 linear，此时作用于每个关键点的权重都是相同的
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=(1,), bias=True)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).reshape(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value, self.softmax)
        return self.merge(x.reshape(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        """
        (default usage) num_heads = 4
        """
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP(feature_dim * 2, [feature_dim * 2, feature_dim])

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):

    def __init__(self, desc_channels, gnn_layer_pairs):
        """
        :param desc_channels:
        :param gnn_layer_pairs:

        layers = ['self', 'cross', 'self', 'cross', ...,  'self', 'cross'] length = 2 * gnn_layer_pairs
        every layer: AttentionalPropagation = attn + mlp
        attn(MultiHeadedAttention) = merge + proj
        """

        super().__init__()
        layer_names = ['self', 'cross'] * gnn_layer_pairs
        self.layers = nn.ModuleList([
            AttentionalPropagation(desc_channels, num_heads=4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int):
    """
    Perform Sinkhorn Normalization in Log-space for stability
    Z: 即经过 einsum 以及 / self.config['descriptor_dim'] ** .5 ，并加入了 bin_scores 行和列的 scores 张量（矩阵）
    epsilon 取值默认为 1？
    """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(z + u.unsqueeze(2), dim=1)
    return z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha, iters: int):
    """
    Perform Differentiable Optimal Transport in Log-space for stability
    变量均为tensor，因此可以进行梯度的后向传播，端到端训练

    alphe: 即 bin_score，论文中的 z
    """
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], dim=-1),
                           torch.cat([bins1, alpha], dim=-1)], dim=1)

    norm = - (ms + ns).log()  # TODO: 此处关于log_sinkhorn_iterations的前处理原理还需要理解
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)  # 只是按batch_size扩展张量

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def normalize_coords(coords, image_shape):
    """ Normalize locations based on image image_shape"""
    size = coords.new_tensor(image_shape)[None, :, None]
    center = (size / 2)
    scaling = size.max(dim=1, keepdim=True).values * 0.5  # TODO: 此处指定的keypoints坐标normalize scaling值可能需要修改
    return (coords - center) / scaling  # 将值缩到 -1 ~ 1 之间


class TrajectoryManagerV1(nn.Module):
    """
    提供两帧之间的匹配管理
    """

    def __init__(self, points0):
        super().__init__()

    def forward(self, points1, matches1):
        """
            track_dict = {
                last_keypoint_idx: [start_frame_idx, (x0, y0), (x1, y1), ...],
                l1: ... ,
                l2: ... ,
            }

            last_keypoint_idx = 'finished_track_id_x' 则轨迹已停止
        """
        self.frame_idx += 1
        for idx, match in enumerate(matches1):
            if match > -1:
                # 完成匹配的关键点添加到对应的原轨迹尾部
                self.tracking_dict['last:' + str(match)].append(tuple(points1[idx]))
                # 将 last_keypoint_idx 更新为 now_keypoint_idx
                self.tracking_dict.update({'now:' + str(idx): self.tracking_dict.pop('last:' + str(match))})
            else:
                self.tracking_dict['now:' + str(idx)] = [self.frame_idx, tuple(points1[idx])]

        for key in list(self.tracking_dict):
            if key.startswith('last'):
                self.track_finished.append(self.tracking_dict.pop(key))
        for key in list(self.tracking_dict):
            self.tracking_dict.update({key.replace('now', 'last'): self.tracking_dict.pop(key)})

    def stop(self, length_threshold=None):
        for key in list(self.tracking_dict):
            self.track_finished.append(self.tracking_dict.pop(key))
        if length_threshold is not None:
            self._length_filter(length_threshold)

    def _length_filter(self, length_threshold):
        for track in self.track_finished:
            if len(track) - 1 >= length_threshold:  # -1 为了排除列表中指定起始帧的元素（第一个数）
                self.track_length_filtered.append(track)
