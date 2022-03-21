import torch
from torch import nn

from models.loss_fn import particle_matcher_score, particle_matcher_loss
from models.particle_matcher import MatchScoreDealer
from models.utils import CoordsTranslater, TrajectoryManagerV1, ConvBlock, CoordsEncoder
import time


class ParticleTrackerV1(nn.Module):  # 用于训练 matcher
    def __init__(
            self,
            particle_detector_model,
            particle_matcher_model,
            cell_size=2 ** 4,
            max_num_in_cell=3,
            basic_pow=6,
            desc_pow=7,
            match_threshold=0.0,
    ) -> None:
        super().__init__()

        # 以下部分输入单张图片组成的 mini-batch
        self.particle_detector = particle_detector_model
        self.feature_preprocessor = ConvBlock(
            2 ** (basic_pow + 1),
            2 ** (basic_pow + 1),
            conv_num_in_block=3,
            conv_norm='instance'
        )
        # TODO: CoordsTranslater 的性能还需优化
        self.coords_translater = CoordsTranslater(cell_size=cell_size, max_num_in_cell=max_num_in_cell)
        self.coords_encoder = CoordsEncoder(basic_pow=basic_pow, desc_pow=desc_pow)

        # 以下部分输入成对图片组成的 mini-batch
        self.particle_matcher = particle_matcher_model
        self.match_score_dealer = MatchScoreDealer(match_threshold=match_threshold)
        # self.trajectory_manager = None

    def forward(self, x, y=None, eval=False):
        # x -- (N, T, C=1, H, W)
        # y -- (N, T, 13=1 + 3 scores + 2*3 coords + 3 ids, H/16, W/16)
        batch_size, frame_length, img_channels, img_h, img_w = x.shape

        # 首帧检测
        x0_loc, x0_fea = self.particle_detector(x[:, 0, :, :, :])
        x0_fea = self.feature_preprocessor(x0_fea)
        # x_loc -- (N, 10=1 + 3 scores + 2*3 coords, H/16, W/16)
        # x_fea -- (N, 128=2 ** (basic_pow+1), H, W)

        # TODO: 这里的 x_loc 使用 ground truth，用于训练 matcher
        if y is None:
            dummy_id = x0_loc.new_full((batch_size, 3, *x0_loc.shape[-2:]), -1.)
            x0_loc = torch.cat([x0_loc, dummy_id], dim=1)
        else:
            x0_loc = y[:, 0, :, :, :]
            # x_loc -- (N, 13, H/16, W/16)
            dummy_id = None

        x0_coords_list, x0_fea_list, x0_ids_list = self.coords_translater(x0_loc, x0_fea)
        x0_fea_list = self.coords_encoder(x0_coords_list, x0_fea_list, img_h, img_w)

        if eval:
            coords_list = [x0_coords_list]
            matches_list = []
            movie_batch_loss = None
            movie_batch_score = None
        else:
            movie_batch_loss = x.new_tensor([0.])
            movie_batch_score = x.new_tensor([[0., 0., 0.]])
            coords_list = None
            matches_list = None

        # self.trajectory_manager = TrajectoryManagerV1(x0_coords_list)

        for t in range(1, frame_length):  # 逐帧运行
            x1_loc, x1_fea = self.particle_detector(x[:, t, :, :, :])
            x1_fea = self.feature_preprocessor(x1_fea)

            # TODO: 这里的 x_loc 使用 ground truth，用于训练 matcher
            if y is None:
                x1_loc = torch.cat([x1_loc, dummy_id], dim=1)
            else:
                x1_loc = y[:, t, :, :, :]

            x1_coords_list, x1_fea_list, x1_ids_list = self.coords_translater(x1_loc, x1_fea)
            x1_fea_list = self.coords_encoder(x1_coords_list, x1_fea_list, img_h, img_w)

            scores_list = self.particle_matcher(x0_fea_list, x1_fea_list)
            matches0_list = self.match_score_dealer(scores_list)

            # self.trajectory_manager(x1_coords_list[0].detach().cpu().numpy(),
            #                         matches1_list[0].detach().cpu().numpy())

            # x0_loc, x0_fea, x0_fea_list, x0_coords_list = x1_loc, x1_fea, x1_fea_list, x1_coords_list

            if eval:
                coords_list.append(x1_coords_list)
                matches_list.append(matches0_list)
            else:
                # 一个 T 帧的 movie 由 T-1 个 loss 和 score 分别求和
                # 此处两个函数输出的是在 batch (单设备) 上平均过的单次匹配 loss 和 score
                movie_batch_loss += particle_matcher_loss(scores_list, x0_ids_list, x1_ids_list, loss_gamma=1.0)
                movie_batch_score += particle_matcher_score(x0_ids_list, x1_ids_list, matches0_list)

            x0_fea_list, x0_ids_list = x1_fea_list, x1_ids_list

        if eval:
            return coords_list, matches_list
        else:
            # 将 movie 求和的 loss 和 score 除以 T - 1，得到单次匹配的平均 loss 和 score
            # 跨设备的 loss 和 score 求平均在 model 外部进行
            batch_loss = movie_batch_loss / (frame_length - 1)
            batch_score = movie_batch_score / (frame_length - 1)
            return batch_loss, batch_score


def _module_test():
    from torch.profiler import profile, record_function, ProfilerActivity
    import argparse
    import numpy as np
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--cell_size', type=int, default=2 ** 4,
        help='UNet cell size, NOT particle size!')
    parse.add_argument(
        '--max_num_in_cell', type=int, default=3,
        help='maximal number of particles in one cell')
    parse.add_argument(
        '--alpha', type=np.ndarray, default=np.array([0.5, 1, 5, 10]),
        help='focal loss alpha')
    parse.add_argument(
        '--basic_pow', type=int, default=6,
        help='Every conv layer outputs "2 ** basic_pow" channels')
    parse.add_argument(
        '--dense_div', type=int, default=8,
        help='SENet squeeze the input channels to "n // dense_div" channels')
    parse.add_argument(
        '--conv_num_in_block', type=int, default=3,
        help='A ConvBlock will have "conv_num_in_block" conv layers')
    parse.add_argument(
        '--conv_norm', type=str, default='instance',
        help='use which ("none", "batch", "instance") normalization method in ConvBlock')
    parse.add_argument(
        '--dropout_rate', type=float, default=0.3,
        help='UNetEncoder Dropout2d drop rate')
    parse.add_argument(
        '--desc_pow', type=int, default=7,
        help='to be filled')
    parse.add_argument(
        '--gnn_layer_pairs', type=int, default=6,
        help='to be filled')
    parse.add_argument(
        '--sinkhorn_iters', type=int, default=50,
        help='to be filled')
    parse.add_argument(
        '--match_threshold', type=float, default=0.0,  # TODO: 后期删除，也即设为 0.0
        help='to be filled')
    parse.add_argument(
        '--cuda_ids', type=list, default=[0],
        help='CUDA device ids')
    parse.add_argument(
        '--force_cpu', type=bool, default=True,
        help='to be filled')
    opt = parse.parse_args()
    opt.particle_detector_checkpoint_path = None
    # opt.particle_tracker_checkpoint_path = '../saves_pm_checkpoints/2022_03_08 21_44_47/epoch_710_loss_7.8934.pth'
    opt.particle_tracker_checkpoint_path = None

    from models.make_model import pt_model_preparation
    pt_model, device = pt_model_preparation(opt, train=True)
    # x -- (N, T, C, H, W)
    # y -- (N, T, 9, H/16, W/16)
    x = torch.rand(1, 2, 1, 512, 512).to(device)
    y = torch.rand(1, 2, 13, 32, 32).to(device)
    # pt_model(x, y)

    from torchsummaryX import summary
    summary(pt_model, x, y)

    # with profile(activities=[
    #     ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         pt_model(x, y)
    #
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="self_cuda_mem", row_limit=10))


if __name__ == '__main__':
    _module_test()
