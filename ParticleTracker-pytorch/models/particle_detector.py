import math
import torch
from torch import nn

from models.utils import UNetEncoder, UNetDecoder, UNetFinal


class ParticleDetector(nn.Module):

    def __init__(
            self,
            img_channels=1,
            cell_size=2 ** 4,
            basic_pow=6,
            max_num_in_cell=3,
            dense_div=8,
            conv_num_in_block=3,
            conv_norm='None',
            dropout_rate=0.3,
    ):
        super().__init__()
        num_coders = int(math.log2(cell_size))

        # 1 no_particle + 3(max_num_in_cell) score + 2 coords_center + 3(max_num_in_cell) * 2 coords_bias
        final_channels = 3 + 3 * max_num_in_cell
        self.unet_encoder1 = UNetEncoder(
            img_channels,
            num_coders,
            basic_pow,
            dense_div,
            conv_num_in_block,
            conv_norm,
            dropout_rate,
        )

        self.unet_decoder = UNetDecoder(
            self.unet_encoder1.out_channels,
            num_coders,
            basic_pow,
            dense_div,
            conv_num_in_block,
            conv_norm,
            self.unet_encoder1.cat_channels_list,
        )
        self.unet_encoder2 = UNetEncoder(
            self.unet_decoder.out_channels,
            num_coders,
            basic_pow,
            dense_div,
            conv_num_in_block,
            conv_norm,
            dropout_rate,
        )

        self.unet_final = UNetFinal(
            # self.unet_encoder1.final_cat_channels + self.unet_encoder2.out_channels,  # cat_final1
            self.unet_decoder.final_cat_channels + self.unet_encoder2.out_channels,  # cat_final2
            basic_pow,
            final_channels,
            dense_div,
            conv_num_in_block,
            conv_norm,
        )

    def forward(self, x):
        # x -- (N, C=1, H, W)
        x, x_cat_list = self.unet_encoder1(x)
        # x_cat_final1 = x
        x_fea, x_cat_final2 = self.unet_decoder(x, x_cat_list)
        x, _ = self.unet_encoder2(x_fea)  # TODO: 初步考虑将此处的 x_fea 引出，作为特征描述子的原始向量

        # TODO: 此处采用底层处理后的特征图 x_cat_final2 进行拼接（ x_cat_final1 是底层处理前的特征图）
        # x = torch.cat((x_cat_final1, x), dim=1)
        x = torch.cat((x_cat_final2, x), dim=1)

        x = self.unet_final(x)
        return x, x_fea


def _module_test():
    import argparse
    import numpy as np
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--cell_size', type=int, default=2 ** 4,  # TODO: 4 = 2 ** 2 in paper
        help='UNet cell size, NOT particle size!')
    parse.add_argument(
        '--max_num_in_cell', type=int, default=3,
        help='maximal number of particles in one cell')
    parse.add_argument(
        '--basic_pow', type=int, default=6,  # TODO: 5 in paper  (6 in Code)
        help='Every conv layer outputs "2 ** basic_pow" channels')
    parse.add_argument(
        '--dense_div', type=int, default=8,  # TODO: 8 in paper
        help='SENet squeeze the input channels to "n // dense_div" channels')
    parse.add_argument(
        '--conv_num_in_block', type=int, default=3,  # TODO: 3 in paper
        help='A ConvBlock will have "conv_num_in_block" conv layers')
    parse.add_argument(
        '--conv_norm', type=str, default='instance',  # TODO: 'none' in paper
        help='use which ("none", "batch", "instance") normalization method in ConvBlock')
    parse.add_argument(
        '--dropout_rate', type=float, default=0.3,  # TODO: 0.3 in paper
        help='UNetEncoder Dropout2d drop rate')
    parse.add_argument(
        '--cuda_ids', type=list, default=[0],
        help='CUDA device ids')
    parse.add_argument(
        '--force_cpu', type=bool, default=False,
        help='to be filled')
    opt = parse.parse_args()
    opt.particle_detector_checkpoint_path = '../checkpoints/pd/2022_03_06 10_41_02/epoch_180_loss_0.0630.pth'
    # opt.particle_detector_checkpoint_path = None

    from models.make_model import pd_model_preparation
    pd_model, device = pd_model_preparation(opt, train=True)
    x = torch.ones((1, 1, 512, 512)).to(device)

    # from torchsummaryX import summary
    # summary(pd_model, x)

    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            pd_model(x)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # y = pp(x)
    # print(pp)
    # print(x.shape, y.shape)


if __name__ == '__main__':
    _module_test()
