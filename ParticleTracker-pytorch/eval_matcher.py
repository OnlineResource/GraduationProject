import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import os

from datasets.eval_preprocess import eval_pm_preprocess
from datasets.utils import particle_matcher_paint
from models.make_model import pt_model_preparation


def _eval(opt):
    if opt.do_log:
        writer = SummaryWriter(opt.logs_path)
        print(f'tensorboard --logdir="{os.path.dirname(os.path.abspath(__file__)) + opt.logs_path[1:]}"',
              ' --port=10219 --bind_all')

    pt_model, device = pt_model_preparation(opt, train=False)

    # start eval
    pt_model.eval()
    with torch.no_grad():
        img_pair = eval_pm_preprocess(opt.eval_path0, opt.eval_path1)
        img_pair = img_pair.to(device)
        # TODO: 使用两张图片进行 matcher 测试，此时不需要传入 label
        start = time.time()

        # TODO: 当前若 eval=False 时也不传入 y ，则会得到非常大的 loss 和 非常小的 scores
        coords_list, matches_list = pt_model(img_pair, y=None, eval=True)
        end = time.time()
        print(f'Elapsed time: {end - start}')
        particle_matcher_paint(
            img_pair[0, :, 0].detach().cpu().numpy(),
            [coords_list[0][0].detach().cpu().numpy(), coords_list[1][0].detach().cpu().numpy()],
            matches_list[0][0][0].detach().cpu().numpy(),
            max_num_in_cell=opt.max_num_in_cell,
            circle_scale=opt.circle_scale,
            fig_title=(opt.particle_detector_checkpoint_path + '\n'
                       + opt.particle_tracker_checkpoint_path + '\n'
                       + opt.eval_path0 + '\n'
                       + opt.eval_path1),
        )
    if opt.do_log:
        writer.close()


def eval_main():
    start_time = time.strftime('%Y_%m_%d %H_%M_%S')
    print(f'Start time: {start_time}')

    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--logs_path', type=str, default='./logs/eval/pm' + start_time + '/',
        help='Checkpoints save path')
    parse.add_argument(
        '--cell_size', type=int, default=2 ** 4,  # TODO: 4 = 2 ** 2 in paper
        help='UNet cell size, NOT particle size!')
    parse.add_argument(
        '--max_num_in_cell', type=int, default=3,
        help='maximal number of particles in one cell')
    parse.add_argument(
        '--dense_div', type=int, default=8,  # TODO: 8
        help='SENet squeeze the input channels to "n // dense_div" channels')
    parse.add_argument(
        '--conv_num_in_block', type=int, default=3,  # TODO: 3
        help='A ConvBlock will have "conv_num_in_block" conv layers')
    parse.add_argument(
        '--conv_norm', type=str, default='instance',  # TODO: 'instance'
        help='use which ("none", "batch", "instance") normalization method in ConvBlock')
    parse.add_argument(
        '--dropout_rate', type=float, default=0.3,  # TODO: 0.3
        help='UNetEncoder Dropout2d drop rate')
    parse.add_argument(
        '--sinkhorn_iters', type=int, default=50,  # TODO: 20
        help='to be filled')
    parse.add_argument(
        '--match_threshold', type=float, default=0.0,  # TODO: 后期删除，也即设为 0.0
        help='to be filled')
    parse.add_argument(
        '--cuda_ids', type=list, default=[3],
        help='CUDA device ids')
    parse.add_argument(
        '--particle_detector_checkpoint_path', type=str, default=None,
        help='Load ParticleDetector checkpoint from the path "particle_detector_checkpoint_path", None for ERROR')
    parse.add_argument(
        '--particle_tracker_checkpoint_path', type=str, default=None,
        help='Load ParticleTracker checkpoint from the path "particle_tracker_checkpoint_path"')
    parse.add_argument(
        '--do_log', type=bool, default=False,
        help='Do tensorboard logging')
    parse.add_argument(
        '--force_cpu', type=bool, default=True,
        help='to be filled')
    parse.add_argument(
        '--circle_scale', type=float, default=1.5,
        help='to be filled')

    opt = parse.parse_args()
    basic7 = False
    if basic7:
        opt.particle_detector_checkpoint_path = 'saved_pd_checkpoints/2022_03_08 15_49_46/epoch_50_loss_0.0349.pth'
        opt.particle_tracker_checkpoint_path = 'checkpoints/pm/2022_03_07 19_53_00/epoch_20_loss_26.2513.pth'
        opt.basic_pow = 7
        opt.desc_pow = 7
        opt.gnn_layer_pairs = 6
    else:
        opt.particle_detector_checkpoint_path = 'saved_pd_checkpoints/2022_03_10 12_56_54/epoch_50_loss_0.0164.pth'
        opt.particle_tracker_checkpoint_path = 'saved_pm_checkpoints/2022_03_11 15_20_05/epoch_100_loss_3.0590.pth'
        opt.basic_pow = 6
        opt.desc_pow = 7
        opt.gnn_layer_pairs = 6

    opt.eval_path0 = 'data/eval/syn/eval0008.tif'
    opt.eval_path1 = 'data/eval/syn/eval0009.tif'
    # opt.eval_path0 = 'data/eval/real/eval_01.tif'
    # opt.eval_path1 = 'data/eval/real/eval_02.tif'

    _eval(opt)


if __name__ == '__main__':
    eval_main()
