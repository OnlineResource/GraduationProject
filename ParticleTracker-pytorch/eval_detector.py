import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import os

from datasets.eval_preprocess import eval_pd_preprocess
from datasets.utils import particle_detector_paint
from models.make_model import pd_model_preparation


def _eval(opt):
    if opt.do_log:
        writer = SummaryWriter(opt.logs_path)
        print(f'tensorboard --logdir="{os.path.dirname(os.path.abspath(__file__)) + opt.logs_path[1:]}"',
              ' --port=10219 --bind_all')

    pd_model, device = pd_model_preparation(opt, train=False)

    # start eval
    pd_model.eval()
    with torch.no_grad():
        img = eval_pd_preprocess(opt.eval_path)
        img = img.to(device)
        start = time.time()
        pred, _ = pd_model(img)
        end = time.time()
        print(f'Elapsed time: {end - start}')
        particle_detector_paint(
            img.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
            max_num_in_cell=opt.max_num_in_cell,
            circle_scale=opt.circle_scale,
            fig_title=opt.particle_detector_checkpoint_path + '\n' + opt.eval_path,
        )
    if opt.do_log:
        # writer.add_graph()
        writer.close()


def eval_main():
    start_time = time.strftime('%Y_%m_%d %H_%M_%S')
    print(f'Start time: {start_time}')

    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--logs_path', type=str, default='./logs/eval/pd' + start_time + '/',
        help='Checkpoints save path')
    parse.add_argument(
        '--cell_size', type=int, default=2 ** 4,  # TODO: 2 ** 4
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
        '--cuda_ids', type=list, default=[3],
        help='CUDA device ids')
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
        opt.particle_detector_checkpoint_path = 'saved_pd_checkpoints/2022_03_10 21_53_32/epoch_65_loss_0.0222.pth'
        opt.basic_pow = 7
    else:
        opt.particle_detector_checkpoint_path = 'checkpoints/pd/2022_03_16 09_20_45/epoch_10_loss_0.0480.pth'
        opt.basic_pow = 6

    opt.eval_path = 'data/eval/syn/eval0004.tif'

    _eval(opt)


if __name__ == '__main__':
    eval_main()
