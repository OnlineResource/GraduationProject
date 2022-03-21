import argparse
import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os

from datasets.make_dataset import pd_dataset_preparation
from models.loss_fn import particle_detector_score, particle_detector_loss
from models.make_model import pd_model_preparation


def _train(opt):
    os.makedirs(opt.checkpoints_path)
    with open(opt.checkpoints_path + 'opt.txt', 'w') as f:
        print(f'Writing configurations to {opt.checkpoints_path}opt.txt')
        f.write(str(opt))
    if opt.do_log:
        writer = SummaryWriter(opt.logs_path)
        print(f'tensorboard --logdir="{os.path.dirname(os.path.abspath(__file__)) + opt.logs_path[1:]}"',
              ' --port=10219 --bind_all')

    pd_model, device = pd_model_preparation(opt, train=True)
    train_loader, valid_loader = pd_dataset_preparation(opt)

    adam_optimizer = torch.optim.Adam(
        pd_model.parameters(),
        lr=1e-4,
    )

    epochs = opt.epochs
    # start training
    for epoch in range(1, epochs + 1):
        pd_model.train()
        epoch_loss = 0
        epoch_score = np.zeros(4)
        for img, label in tqdm(train_loader):
            pd_model.zero_grad()

            img = img.to(device)
            label = label.to(device)
            pred, _ = pd_model(img)

            batch_mean_loss = particle_detector_loss(pred, label, alpha=opt.alpha)
            batch_mean_loss.backward()
            adam_optimizer.step()

            epoch_loss += batch_mean_loss.item()
            epoch_score += particle_detector_score(pred, label)

        epoch_mean_loss = epoch_loss / len(train_loader)
        epoch_mean_score = epoch_score / len(train_loader)

        print('Epoch [{}/{}], loss: {:.4f}'.format(
            epoch,
            epochs,
            epoch_mean_loss,
        ))
        print('Epoch f1: {:.4f}, recall: {:.4f}, precision: {:.4f}, rmse: {:.4f}\n'.format(
            *epoch_mean_score
        ))
        if opt.do_log:
            writer.add_scalar("loss/train", epoch_mean_loss, epoch)
            writer.add_scalar("train_score/1.f1", epoch_mean_score[0], epoch)
            writer.add_scalar("train_score/2.recall", epoch_mean_score[1], epoch)
            writer.add_scalar("train_score/3.precision", epoch_mean_score[2], epoch)
            writer.add_scalar("train_score/4.rmse", epoch_mean_score[3], epoch)

        # save checkpoint & do validation every "opt.save_epoch_num" epoch(s)
        if epoch % opt.save_epoch_num == 0:
            print('### Epoch [{}/{}] finished, doing validation'.format(
                epoch,
                epochs,
            ))

            pd_model.eval()
            valid_loss = 0
            valid_score = np.zeros(4)
            with torch.no_grad():
                for img, label in valid_loader:
                    img = img.to(device)
                    label = label.to(device)
                    pred, _ = pd_model(img)
                    batch_mean_loss = particle_detector_loss(pred, label, alpha=opt.alpha)
                    valid_loss += batch_mean_loss.item()
                    valid_score += particle_detector_score(pred, label)

            valid_mean_loss = valid_loss / len(valid_loader)
            valid_mean_score = valid_score / len(valid_loader)

            print('Val loss: {:.4f}'.format(valid_mean_loss, ))
            print('Val f1: {:.4f}, recall: {:.4f}, precision: {:.4f}, rmse: {:.4f}\n'.format(
                *valid_mean_score
            ))
            if opt.do_log:
                writer.add_scalar("loss/valid", valid_mean_loss, epoch)
                writer.add_scalar("valid_score/1.f1", valid_mean_score[0], epoch)
                writer.add_scalar("valid_score/2.recall", valid_mean_score[1], epoch)
                writer.add_scalar("valid_score/3.precision", valid_mean_score[2], epoch)
                writer.add_scalar("valid_score/4.rmse", valid_mean_score[3], epoch)

            model_out_path = opt.checkpoints_path + 'epoch_{}_loss_{:.4f}.pth'.format(
                epoch,
                valid_mean_loss
            )
            if isinstance(pd_model, DataParallel):
                torch.save(pd_model.module.state_dict(), model_out_path)
            else:
                torch.save(pd_model.state_dict(), model_out_path)
            with open(model_out_path + '_score.txt', 'w') as f:
                f.write(str(valid_mean_score))
            print('### Checkpoint saved to {}\n'.format(model_out_path))

    if opt.do_log:
        writer.close()


def train_main():
    start_time = time.strftime('%Y_%m_%d %H_%M_%S')
    print(f'Start time: {start_time}')

    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--datasets_path', type=str,
        default='data/synthesized_data_10_frames_per_file_150i200-1700_30n50_10d20_prepared.npz',
        help='Dataset file path')
    parse.add_argument(
        '--checkpoints_path', type=str, default='./checkpoints/pd/' + start_time + '/',
        help='Checkpoints save path')
    parse.add_argument(
        '--logs_path', type=str, default='./logs/train/' + start_time + '/',
        help='Checkpoints save path')
    parse.add_argument(
        '--cell_size', type=int, default=2 ** 4,  # TODO: 2 ** 4
        help='UNet cell size, NOT particle size!')
    parse.add_argument(
        '--max_num_in_cell', type=int, default=3,
        help='maximal number of particles in one cell')
    parse.add_argument(
        '--basic_pow', type=int, default=6,  # TODO: 6 !!!
        help='Every conv layer outputs "2 ** basic_pow" channels')
    parse.add_argument(
        '--dense_div', type=int, default=8,  # TODO: 8
        help='SENet squeeze the input channels to "n // dense_div" channels')
    parse.add_argument(
        '--conv_num_in_block', type=int, default=3,  # TODO: 3
        help='A ConvBlock will have "conv_num_in_block" conv layers')
    parse.add_argument(
        '--conv_norm', type=str, default='instance',  # TODO: instance
        help='use which ("none", "batch", "instance") normalization method in ConvBlock')
    parse.add_argument(
        '--dropout_rate', type=float, default=0.3,  # TODO: 0.3
        help='UNetEncoder Dropout2d drop rate')
    parse.add_argument(
        '--alpha', type=np.ndarray, default=np.array([1, 1, 1, 1]),  # TODO: [0.5, 1, 5, 10]
        help='focal loss alpha')
    parse.add_argument(
        '--batch_size', type=int, default=8,  # TODO: ?
        help='Training batch size')
    parse.add_argument(
        '--epochs', type=int, default=300,  # TODO: 300
        help='The total number of epoch')
    parse.add_argument(
        '--save_epoch_num', type=int, default=5,  # TODO: 10
        help='Save checkpoints every "save_epoch_num" epoch(s)')
    parse.add_argument(
        '--cuda_ids', type=list, default=[0, 1],
        help='CUDA device ids')
    parse.add_argument(
        '--particle_detector_checkpoint_path', type=str, default=None,
        help='Load ParticleDetector checkpoint from the path "particle_detector_checkpoint_path", None for ERROR')
    parse.add_argument(
        '--do_log', type=bool, default=False,
        help='Do tensorboard logging')
    parse.add_argument(
        '--force_cpu', type=bool, default=False,
        help='to be filled')

    opt = parse.parse_args()

    # opt.particle_detector_checkpoint_path = 'saved_pd_checkpoints/2022_03_08 15_49_31/epoch_90_loss_0.0222.pth'

    _train(opt)


if __name__ == '__main__':
    train_main()
