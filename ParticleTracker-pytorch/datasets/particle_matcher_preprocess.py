import glob
import os

import numpy as np

from utils.csv_process import csv_particle_matcher
from utils.tif_process import tif_particle

if __name__ == '__main__':
    frame_per_file = 10
    file_num = 500
    frame_h, frame_w = 500, 500

    train_file_num = 40
    valid_file_num = 10
    test_file_num = 0

    tif_dir = './data/synthesized_data_10_frames_per_file_150i200-1700_30n50_10d20'
    csv_dir = tif_dir
    tif_list = glob.glob(os.path.join(tif_dir, '*.tif'))
    if len(tif_list) < train_file_num + valid_file_num + test_file_num:
        raise ValueError('Not enough files!')

    x_all = np.zeros((file_num, frame_per_file, frame_h, frame_w))
    for idx, tif_path in enumerate(tif_list):
        print(idx)
        # TODO: important mark
        #  返回的是将图像最大最小值限制到 0~1 之间后的单通道图像数据 ndarray--[file_num{idx}, frame_per_file, frame_h, frame_w]
        x_all[idx] = tif_particle(
            tif_path, standardize=True, uint16_min=None, uint16_max=None)

    csv_list = glob.glob(os.path.join(csv_dir, '*.csv'))
    if len(csv_list) < train_file_num + valid_file_num + test_file_num:
        raise ValueError('Not enough files!')

    y_all = np.empty((file_num, frame_per_file), dtype=object)
    for idx, csv_path in enumerate(csv_list):
        print(idx)
        # TODO: important mark
        #  返回的是将图像最大最小值限制到 0~1 之间后的单通道图像数据 ndarray--[file_num{idx}, frame_per_file]
        #  每一个元素包含一个 tuple( 粒子ID：ndarray(n,), 粒子绝对坐标：ndarray(n, 2) )，其中 n 为该帧中的粒子数量
        y_all[idx] = csv_particle_matcher(csv_path)

    x_train = x_all[0:train_file_num]
    x_valid = x_all[train_file_num:train_file_num + valid_file_num]
    x_test = x_all[train_file_num + valid_file_num:train_file_num + valid_file_num + test_file_num]
    y_train = y_all[0:train_file_num]
    y_valid = y_all[train_file_num:train_file_num + valid_file_num]
    y_test = y_all[train_file_num + valid_file_num:train_file_num + valid_file_num + test_file_num]
    np.savez(
        tif_dir + '_starter.npz',
        x_train=x_train,
        x_valid=x_valid,
        x_test=x_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
    )
    pass
