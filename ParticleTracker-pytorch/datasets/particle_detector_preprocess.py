import glob
import os

import numpy as np

from utils.csv_process import csv_particle_detector
from utils.tif_process import tif_particle

if __name__ == '__main__':
    frame_per_file = 50
    file_num = 100
    frame_h, frame_w = 500, 500

    train_file_num = 80
    valid_file_num = 20
    test_file_num = 0

    train_frame_num = train_file_num * frame_per_file
    valid_frame_num = valid_file_num * frame_per_file
    test_frame_num = test_file_num * frame_per_file

    tif_dir = './data/synthesized_data3_50_frames_per_file_600_max'
    tif_list = glob.glob(os.path.join(tif_dir, '*.tif'))
    if len(tif_list) < train_file_num + valid_file_num + test_file_num:
        raise ValueError('Not enough files!')
    x_all = np.zeros((frame_per_file * file_num, frame_h, frame_w))
    for idx, tif_path in enumerate(tif_list):
        print(idx)
        x_all[idx * frame_per_file:idx * frame_per_file + frame_per_file] = tif_particle(
            tif_path, standardize=True, uint16_min=None, uint16_max=None)

    csv_dir = './data/synthesized_data3_50_frames_per_file_600_max'
    csv_list = glob.glob(os.path.join(csv_dir, '*.csv'))
    if len(csv_list) < train_file_num + valid_file_num + test_file_num:
        raise ValueError('Not enough files!')
    y_all = np.empty(frame_per_file * file_num, dtype=object)
    for idx, csv_path in enumerate(csv_list):
        print(idx)
        y_all[idx * frame_per_file:idx * frame_per_file + frame_per_file] = csv_particle_detector(csv_path)

    x_train = x_all[0:train_frame_num]
    x_valid = x_all[train_frame_num:train_frame_num + valid_frame_num]
    x_test = x_all[train_frame_num + valid_frame_num:train_frame_num + valid_frame_num + test_frame_num]
    y_train = y_all[0:train_frame_num]
    y_valid = y_all[train_frame_num:train_frame_num + valid_frame_num]
    y_test = y_all[train_frame_num + valid_frame_num:train_frame_num + valid_frame_num + test_frame_num]
    np.savez(
        tif_dir + '.npz',
        x_train=x_train,
        x_valid=x_valid,
        x_test=x_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
    )
    pass

