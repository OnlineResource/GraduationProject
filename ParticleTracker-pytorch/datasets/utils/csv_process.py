import glob
import os

import numpy as np
import pandas as pd


def csv_read(file_path, get_frame_data=True):
    csv_data = pd.read_csv(file_path)
    if not get_frame_data:
        return csv_data
    else:
        col_name = list(csv_data.columns)[1:]
        csv_data = np.array(csv_data.loc[:, :])

        frame_num = np.max(csv_data[:, 0]).astype(np.uint8)  # 从第一列（帧数）中，获取该数据对应的 tif 图像总帧数
        frame_data = list()
        for frame_index in range(1, frame_num + 1):
            frame_data.append(csv_data[csv_data[:, 0] == frame_index, 1:])

        return frame_data, col_name


def csv_particle_detector(csv_path):
    frame_data, _ = csv_read(csv_path)
    y = np.empty(len(frame_data), dtype=object)
    for idx, frame_ndarray in enumerate(frame_data):
        y[idx] = frame_ndarray[:, 1:3]
    return y

def csv_particle_matcher(csv_path):
    frame_data, _ = csv_read(csv_path)
    y = np.empty(len(frame_data), dtype=object)
    for idx, frame_ndarray in enumerate(frame_data):
        y[idx] = (frame_ndarray[:, 0].astype(np.int32), frame_ndarray[:, 1:3])  # 多了一个 ID
    return y


if __name__ == '__main__':
    csv_dir = '../data/synthesized_data_1'
    csv_list = glob.glob(os.path.join(csv_dir, '*.csv'))
    for csv_path in csv_list:
        re = csv_particle_detector(csv_path)

    pass
