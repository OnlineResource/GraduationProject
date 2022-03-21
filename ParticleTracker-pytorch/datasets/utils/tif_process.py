import glob
import os

import cv2
# from torch.utils.tensorboard import SummaryWriter
import numpy as np


def tif_read(file_path):
    """

    :param file_path: 读取的 tif 文件地址
    :return: list(各帧图像: ndarray)
    """
    img_list = cv2.imreadmulti(file_path,
                               flags=-1)
    img_list = list(img_list[1])
    return img_list


# def tif_tensorboard(img_list, log_path="log"):
#     """
#
#     :param img_list: list(各帧图像: ndarray)
#     :param log_path: 默认值"log"需要调用时在工程根目录下，否则需要手动改动
#     :return: None
#     """
#     writer = SummaryWriter(log_path)
#     for index, img in enumerate(img_list, start=1):
#         img_show = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
#         writer.add_image("test", img_show, index, dataformats='HW')
#     writer.close()


def tif_cv2show(img_list, winname="default"):
    """

    :param img_list: list(各帧图像: ndarray)
    :param winname: 创建的cv显示窗口名，推荐采用图片文件名
    :return: None
    """
    cv2.namedWindow(winname)
    for index, img in enumerate(img_list, start=1):
        img_show = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow(winname, img_show)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def tif_particle(tif_path, standardize=True, uint16_min=None, uint16_max=None):
    frame_data = tif_read(tif_path)
    if standardize:
        for idx, frame in enumerate(frame_data):
            frame = np.clip(
                frame,
                uint16_min if uint16_min is not None else np.min(frame),
                uint16_max if uint16_max is not None else np.max(frame),
            )
            frame_data[idx] = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    x = np.stack(frame_data)

    return x


if __name__ == '__main__':
    tif_dir = '../data/synthesized_data_1'
    tif_list = glob.glob(os.path.join(tif_dir, '*.tif'))
    for tif_path in tif_list:
        re = tif_particle_point(tif_path)

    pass
