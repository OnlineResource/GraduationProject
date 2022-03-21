import cv2
import numpy as np
import torch

from datasets.utils import pad_img


def tif_read(file_path):
    """

    :param file_path: 读取的 tif 文件地址
    :return: list(各帧图像: ndarray)
    """
    img_list = cv2.imreadmulti(file_path, flags=-1)
    img_list = list(img_list[1])
    return img_list


def tif_particle_detector(tif_path, standardize=True, uint16_min=None, uint16_max=None):
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


def eval_pd_preprocess(file_path):
    x_ori = tif_particle_detector(file_path, standardize=True)
    x = pad_img(
        x_ori,
        x_ori[0].shape[0],
        x_ori[0].shape[1],
        new_size=512,
    )
    return torch.from_numpy(x).unsqueeze(1)


def eval_pm_preprocess(file_path0, file_path1):
    x0 = eval_pd_preprocess(file_path0)[:, None]
    x1 = eval_pd_preprocess(file_path1)[:, None]
    x = torch.cat([x0, x1], dim=1)
    return x  # Tensor:(1,2,1,512,512)


def eval_pm_seq_preprocess(file_path):
    import glob
    img_path_list = glob.glob(file_path + '*')
    img_path_list.sort()
    img_list = []
    for img_path in img_path_list:
        img_list.append(eval_pd_preprocess(img_path)[:, None])
    x = torch.cat(img_list, dim=1)
    return x  # Tensor:(1,T,1,512,512)


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


# def tif_cv2show(img_list, winname="default"):
#     """
#
#     :param img_list: list(各帧图像: ndarray)
#     :param winname: 创建的cv显示窗口名，推荐采用图片文件名
#     :return: None
#     """
#     cv2.namedWindow(winname)
#     for index, img in enumerate(img_list, start=1):
#         img_show = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
#         cv2.imshow(winname, img_show)
#         cv2.waitKey(0)
#
#     cv2.destroyAllWindows()
#
#     return x


if __name__ == '__main__':
    re = eval_pd_preprocess('../data/test.tif')
    print('Module test')
