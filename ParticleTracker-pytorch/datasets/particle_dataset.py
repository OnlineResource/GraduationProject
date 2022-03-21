import torch
from torch.utils.data import Dataset
import numpy as np

from datasets.utils import get_cell_label, pad_img, pad_img_with_id, get_cell_label_with_id


class ParticleDetectorDataset(Dataset):

    def __init__(
            self,
            file_path,
            purpose,
            cell_size=2 ** 4,
    ) -> None:
        super().__init__()
        if 'prepared' in file_path:
            if purpose == 'valid':
                file_path = file_path.replace('.npz', '_valid.npz')
            self.images = np.load(file_path, allow_pickle=True)
            self.x = self.images['x'].reshape(-1, 1, 512, 512)
            self.y = self.images['y'][:, :, :10, :, :].reshape(-1, 10, 32, 32)
        else:
            self.cell_size = cell_size
            # 此处导入的数据集已经为[0, 1]的浮点数
            self.images = np.load(file_path, allow_pickle=True)

            if purpose == 'train':
                self.x_ori = self.images['x_train']
                self.y_ori = self.images['y_train']
            elif purpose == 'valid':
                self.x_ori = self.images['x_valid']
                self.y_ori = self.images['y_valid']
            else:
                self.x_ori = self.images['x_test']
                self.y_ori = self.images['y_test']

            self.x_ori = self.x_ori.reshape(-1, 500, 500)
            self.y_ori = self.y_ori.reshape(-1, )

            self.x = pad_img(
                self.x_ori,
                self.x_ori.shape[-2],
                self.x_ori.shape[-1],
                new_size=512,
            )[:, None, ...]
            self.image_size = self.x.shape[-1]
            self.y = self.make_label()
            del self.x_ori
            del self.y_ori

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def make_label(self):
        labels = []
        for _, abs_coords in self.y_ori:  # 当前的 y 还是长度为图片数量的一维 ndarray，每个元素为 (n, 2) 的 n 个坐标数据
            cell_label = get_cell_label(abs_coords, self.image_size, self.cell_size)
            labels.append(cell_label)
        return np.array(labels)


class ParticleMatcherDataset(Dataset):

    def __init__(
            self,
            file_path,
            purpose,
            cell_size=2 ** 4,
    ) -> None:
        super().__init__()
        if 'prepared' in file_path:
            if purpose == 'valid':
                file_path = file_path.replace('.npz', '_valid.npz')
            self.images = np.load(file_path, allow_pickle=True)
            self.x = self.images['x']
            self.y = self.images['y']
        else:
            self.cell_size = cell_size
            # 此处导入的数据集已经为[0, 1]的浮点数
            self.images = np.load(file_path, allow_pickle=True)

            if purpose == 'train':
                self.x_ori = self.images['x_train']
                self.y_ori = self.images['y_train']
            elif purpose == 'valid':
                self.x_ori = self.images['x_valid']
                self.y_ori = self.images['y_valid']
            else:
                self.x_ori = self.images['x_test']
                self.y_ori = self.images['y_test']

            self.x = pad_img_with_id(
                self.x_ori,
                self.x_ori.shape[-2],
                self.x_ori.shape[-1],
                new_size=512,
            )[:, :, None, :, :]
            self.image_size = self.x.shape[-1]
            self.y = self.make_label()
            del self.x_ori
            del self.y_ori
        self.seq_length = self.x.shape[1]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        # seq_idx = torch.randint(self.seq_length - 1, ())
        # return self.x[index][seq_idx:seq_idx + 2], self.y[index][seq_idx:seq_idx + 2]
        return self.x[index], self.y[index]

    def make_label(self):
        labels = []
        for y_seq in self.y_ori:  # y_ori 是(N,T) ndarray
            seq_labels = []
            for ids, abs_coords in y_seq:  # y_seq 是(T,) ndarray，每个包含 tuple ( (n,) ndarray, (n,2) ndarray )
                cell_label = get_cell_label_with_id(ids, abs_coords, self.image_size, self.cell_size)
                seq_labels.append(cell_label)
            labels.append(np.array(seq_labels))
        return np.array(labels)


def _module_test1():
    pd = ParticleDetectorDataset(
        file_path='../data/synthesized_data_10_frames_per_file_150i200-1700_30n50_10d20_prepared.npz',
        purpose='train',
        cell_size=2 ** 4,
    )
    print(pd.x.shape)
    print(pd.y.shape)
    for idx, (img, label) in enumerate(pd):
        print(img.shape)
        print(label.shape)


if __name__ == '__main__':
    _module_test1()
