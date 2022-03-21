import scipy.io as scio


def mat_read(file_path):
    mat_data = scio.loadmat(file_path, struct_as_record=True)
    return mat_data


if __name__ == '__main__':
    data = mat_read("../data/test_2022_01_23__21_36_07.mat")
    pass
