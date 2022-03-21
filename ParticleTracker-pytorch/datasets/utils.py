import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt


def pad_img(x_ori, ori_image_h, ori_image_w, new_size=512):
    """
    :param x_ori: (N,H,W)
    :param ori_image_h:
    :param ori_image_w:
    :param new_size:
    :return:
    """
    if ori_image_h > new_size or ori_image_w > new_size:
        raise ValueError(f"img h:{ori_image_h} or w:{ori_image_w} > new size:{new_size}")
    median = np.median(x_ori)
    return np.pad(
        x_ori,
        ((0, 0), (0, new_size - ori_image_h), (0, new_size - ori_image_w)),
        'constant',
        constant_values=(median, median),
    ).astype(np.float32)


def pad_img_with_id(x_ori, ori_image_h, ori_image_w, new_size=512):
    """
    :param x_ori: (N,T,H,W)
    :param ori_image_h:
    :param ori_image_w:
    :param new_size:
    :return:
    """
    if ori_image_h > new_size or ori_image_w > new_size:
        raise ValueError(f"img h:{ori_image_h} or w:{ori_image_w} > new size:{new_size}")
    median = np.median(x_ori)
    return np.pad(
        x_ori,
        ((0, 0), (0, 0), (0, new_size - ori_image_h), (0, new_size - ori_image_w)),
        'constant',
        constant_values=(median, median),
    ).astype(np.float32)


def get_cell_label(
        abs_coords, image_size, cell_size, max_num_in_cell=3
):
    """

    :param max_num_in_cell:
    :param abs_coords: (n_particles, 2)
    :param image_size: int  图像为正方形，且为2的幂
    :param cell_size: int  2 ** k
    :return:
    """

    num_h = image_size // cell_size
    if num_h * cell_size != image_size:
        raise ValueError(f'Bad args: image_size={image_size}, cell_size={cell_size}')
    num_w = num_h

    cell_label = np.zeros((1 + 3 * max_num_in_cell, num_h, num_w))  # 有 0,1,2,3 个粒子，再加 3 个h, w坐标, 共 10 个
    cell_label[0, :, :] = 1.
    for abs_h, abs_w in abs_coords:

        # 计算当前粒子位于哪一个 cell
        cell_h = int(abs_h // cell_size)
        cell_w = int(abs_w // cell_size)
        if cell_h >= num_h or cell_w >= num_w:
            raise ValueError('abs_coords out of cell range')

        # cell 内的相对坐标（即绝对坐标减去 cell 左上角坐标，再除以 cell_size）
        rel_h = (abs_h - cell_h * cell_size) / cell_size
        rel_w = (abs_w - cell_w * cell_size) / cell_size

        # Assign values along prediction matrix dimension 3
        if cell_label[0, cell_h, cell_w] == 1.:
            cell_label[0:2, cell_h, cell_w] = 0., 1.
            cell_label[max_num_in_cell + 1:max_num_in_cell + 3, cell_h, cell_w] = rel_h, rel_w
        elif cell_label[1, cell_h, cell_w] == 1.:
            cell_label[1:3, cell_h, cell_w] = 0., 1.
            cell_label[max_num_in_cell + 3:max_num_in_cell + 5, cell_h, cell_w] = rel_h, rel_w
        elif cell_label[2, cell_h, cell_w] == 1.:
            cell_label[2:4, cell_h, cell_w] = 0., 1.
            cell_label[max_num_in_cell + 5:max_num_in_cell + 7, cell_h, cell_w] = rel_h, rel_w

    return cell_label.astype(np.float32)


def get_cell_label_with_id(
        ids, abs_coords, image_size, cell_size, max_num_in_cell=3
):
    """
    :return:
    :param ids: (n_particles,)
    :param abs_coords: (n_particles, 2)
    :param image_size: int  图像为正方形，且为2的幂
    :param cell_size: int  2 ** k
    :param max_num_in_cell:
    :return:
    """

    num_h = image_size // cell_size
    if num_h * cell_size != image_size:
        raise ValueError(f'Bad args: image_size={image_size}, cell_size={cell_size}')
    num_w = num_h

    cell_label = np.zeros((1 + 4 * max_num_in_cell, num_h, num_w))  # 有 0,1,2,3 个粒子，再加3个h, w坐标,3个ID, 共13个
    cell_label[0, :, :] = 1.
    cell_label[-max_num_in_cell:, :, :] = -1.
    for particle_id, (abs_h, abs_w) in zip(ids, abs_coords):

        # 计算当前粒子位于哪一个 cell
        cell_h = int(abs_h // cell_size)
        cell_w = int(abs_w // cell_size)
        if cell_h >= num_h or cell_w >= num_w:
            raise ValueError('abs_coords out of cell range')

        # cell 内的相对坐标（即绝对坐标减去 cell 左上角坐标，再除以 cell_size）
        rel_h = (abs_h - cell_h * cell_size) / cell_size
        rel_w = (abs_w - cell_w * cell_size) / cell_size

        # Assign values along prediction matrix dimension 3
        if cell_label[0, cell_h, cell_w] == 1.:
            cell_label[0:2, cell_h, cell_w] = 0., 1.
            cell_label[max_num_in_cell + 1:max_num_in_cell + 3, cell_h, cell_w] = rel_h, rel_w
            cell_label[-3, cell_h, cell_w] = particle_id
        elif cell_label[1, cell_h, cell_w] == 1.:
            cell_label[1:3, cell_h, cell_w] = 0., 1.
            cell_label[max_num_in_cell + 3:max_num_in_cell + 5, cell_h, cell_w] = rel_h, rel_w
            cell_label[-2, cell_h, cell_w] = particle_id
        elif cell_label[2, cell_h, cell_w] == 1.:
            cell_label[2:4, cell_h, cell_w] = 0., 1.
            cell_label[max_num_in_cell + 5:max_num_in_cell + 7, cell_h, cell_w] = rel_h, rel_w
            cell_label[-1, cell_h, cell_w] = particle_id

    return cell_label.astype(np.float32)


def get_abs_coords(pred, cell_size, max_num_in_cell):
    abs_coords_list = [np.empty((0, 2), dtype=np.float32),
                       np.empty((0, 4), dtype=np.float32),
                       np.empty((0, 6), dtype=np.float32)]
    cell_h = pred.shape[1]
    cell_w = pred.shape[2]
    if pred.shape[0] != 1 + 3 * max_num_in_cell:  # pred.shape[0] = 1 + 3 * max_num_in_cell
        raise ValueError(f'pred.shape[0]: {pred.shape[0]} & max_num_cell: {max_num_in_cell} * 3 + 1 not match!')

    cell_grid_h, cell_grid_w = np.meshgrid(np.arange(0, cell_h), np.arange(0, cell_w), indexing='ij')
    cell_grid = np.concatenate([cell_grid_h[None], cell_grid_w[None]] * 3, axis=0)
    abs_coords = cell_size * (pred[max_num_in_cell + 1:, :, :] + cell_grid)

    for h in range(cell_h):
        for w in range(cell_w):
            num_in_cell = np.argmax(pred[:max_num_in_cell + 1, h, w])
            if num_in_cell > 0:
                abs_coords_list[num_in_cell - 1] = np.concatenate(
                    [abs_coords_list[num_in_cell - 1], abs_coords[:2 * num_in_cell, h, w][None]], axis=0)
    return abs_coords_list


def particle_detector_paint(imgs, preds, max_num_in_cell=3, circle_scale=1.0, fig_title='No title'):
    for img, pred in zip(imgs, preds):
        img_size = img.shape[-1]
        cell_size = int(img_size / pred.shape[1])
        abs_coords_list = get_abs_coords(pred, cell_size, max_num_in_cell=max_num_in_cell)

        img_point = draw_bounding_circle(img[0], abs_coords_list, cell_size, circle_scale)
        plt.title(fig_title)
        plt.imshow(img_point)
        plt.show()


def particle_matcher_paint(img_pair, coords_list, matches, max_num_in_cell=3, circle_scale=1.0,
                           fig_title='No title'):
    # img_size = img_pair.shape[-1]
    cell_size = 2 ** 4
    img0_point = draw_bounding_circle(img_pair[0], [coords_list[0]], cell_size, circle_scale)
    img1_point = draw_bounding_circle(img_pair[1], [coords_list[1]], cell_size, circle_scale)
    fig = draw_img_pair(img0_point, img1_point, coords_list, matches)
    fig.suptitle(fig_title)
    fig.show()
    pass


def draw_img_pair(img0_point, img1_point, coords_list, matches):
    from matplotlib.patches import ConnectionPatch
    valid_match = matches > -0.5
    match0_coords = coords_list[0][valid_match]
    match1_coords = coords_list[1][matches[valid_match]]

    fig = plt.figure(figsize=[18, 9], dpi=120)
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    ax0.axis('off')
    ax1.axis('off')
    ax0.imshow(img0_point)
    ax1.imshow(img1_point)
    for idx in range(match0_coords.shape[0]):
        xy0 = [match0_coords[idx, 1], match0_coords[idx, 0]]
        xy1 = [match1_coords[idx, 1], match1_coords[idx, 0]]
        # if 250 < match1_coords[idx, 0] < 290 and 250 < match1_coords[idx, 1] < 280:
        #     con = ConnectionPatch(xyA=xy0, xyB=xy1, coordsA="data", coordsB="data",
        #                           axesA=ax0, axesB=ax1, color="yellow")
        # elif 250 < match1_coords[idx, 0] < 290 and 200 < match1_coords[idx, 1] < 250:
        #     con = ConnectionPatch(xyA=xy0, xyB=xy1, coordsA="data", coordsB="data",
        #                           axesA=ax0, axesB=ax1, color="magenta")
        # else:
        #     con = None
        con = ConnectionPatch(xyA=xy0, xyB=xy1, coordsA="data", coordsB="data",
                              axesA=ax0, axesB=ax1, color="cyan")
        if con is not None:
            ax1.add_artist(con)
    return fig


def draw_bounding_circle(img_gray, abs_coords_list, cell_size, circle_scale=1.0):
    img_cv2 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    # 画矩形框
    for num_in_cell, abs_coords in enumerate(abs_coords_list, start=1):
        for abs_coord in abs_coords:
            if num_in_cell == 1:
                cv2.circle(
                    img_cv2,
                    np.floor(abs_coord[::-1]).astype(np.int32),
                    # 用floor因为最左上角是(0, 0)  abs_coord[::-1] (h, w) --> (x, y)
                    np.round(cell_size * circle_scale / 2.).astype(np.int32),
                    (1., 0., 0.),  # color: RGB for matplotlib
                    thickness=num_in_cell,
                )
            elif num_in_cell == 2:
                abs_coord_temp = np.mean(abs_coord.reshape(2, -1), axis=0)
                cv2.circle(
                    img_cv2,
                    np.floor(abs_coord_temp[::-1]).astype(np.int32),
                    # 用floor因为最左上角是(0, 0)  abs_coord[::-1] (h, w) --> (x, y)
                    np.round(cell_size * circle_scale / 2.).astype(np.int32),
                    (1., 0., 0.),  # color: RGB for matplotlib
                    thickness=num_in_cell,
                )
                cv2_green_dot(img_cv2, np.floor(abs_coord[1::-1]).astype(np.int32))
                cv2_green_dot(img_cv2, np.floor(abs_coord[3:1:-1]).astype(np.int32))
            else:
                abs_coord_temp = np.mean(abs_coord.reshape(3, -1), axis=0)
                cv2.circle(
                    img_cv2,
                    np.floor(abs_coord_temp[::-1]).astype(np.int32),
                    # 用floor因为最左上角是(0, 0)  abs_coord[::-1] (h, w) --> (x, y)
                    np.round(cell_size * circle_scale / 2.).astype(np.int32),
                    (1., 0., 0.),  # color: RGB for matplotlib
                )
                cv2_green_dot(img_cv2, np.floor(abs_coord[1::-1]).astype(np.int32))
                cv2_green_dot(img_cv2, np.floor(abs_coord[3:1:-1]).astype(np.int32))
                cv2_green_dot(img_cv2, np.floor(abs_coord[5:3:-1]).astype(np.int32))
    return img_cv2


def cv2_green_dot(img_cv2, center):
    cv2.circle(
        img_cv2,
        center,
        1,
        (0., 1., 0.),  # color: RGB for matplotlib
    )


def particle_matcher_seq_paint(img_seq,
                               coords_list,
                               matches_list,
                               max_num_in_cell=3,
                               circle_scale=1.0,
                               d_threshold=30,
                               fig_title='No title'):
    img_size = img_seq.shape[-1]
    cell_size = 2 ** 4
    trajectory_cv2 = draw_trajectory(img_size, coords_list, matches_list, d_threshold)

    for idx, img in enumerate(img_seq[0, :, 0].detach().cpu().numpy()):
        plt.figure()
        img_point = draw_bounding_circle(img, [coords_list[idx][0].detach().cpu().numpy()], cell_size, circle_scale)
        plt.imshow(img_point)
        plt.imshow(trajectory_cv2)
        plt.title(fig_title + '\n max_threshold=' + str(d_threshold))
        plt.savefig('./eval_results/seq/' + str(idx) + '.png')
    pass


def draw_trajectory(img_size, coords_list, matches_list, d_threshold):
    trajectory_cv2 = np.zeros((img_size, img_size, 4), dtype=np.float)
    for idx, (previous_frame_coords,
              now_frame_coords,
              previous_frame_matches) in enumerate(zip(coords_list[:-1], coords_list[1:], matches_list)):
        coords_list0 = previous_frame_coords[0].detach().cpu().numpy()
        coords_list1 = now_frame_coords[0].detach().cpu().numpy()
        matches = previous_frame_matches[0][0].detach().cpu().numpy()
        valid_match = matches > -0.5
        match0_coords = np.floor(coords_list0[valid_match]).astype(np.int32)
        match1_coords = np.floor(coords_list1[matches[valid_match]]).astype(np.int32)
        # RGB alpha
        green = (0., 1., 0., 1.)
        # green_light = (0., 1., 0., 0.5)
        for match0_coord, match1_coord in zip(match0_coords, match1_coords):
            if np.linalg.norm(match0_coord - match1_coord) < d_threshold:
                cv2.line(trajectory_cv2, match0_coord[::-1], match1_coord[::-1], green, 2)
    return trajectory_cv2


def _module_test():
    pass


if __name__ == '__main__':
    _module_test()
