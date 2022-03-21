import numpy as np
import torch
from torch.nn import functional as F


# def bce_loss(x: torch.Tensor, y: torch.Tensor):
#     """
#     :param x: (N,H,W) 每一格中是否有粒子
#     :param y: (N,H,W) ground truth
#     :return: Tensor: ()
#     """
#     return torch.nn.functional.binary_cross_entropy(x.to(torch.float32), y.to(torch.float32))

def ce_loss(x: torch.Tensor, y: torch.Tensor):
    """
    :param x: (N,conf*4,H,W) 每一格中是否有粒子
    :param y: (N,conf*4,H,W) ground truth
    :return: Tensor: ()
    """
    return F.cross_entropy(x.to(torch.float32), y.to(torch.float32))


def soft_dice_loss(x: torch.Tensor, y: torch.Tensor, smooth=1.):
    """
    1 - 2 * (|x∩y| + smooth) / (|x| + |y| + smooth)
    :param x: (N,conf*4,H,W) 每一格中是否有粒子
    :param y: (N,conf*4,H,W) ground truth
    :param smooth: 1.
    :return: Tensor: ()
    """

    numerator = 2. * torch.sum(x * y, dim=(1, 2, 3))
    denominator = torch.sum(x, dim=(1, 2, 3)) + torch.sum(y, dim=(1, 2, 3))

    dice = (numerator + smooth) / (denominator + smooth)
    dice_loss = 1. - dice

    return torch.mean(dice_loss)


def rmse_loss(x: torch.Tensor, y: torch.Tensor, y_num: torch.Tensor):
    """
    sqrt(sum(|x-y|_2^2)/n)
    :param y_num: (N,conf*4,H,W)
    :param x: (N,(h,w)*3,H,W) 每一格中粒子的位置
    :param y: (N,(h,w)*3,H,W) ground truth
    :return: Tensor: ()
    """
    eps = torch.finfo(x.dtype).eps  # TODO: 这里的eps可能比较大

    # True则0，False则x，根据y中为0的位置设为0，也即不比较原本就不存在粒子的位置
    x_new = torch.where(torch.eq(y, y.new_tensor(0.)), x.new_tensor(0.), x)
    # y_new = torch.where(comparison, y.new_tensor(0.), y)
    y_new = y

    # 计算ground truth中的粒子总数
    num_truth = torch.sum(torch.argmax(y_num, dim=1))
    sum_square = torch.sum(torch.square(y_new - x_new), dim=(1, 2, 3))

    rmse = torch.sqrt(sum_square / (num_truth + eps))

    return torch.mean(rmse)


def f1_score(x: torch.Tensor, y: torch.Tensor):
    """
    TP / (TP + FN)
    TP / (TP + FP)
    :param x: (N,conf*4,H,W) 每一格中是否有粒子
    :param y: (N,conf*4,H,W) ground truth
    :return: (Tensor: (), Tensor: (), Tensor: ())
    """
    eps = torch.finfo(torch.float32).eps

    tp = torch.sum(x * y, dim=(1, 2))
    tp_add_fp = torch.sum(x, dim=(1, 2))
    tp_add_fn = torch.sum(y, dim=(1, 2))

    precision = tp / (tp_add_fp + eps)
    recall = tp / (tp_add_fn + eps)
    f1 = 2. * ((precision * recall) / (precision + recall + eps))
    return torch.mean(f1), torch.mean(recall), torch.mean(precision)


# def f1_loss(x: torch.Tensor, y: torch.Tensor):
#     """
#     :param x: (N,conf*4,H,W) 每一格中是否有粒子
#     :param y: (N,conf*4,H,W) ground truth
#     :return: Tensor: ()
#     """
#     x_new = torch.where(torch.argmax(x, dim=1) > 0, 1, 0)
#     y_new = torch.where(torch.argmax(y, dim=1) > 0, 1, 0)
#     f1, _, _ = f1_score(x, y)
#     return 1. - f1


def particle_detector_score(x: torch.Tensor, y: torch.Tensor):
    """
    :param x: (N,(conf*4,(h,w)*3),H,W) 每一格中是否有粒子以及粒子的位置
    :param y: (N,(conf*4,(h,w)*3),H,W) ground truth
    :return: (ndarray)
    """
    x_new = x.detach()
    y_new = y.detach()
    x_f1 = torch.where(torch.greater(torch.argmax(x_new[:, :4, :, :], dim=1), 0), 1, 0)
    y_f1 = torch.where(torch.greater(torch.argmax(y_new[:, :4, :, :], dim=1), 0), 1, 0)
    f1, recall, precision = f1_score(x_f1, y_f1)
    rmse = rmse_loss(x_new[:, 4:, :, :], y_new[:, 4:, :, :], y_new[:, :4, :, :])

    return (
        f1.cpu().numpy(),
        recall.cpu().numpy(),
        precision.cpu().numpy(),
        rmse.cpu().numpy()
    )


def focal_loss(x, y, class_num=4, gamma=2, alpha=None):
    """
    :param x: (N,conf*4,H,W) 每一格中是否有粒子
    :param y: (N,conf*4,H,W) ground truth(one hot)
    :param class_num:
    :param gamma:
    :param alpha:
    :return: Tensor: ()
    """
    one = x.new_tensor(1.)
    if alpha is None:
        alpha = np.ones(class_num)
    else:
        alpha = alpha
    alpha = x.new_tensor(alpha)
    y_ = torch.argmax(y[:, :4, :, :], dim=1)
    ids = y_.view(-1)
    alpha_list = alpha[ids.data]
    probs = torch.sum(x * y, dim=1).view(-1)
    f_loss = -alpha_list * ((one - probs) ** gamma) * torch.log(probs)
    return torch.mean(f_loss)


def particle_detector_loss(x: torch.Tensor, y: torch.Tensor, alpha=None):
    """
    soft_dice_loss + 2 * rmse_loss
    :param alpha:
    :param x: (N,(conf*4,(h,w)*3),H,W) 每一格中是否有粒子以及粒子的位置
    :param y: (N,(conf*4,(h,w)*3),H,W) ground truth
    :return: Tensor: ()
    """

    # ce_l = ce_loss(x[:, :4, :, :], y[:, :4, :, :])
    # f1_l = f1_loss(x[:, :4, :, :], y[:, :4, :, :])
    # soft_dice_l = soft_dice_loss(x[:, :4, :, :], y[:, :4, :, :])
    rmse_l = rmse_loss(x[:, 4:, :, :], y[:, 4:, :, :], y[:, :4, :, :])
    fl_l = focal_loss(x[:, :4, :, :], y[:, :4, :, :], alpha=alpha)

    return (fl_l + rmse_l.new_tensor(0.5) * rmse_l).requires_grad_(requires_grad=True)


def particle_matcher_loss(scores_list, x0_ids_list, x1_ids_list, loss_gamma=1.0):
    """
    :param scores_list:
    :param x0_ids_list:
    :param x1_ids_list:
    :return:
    """
    eps = torch.finfo(scores_list[0].dtype).eps
    pm_loss = torch.empty((len(scores_list),), dtype=torch.float32, device=scores_list[0].device)
    for idx, (scores, x0_ids, x1_ids) in enumerate(zip(scores_list, x0_ids_list, x1_ids_list)):
        gt_loc = torch.eq(*torch.meshgrid((x0_ids, x1_ids), indexing='ij'))
        gt_row_sum = torch.eq(torch.sum(gt_loc, dim=1), 0)[:, None]  # gt 中上一帧未能与当前帧对应的粒子
        gt_col_sum = torch.eq(torch.sum(gt_loc, dim=0), 0)[None]  # gt 中当前帧未能与上一帧对应的粒子
        gt_loc = torch.cat([
            torch.cat([gt_loc, gt_row_sum], dim=1),
            torch.cat([gt_col_sum, gt_loc.new_tensor(False)[None, None]], dim=1)
        ], dim=0)
        scores_gamma = torch.where(scores > loss_gamma, scores.new_tensor(1.), scores + eps)
        pm_loss_temp = - torch.log(scores_gamma) * gt_loc
        pm_loss_temp = torch.where(torch.isnan(pm_loss_temp), pm_loss_temp.new_tensor(0.), pm_loss_temp)
        pm_loss[idx] = torch.sum(pm_loss_temp)
    return torch.mean(pm_loss)


def particle_matcher_score(x0_ids_list, x1_ids_list, matches0_list):
    """
    :param x0_ids_list:
    :param x1_ids_list:
    :param matches0_list:
    :return:
    """
    eps = torch.finfo(torch.float32).eps
    tp = x0_ids_list[0].new_tensor(0, dtype=torch.long)
    fp = x0_ids_list[0].new_tensor(0, dtype=torch.long)
    fn = x0_ids_list[0].new_tensor(0, dtype=torch.long)

    for idx, (x0_ids, x1_ids, matches0) in enumerate(zip(
            x0_ids_list, x1_ids_list, matches0_list
    )):
        row_idx = torch.arange(0, len(matches0[0]))

        valid_match = torch.greater(matches0[0], -0.5)
        matches0_new = matches0[0][valid_match]
        row_idx_new = row_idx[valid_match]

        all_match = torch.sum(valid_match)
        gt_loc = torch.eq(*torch.meshgrid([x0_ids, x1_ids], indexing='ij'))
        gt_match = torch.sum(gt_loc)
        tp_temp = torch.sum(gt_loc[row_idx_new, matches0_new])
        fp_temp = all_match - tp_temp
        fn_temp = gt_match - tp_temp
        tp += tp_temp
        fp += fp_temp
        fn += fn_temp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2. * ((precision * recall) / (precision + recall + eps))

    return x0_ids_list[0].new_tensor([[f1, recall, precision]], dtype=torch.float32)
