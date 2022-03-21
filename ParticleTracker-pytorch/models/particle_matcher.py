import torch
from torch import nn

from models.utils import AttentionalGNN, log_optimal_transport


class MatchScoreDealer(nn.Module):

    def __init__(self, match_threshold=0.0) -> None:
        super().__init__()
        self.match_threshold = match_threshold

    def forward(self, scores_list):
        matches0_list = []
        # matches1_list = []

        for scores in scores_list:
            # 每行最大值max0（1, 行数:上一时刻的点数+1）,每列最大值max1（1, 列数:当前时刻的点数+1）
            # p0_num = scores.shape[-2] - 1  # 上一帧中的粒子数量
            p1_num = scores.shape[-1] - 1  # 当前帧中的粒子数量

            max0, max1 = scores.max(dim=2), scores.max(dim=1)

            # matches0即上一时刻中的每个点对应 当前时刻中的点 概率最大的序号
            # matches1即当前时刻中的每个点对应 上一时刻中的点 概率最大的序号
            matches0, matches1 = max0.indices, max1.indices

            # mutual0(1) 即判断某一行(列)的最大值是否也为该列(行)的最大值，两者各自True的个数应相等
            mutual0 = torch.eq(
                torch.arange(matches0.shape[1], device=matches0.device)[None],  # 第 i 行的行号 i
                matches1.gather(dim=1, index=matches0)  # 第 i 行的最大值对应于列号 j ，则第 j 列的最大值是否对应行号 i
            )
            # mutual1 = torch.eq(
            #     torch.arange(matches1.shape[1], device=matches1.device)[None],  # 第 i 列的列号 i
            #     matches0.gather(dim=1, index=matches1)  # 第 i 列的最大值对应于行号 j ，则第 j 行的最大值是否对应列号 i
            # )

            # 对于上一时刻中的点，满足mutual0的位置填入其scores，其余为0
            scores0 = torch.where(mutual0, max0.values, scores.new_tensor(0.))[:, :-1]  # 去除最后一列(bin)所对应的数据
            # scores1 = torch.where(mutual1, max1.values, scores.new_tensor(0.))[:, :-1]  # 去除最后一行(bin)所对应的数据
            matches0 = matches0[:, :-1]
            # matches1 = matches1[:, :-1]

            # 判断 scores0(1) 是否满足门限值，同时将匹配到 bin 的结果也设为 invalid
            valid0 = torch.bitwise_and(torch.greater(scores0, self.match_threshold), matches0 <= p1_num)
            # valid1 = torch.bitwise_and(torch.greater(scores1, self.match_threshold), matches1 <= p0_num)

            # 根据门限值的判断，得出最后的匹配结果，未匹配的点置为-1
            matches0 = torch.where(valid0, matches0, matches0.new_tensor(-1))
            # matches1 = torch.where(valid1, matches1, matches1.new_tensor(-1))

            matches0_list.append(matches0)
            # matches1_list.append(matches1)

        # return matches0_list, matches1_list
        return matches0_list


class ParticleMatcher(nn.Module):
    def __init__(self, desc_pow, gnn_layer_pairs, sinkhorn_iters) -> None:
        super().__init__()
        self.desc_channels = 2 ** desc_pow
        self.sinkhorn_iters = sinkhorn_iters
        self.attentional_gnn = AttentionalGNN(self.desc_channels, gnn_layer_pairs)
        self.matcher_final_mlp = nn.Conv1d(self.desc_channels, self.desc_channels, kernel_size=(1,), bias=True)
        self.register_parameter('bin_score', torch.nn.Parameter(torch.tensor(1.)))

    def forward(self, x0_desc_list, x1_desc_list):
        scores_list = []

        for x0_desc, x1_desc in zip(x0_desc_list, x1_desc_list):
            if x0_desc.shape[2] == 0 or x1_desc.shape[2] == 0:  # if no particles were detected
                scores = x0_desc.new_full((1, x0_desc.shape[2] + 1, x1_desc.shape[2] + 1), 1., dtype=torch.float32)
                scores[-1, -1, -1] = 0.
                scores_list.append(scores)
                continue

            x0_desc, x1_desc = self.attentional_gnn(x0_desc, x1_desc)
            x0_desc, x1_desc = self.matcher_final_mlp(x0_desc), self.matcher_final_mlp(x1_desc)

            scores = torch.einsum('bdn,bdm->bnm', x0_desc, x1_desc)
            scores = scores / (self.desc_channels ** 0.5)

            scores = log_optimal_transport(scores, self.bin_score, iters=self.sinkhorn_iters)
            scores = torch.exp(scores)

            scores_list.append(scores)

        return scores_list
