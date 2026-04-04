#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# mode='circular'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import __init__ as booger

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # 创建一个 (kernel_size, kernel_size, 2) 的坐标网格
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # 计算二维高斯核：每个元素对应 (1/(2πvariance)) * exp(-||xy - mean||²/(2*variance))
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
        torch.exp(-torch.sum((xy_grid - mean) ** 2, dim=-1) / (2 * variance))

    # 归一化：确保所有值的和为1
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size)
    return gaussian_kernel

class KNN(nn.Module):
    def __init__(self, params, nclasses):
        super().__init__()
        print("*" * 80)
        print("Cleaning point-clouds with kNN post-processing")
        self.knn = params["knn"]
        self.search = params["search"]
        self.sigma = params["sigma"]
        self.cutoff = params["cutoff"]
        self.nclasses = nclasses
        print("kNN parameters:")
        print("knn:", self.knn)
        print("search:", self.search)
        print("sigma:", self.sigma)
        print("cutoff:", self.cutoff)
        print("nclasses:", self.nclasses)
        print("*" * 80)

    def forward(self, proj_range, unproj_range, proj_argmax, px, py):
        """
        proj_range: [H, W]  投影图（例如距离图）
        unproj_range: [num_points]  对应点云中每个点的距离
        proj_argmax: [H, W]  投影图中每个像素的预测类别
        px, py: 分别为点云在投影图中的 x 和 y 坐标（均为一维索引向量）
        
        注意：仅适用于非 batch 点云；若数据是 batched，则需要对每个 batch 分开处理。
        """
        # 直接使用 proj_range 的设备
        device = proj_range.device

        # 原始投影图尺寸
        H, W = proj_range.shape
        # 点云数目
        num_points = unproj_range.shape[0]

        # 检查 kernel 尺寸是否为奇数
        if (self.search % 2 == 0):
            raise ValueError("Nearest neighbor kernel must be an odd number")

        pad = (self.search - 1) // 2

        # 对投影图进行 circular padding（仅对宽度）并展开邻域
        proj_range_padded = F.pad(proj_range.unsqueeze(0).unsqueeze(0), (pad, pad, 0, 0), mode='circular')
        # 使用 unfold 展开窗口，其中 height 的 padding 由 unfold 参数提供
        proj_unfold = F.unfold(proj_range_padded, kernel_size=(self.search, self.search), padding=(pad, 0))
        # 得到的 unfolded 张量形状为 [1, search*search, H*W]
        # 根据 px,py 计算每个点对应的平面索引
        idx_list = (py * W + px).long()  # shape: [num_points]
        # 取出对应点的邻域数据，结果形状 [1, search*search, num_points]
        unproj_neighborhood = proj_unfold[:, :, idx_list]

        # 将无效（小于0）的距离置为无穷大，避免干扰最近邻搜索
        unproj_neighborhood[unproj_neighborhood < 0] = float("inf")

        # 用真实的 unproj_range 替换窗口中心点
        center_idx = (self.search * self.search - 1) // 2
        unproj_neighborhood[:, center_idx, :] = unproj_range.unsqueeze(0)

        # 计算邻域中每个候选点与真实值的绝对差值
        k2_distances = torch.abs(unproj_neighborhood - unproj_range.unsqueeze(0))

        # 构造高斯权重（1 - 高斯核），使得 (x,y) 距离越近的邻域权重越高
        inv_gauss = (1 - get_gaussian_kernel(self.search, self.sigma, 1)).view(1, -1, 1)
        inv_gauss = inv_gauss.to(device).to(proj_range.dtype)
        k2_distances = k2_distances * inv_gauss

        # 选择距离最小的 k 个邻域（即最近邻）
        _, knn_idx = k2_distances.topk(self.knn, dim=1, largest=False, sorted=False)

        # 同样对预测类别图进行 unfold 操作
        proj_argmax_padded = F.pad(proj_argmax.unsqueeze(0).unsqueeze(0).float(), (pad, pad, 0, 0), mode='circular')
        proj_argmax_unfold = F.unfold(proj_argmax_padded, kernel_size=(self.search, self.search), padding=(pad, 0)).long()
        argmax_neighborhood = proj_argmax_unfold[:, :, idx_list]

        # 根据 knn_idx 提取每个点的最近邻预测类别，结果形状 [1, knn, num_points]
        knn_argmax = torch.gather(argmax_neighborhood, dim=1, index=knn_idx)

        # 对于距离超过 cutoff 的邻域，将其预测类别置为无效（设为 nclasses，假定该值不在正常类别范围内）
        if self.cutoff > 0:
            knn_distances = torch.gather(k2_distances, dim=1, index=knn_idx)
            knn_invalid = knn_distances > self.cutoff
            knn_argmax[knn_invalid] = self.nclasses

        # 投票表决：对每个点，对 knn 个邻域进行 one-hot 累加投票
        # 创建 one-hot 张量，注意多分配一个通道用于无效类别（下标为 nclasses）
        knn_votes = torch.zeros((1, self.nclasses + 1, num_points), device=device, dtype=proj_range.dtype)
        ones = torch.ones_like(knn_argmax, dtype=proj_range.dtype)
        knn_votes = knn_votes.scatter_add_(1, knn_argmax, ones)

        # 在投票结果中忽略第一个和最后一个通道（分别对应 unlabeled 和无效），选择票数最多的类别
        knn_out = knn_votes[:, 1:-1].argmax(dim=1) + 1  # 加1后得到真实类别标签

        # 将结果恢复成与输入点云对应的一维向量
        knn_out = knn_out.view(num_points)
        return knn_out
