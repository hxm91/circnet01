#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# add pdding_circular, 20241010

import numpy as np
from scipy import signal  # 若无实际用途，可考虑删除
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocallyConnectedXYZLayer(nn.Module):
    """
    局部连接层（LocallyConnectedXYZLayer）
    利用给定窗口内 (x,y,z) 坐标之间的欧氏距离，通过高斯加权对 softmax 进行重新加权，
    其中宽度方向采用 circular padding，而高度方向采用 unfold 内置 padding（零填充）。
    
    参数：
      - h, w: 窗口尺寸（必须为奇数，保证中心对称）。
      - sigma: 用于高斯加权的标准差，内部计算时使用 gauss_den = 2*sigma^2 。
      - nclasses: 类别数，与 softmax 通道数一致。
    """
    def __init__(self, h, w, sigma, nclasses):
        super().__init__()
        self.h = h
        self.w = w
        # 为保证窗口中心对称，要求 h 和 w 必须为奇数
        assert (self.h % 2 == 1 and self.w % 2 == 1), "Window size must be odd."
        # 这里对宽度方向做 circular padding
        self.padh = h // 2
        self.padw = w // 2
        self.sigma = sigma
        self.gauss_den = 2 * (self.sigma ** 2)
        self.nclasses = nclasses

    def forward(self, xyz, softmax, mask):
        """
        参数：
          - xyz: Tensor，尺寸 (N, 3, H, W)，包含 x, y, z 信息
          - softmax: Tensor，尺寸 (N, nclasses, H, W)
          - mask: Tensor，尺寸 (N, H, W)，有效区域为 1，无效区域为 0
        返回：
          - 重新加权后的 softmax，尺寸 (N, nclasses, H, W)
        """
        N, C, H, W = softmax.shape

        # 将 mask 扩展到通道维，并将无效区域置零
        softmax = softmax * mask.unsqueeze(1).float()

        # 提取 x,y,z 分量，保持通道维度为1
        x = xyz[:, 0:1, :, :]  # (N,1,H,W)
        y = xyz[:, 1:2, :, :]
        z = xyz[:, 2:3, :, :]

        # --- 利用 unfold 提取局部窗口内的像素值 ---
        # 注意：采用 circular padding 仅在宽度方向，采用 unfold 内置 padding 对高度进行零填充
        # 对 x 分量：
        x_padded = F.pad(x, (self.padw, self.padw, 0, 0), mode='circular')
        # unfold 时对高度再额外填充 self.padh（宽度已在 F.pad 中处理）
        window_x = F.unfold(x_padded, kernel_size=(self.h, self.w), padding=(self.padh, 0))
        center_x = F.unfold(x, kernel_size=(1, 1), padding=(0, 0))
        
        # 对 y 分量：
        y_padded = F.pad(y, (self.padw, self.padw, 0, 0), mode='circular')
        window_y = F.unfold(y_padded, kernel_size=(self.h, self.w), padding=(self.padh, 0))
        center_y = F.unfold(y, kernel_size=(1, 1), padding=(0, 0))
        
        # 对 z 分量：
        z_padded = F.pad(z, (self.padw, self.padw, 0, 0), mode='circular')
        window_z = F.unfold(z_padded, kernel_size=(self.h, self.w), padding=(self.padh, 0))
        center_z = F.unfold(z, kernel_size=(1, 1), padding=(0, 0))
        
        # --- 计算局部窗口内每个位置与中心点的欧氏距离平方 ---
        # 这里 center_* 的尺寸为 (N, 1, L)；window_* 尺寸为 (N, h*w, L)，L = H*W（滑动窗口数）
        unravel_dist2 = (window_x - center_x) ** 2 + \
                        (window_y - center_y) ** 2 + \
                        (window_z - center_z) ** 2
        
        # 计算高斯权重（局部影响权重），尺寸 (N, h*w, L)
        unravel_gaussian = torch.exp(- unravel_dist2 / self.gauss_den)
        
        # --- 对 softmax 做相同的局部展开，并利用高斯权重加权求和 ---
        # softmax 尺寸 (N, nclasses, H, W)
        # 对 softmax 进行 circular padding（宽度方向）
        softmax_padded = F.pad(softmax, (self.padw, self.padw, 0, 0), mode='circular')
        # unfold 得到尺寸 (N, nclasses * (h*w), L)
        unfolded_softmax = F.unfold(softmax_padded, kernel_size=(self.h, self.w), padding=(self.padh, 0))
        # 重塑为 (N, nclasses, h*w, L)
        unfolded_softmax = unfolded_softmax.view(N, self.nclasses, self.h * self.w, -1)
        
        # 扩展高斯权重维度为 (N, 1, h*w, L)
        gauss_weight = unravel_gaussian.unsqueeze(1)
        # 加权 softmax
        weighted_softmax = unfolded_softmax * gauss_weight
        # 对局部窗口内的加权和求和，得到 (N, nclasses, L)
        added_softmax = weighted_softmax.sum(dim=2)
        # 重塑回 (N, nclasses, H, W)
        output_softmax = added_softmax.view(N, self.nclasses, H, W)
        
        return output_softmax


class CRF(nn.Module):
    """
    条件随机场（CRF）模块：
      利用局部信息对 softmax 预测进行迭代修正。
      1. 首先利用局部连接层（LocallyConnectedXYZLayer）对基于 xyz 坐标的消息传递进行加权；
      2. 通过 1x1 卷积（compat_conv）进行兼容性变换；
      3. 与原 softmax 相加后，重新归一化。
    """
    def __init__(self, params, nclasses):
        super().__init__()
        self.params = params
        # 直接将迭代次数保存为整数
        self.iter = int(params["iter"])
        # 保存局部连接层尺寸（h, w），无需定义为 Parameter
        self.lcn_size = (params["lcn_size"]["h"], params["lcn_size"]["w"])
        # 保存 CRF 参数，注意：xyz_sigma 用于局部加权，高斯计算中使用
        self.xyz_coef = torch.tensor(params["xyz_coef"], dtype=torch.float32)
        self.xyz_sigma = torch.tensor(params["xyz_sigma"], dtype=torch.float32)

        self.nclasses = nclasses
        print("Using CRF!")

        # 初始化兼容性矩阵（排除对角线）
        self.compat_kernel_init = np.reshape(
            np.ones((self.nclasses, self.nclasses)) - np.identity(self.nclasses),
            [self.nclasses, self.nclasses, 1, 1]
        )
        # 1x1 卷积用于兼容性变换，初始权重为 compat_kernel_init * xyz_coef
        self.compat_conv = nn.Conv2d(self.nclasses, self.nclasses, kernel_size=1)
        self.compat_conv.weight = torch.nn.Parameter(
            torch.from_numpy(self.compat_kernel_init).float() * self.xyz_coef,
            requires_grad=True
        )

        # 局部连接层用于基于 xyz 坐标进行消息传递
        # 注意：这里传入的是局部连接层尺寸和 sigma（使用 xyz_sigma 而非 xyz_coef）
        self.local_conn_xyz = LocallyConnectedXYZLayer(
            self.lcn_size[0],
            self.lcn_size[1],
            sigma=self.xyz_sigma,
            nclasses=self.nclasses
        )

    def forward(self, input, softmax, mask):
        """
        参数：
          - input: Tensor，尺寸 (N, C, H, W)，其中第 1:4 通道为 xyz 信息
          - softmax: Tensor，尺寸 (N, nclasses, H, W)
          - mask: Tensor，尺寸 (N, H, W)
        迭代更新 softmax，并返回最终归一化后的结果。
        """
        # 取出 xyz 信息（假设 input 的通道顺序中，通道 1~3 为 x, y, z）
        xyz = input[:, 1:4, :, :]

        for i in range(self.iter):
            # 消息传递：利用局部连接层基于 xyz 信息对 softmax 进行局部加权重构
            locally_connected = self.local_conn_xyz(xyz, softmax, mask)
            # 通过 1x1 卷积对消息进行兼容性变换
            reweight_softmax = self.compat_conv(locally_connected)
            # 将更新后的消息与原 softmax 相加，并重新归一化
            softmax = F.softmax(reweight_softmax + softmax, dim=1)

        return softmax
