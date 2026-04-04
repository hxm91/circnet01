""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class myconv2d(nn.Module):
    """
    自定义卷积层：在卷积前对输入进行 circular padding。
    参数说明：
      - pad: 用于 F.pad 的 padding 参数（4元组）。
      - padding: 传递给 nn.Conv2d 的 padding 参数。
    """
    def __init__(self, in_channels, out_channels, pad=(0, 0, 0, 0),
                 kernel_size=1, padding=(0, 0), stride=1, dilation=1,
                 bias=False, groups=1, device=None, dtype=None):
        super().__init__()
        self.pad = pad
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.device = device
        self.dtype = dtype
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size,
                              padding=self.padding, stride=self.stride,
                              dilation=self.dilation, bias=self.bias, groups=self.groups)

    def forward(self, x):
        # 先进行 circular padding，再做卷积
        x = F.pad(x, self.pad, mode='circular')
        return self.conv(x)


class myconvtranspose2d(nn.Module):
    """
    自定义反卷积层：在反卷积前对输入进行 circular padding。
    注意：修正了 dtype 参数的错误传递。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 pad=(0, 0, 0, 0), padding=(0, 0), output_padding=0,
                 groups=1, bias=True, dilation=1, padding_mode='zeros',
                 device=None, dtype=None):
        super().__init__()
        self.pad = pad
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        self.convtrans2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=self.kernel_size, stride=self.stride,
                                               padding=self.padding, output_padding=self.output_padding,
                                               groups=self.groups, bias=self.bias, dilation=self.dilation,
                                               padding_mode=self.padding_mode, device=self.device, dtype=self.dtype)

    def forward(self, x):
        x = F.pad(x, self.pad, mode='circular')
        return self.convtrans2d(x)


class RCNNBlock(nn.Module):
    """
    RCNNBlock：重复卷积模块
    构造方式：
      - 第1层：卷积、BN、LeakyReLU（输出记为 x1）。
      - 后续4层：先卷积（以及 BN ），然后将结果与 x1 相加。
    使用 ModuleList 简化重复层的构建。
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            myconv2d(in_channel, out_channel, pad=(1, 1, 0, 0),
                     kernel_size=3, stride=1, padding=(1, 0)),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True)
        )
        # 后续重复层（共4层）
        self.res_convs = nn.ModuleList([
            nn.Sequential(
                myconv2d(out_channel, out_channel, pad=(1, 1, 0, 0),
                         kernel_size=3, stride=1, padding=(1, 0)),
                nn.BatchNorm2d(out_channel)
            )
            for _ in range(4)
        ])

    def forward(self, x):
        x1 = self.conv1(x)
        x_n = x1
        for conv in self.res_convs:
            # 每层输出加上初始 x1 的残差
            x_n = x1 + conv(x_n)
        return x_n


class Up(nn.Module):
    """
    上采样模块：先对输入进行上采样，再与 skip 分支特征拼接，最后经过 RCNNBlock 进行处理。
    参数：
      - bilinear: 若为 True，则采用 nn.Upsample，否则使用自定义反卷积。
      - up_W: 控制反卷积的参数配置（宽度方向是否下采样）。
    """
    def __init__(self, in_channels, out_channels, bilinear=False, dropout=False, up_W=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 此处 pad 与 padding 参数的设置与上采样需求相关
            if up_W:
                self.up = myconvtranspose2d(in_channels, out_channels,
                                            kernel_size=4, stride=2,
                                            pad=(1, 1, 0, 0), padding=(1, 3))
            else:
                self.up = myconvtranspose2d(in_channels, out_channels,
                                            kernel_size=[1, 4], stride=[1, 2],
                                            pad=(1, 1, 0, 0), padding=[0, 3])
        # 拼接后通道数翻倍，因此 RCNNBlock 的输入通道为 out_channels*2
        self.conv = RCNNBlock(out_channels * 2, out_channels)
        self.dropout = nn.Dropout2d(p=0.2) if dropout else None

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 拼接时要求两个特征图尺寸一致（可能需要额外裁剪）
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Decoder(nn.Module):
    """
    Decoder 部分：根据跳跃连接（skips）逐级上采样恢复空间分辨率。
    目前构造固定4级上采样，每级均采用 RCNNBlock 进行特征融合。
    """
    def __init__(self, params, stub_skips=None, OS=32, feature_depth=1024, bilinear=False):
        super().__init__()
        self.bilinear = bilinear

        dim = 32
        self.up1 = Up(dim * 16, dim * 8, bilinear, dropout=True, up_W=True)
        self.up2 = Up(dim * 8, dim * 4, bilinear, dropout=True, up_W=True)
        self.up3 = Up(dim * 4, dim * 2, bilinear, dropout=True, up_W=True)
        self.up4 = Up(dim * 2, dim, bilinear, up_W=True)

        # 最后输出的通道数
        self.last_channels = dim

        print("Using RCNN decoder")

    def forward(self, x5, skips):
        # skips 应为一个包含4个元素的列表：[x4, x3, x2, x1]
        [x4, x3, x2, x1] = skips

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    def get_last_depth(self):
        return self.last_channels
