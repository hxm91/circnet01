import torch
import torch.nn as nn
import torch.nn.functional as F

# 如果 ContextBlock 没有用到，可以去掉以下导入
from common.context_block import ContextBlock

'''
this file is a optimiert version
'''
class myconv2d(nn.Module):
    """
    自定义卷积层：在卷积前使用 circular padding。
    """
    def __init__(self, in_channels, out_channels, pad=(0,0,0,0), kernel_size=1, padding=(0,0),
                 stride=1, dilation=1, bias=False, groups=1, device=None, dtype=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding  # 用于 nn.Conv2d 的 padding 参数
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.device = device
        self.dtype = dtype
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size,
                              padding=self.padding, stride=self.stride, dilation=self.dilation,
                              bias=self.bias, groups=self.groups)
        self.pad = pad

    def forward(self, x):
        # 使用 circular padding
        x = F.pad(x, self.pad, mode='circular')
        x = self.conv(x)
        return x

class MultiScaleBlock(nn.Module):
    """
    多尺度模块：使用不同尺寸的卷积分支，随后将结果融合。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ms1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.ms2 = myconv2d(in_channels, out_channels, pad=(1,1,0,0), kernel_size=3, padding=(1,0))
        self.ms3 = myconv2d(in_channels, out_channels, pad=(2,2,0,0), kernel_size=5, padding=(2,0))
        self.ms4 = myconv2d(in_channels, out_channels, pad=(3,3,0,0), kernel_size=7, padding=(3,0))

        self.activate = nn.Sequential(
            nn.BatchNorm2d(out_channels*4),
            nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x1 = self.ms1(x)
        x2 = self.ms2(x)
        x3 = self.ms3(x)
        x4 = self.ms4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.activate(x)
        x = self.conv(x)
        return x

class RDBlock(nn.Module):
    """
    残差密集块：结合多尺度卷积和局部密集连接，并进行残差学习。
    """
    def __init__(self, inplanes, outplanes, block_index, layer_num=2, bn_d=0.1):
        super(RDBlock, self).__init__()
        self.layer_num = layer_num
        self.block_index = block_index

        # 全局多尺度卷积分支
        self.ms_conv = MultiScaleBlock(inplanes, outplanes)

        # 局部密集连接部分，使用 ModuleList 存储各层
        self.local_convs = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        dim_list = [outplanes]
        for i in range(self.layer_num):
            in_dim = sum(dim_list)
            self.local_convs.append(
                myconv2d(in_dim, in_dim, pad=(1,1,0,0), kernel_size=3, stride=1, padding=(1,0), bias=False)
            )
            self.relu_layers.append(nn.ReLU())
            dim_list.append(in_dim)
        final_in_dim = sum(dim_list)
        self.local_conv = nn.Sequential(
            nn.Conv2d(final_in_dim, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        # 全局分支
        x = self.ms_conv(x)
        global_out = x

        # 局部分支：逐层计算并级联所有中间特征
        outputs = [x]
        current = x
        for conv, relu in zip(self.local_convs, self.relu_layers):
            current = conv(current)
            current = relu(current)
            outputs.append(current)
            current = torch.cat(outputs, dim=1)

        local_out = self.local_conv(current)
        return global_out + local_out

class DoubleConv(nn.Module):
    """
    两次卷积块：(Conv => BN => LeakyReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            myconv2d(out_channels, out_channels, pad=(1,1,0,0), kernel_size=3, padding=(1,0)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    下采样块：通过 RDBlock 生成跳跃连接，并进行下采样。
    """
    def __init__(self, in_channels, out_channels, block_index, bn_d=0.01, dropout=False, down_W=True):
        super().__init__()
        # 根据 down_W 参数确定 stride
        if down_W:
            stride = 2  # 对高和宽同时下采样
        else:
            stride = [1, 2]  # 仅对宽度下采样

        self.rdb = RDBlock(in_channels, in_channels, block_index=block_index)
        if not dropout:
            self.down = nn.Sequential(
                myconv2d(in_channels, out_channels, pad=(1,1,0,0), kernel_size=3,
                         stride=stride, dilation=1, padding=(1,0), bias=False),
                nn.BatchNorm2d(out_channels, momentum=bn_d),
                nn.ReLU()
            )
        else:
            self.down = nn.Sequential(
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels, momentum=bn_d),
                nn.ReLU(),
                nn.Dropout2d(p=0.2)
            )

    def forward(self, x):
        skip = self.rdb(x)
        down_x = self.down(skip)
        return down_x, skip

class Backbone(nn.Module):
    """
    主干网络：构建一个 U-Net 风格的编码器，处理多模态输入数据。
    """
    def __init__(self, params):
        super(Backbone, self).__init__()
        self.use_range = params["input_depth"]["range"]
        self.use_xyz = params["input_depth"]["xyz"]
        self.use_remission = params["input_depth"]["remission"]
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]
        self.OS = params["OS"]
        self.layers = params["extra"]["layers"]
        print("Using ms_rdb_unet backbone")

        # 计算输入通道数，并记录对应索引
        self.input_depth = 0
        self.input_idxs = []
        if self.use_range:
            self.input_depth += 1
            self.input_idxs.append(0)
        if self.use_xyz:
            self.input_depth += 3
            self.input_idxs.extend([1, 2, 3])
        if self.use_remission:
            self.input_depth += 1
            self.input_idxs.append(4)

        dim = 32
        # 分别对 range、zxy 和 remission 进行初步特征提取
        self.inc_range = RDBlock(1, dim, block_index='range')
        self.inc_zxy = RDBlock(3, dim, block_index='zxy')
        self.inc_remission = RDBlock(1, dim, block_index='remission')
        self.merge = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU()
        )

        # 编码器部分，下采样块
        self.down1 = Down(dim, dim*2, 1, bn_d=self.bn_d, down_W=True)
        self.down2 = Down(dim*2, dim*4, 2, bn_d=self.bn_d, dropout=True, down_W=True)
        self.down3 = Down(dim*4, dim*8,  3,bn_d=self.bn_d, dropout=True, down_W=True)
        self.down4 = Down(dim*8, dim*16, 4, bn_d=self.bn_d, dropout=True, down_W=True)
        self.mid = RDBlock(dim*16, dim*16, 5)

        self.last_channels = dim * 16

    def forward(self, x):
        # 为避免与内置函数名冲突，将 range 重命名为 range_data
        range_data = x[:, 0, :, :].unsqueeze(1)
        zxy = x[:, 1:4, :, :]
        remission = x[:, -1, :, :].unsqueeze(1)

        range_feat = self.inc_range(range_data)
        zxy_feat = self.inc_zxy(zxy)
        remission_feat = self.inc_remission(remission)
        x = torch.cat((range_feat, zxy_feat, remission_feat), dim=1)
        x = self.merge(x)

        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x = self.mid(x)

        self.range = range_feat
        self.zxy = zxy_feat
        self.remission = remission_feat
        # 返回最后的特征和跳跃连接（注意：解码器端可能需要按照相反顺序使用 skip connections）
        return x, [skip4, skip3, skip2, skip1]

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth
