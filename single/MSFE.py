import torch
import torch.nn as nn
import torch.nn.functional as F



class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)


########################################
# 多尺度卷积增强模块（仅使用空间注意力）
########################################

class MultiScaleEnhancement(nn.Module):
    """
    对输入特征图进行多尺度卷积增强：
    1. 用 1×1 卷积降维；
    2. 分别用 3×3、5×5、7×7 卷积提取不同尺度信息；
    3. 利用空间注意力模块对多尺度信息进行重标定；
    4. 通过可学习缩放因子控制多尺度信息与原始信息的融合。
    """

    def __init__(self, in_channels, reduction_factor=4):
        super(MultiScaleEnhancement, self).__init__()
        mid_channels = in_channels // reduction_factor
        self.down = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, stride=1, padding=2)
        self.conv7 = nn.Conv2d(mid_channels, mid_channels, kernel_size=7, stride=1, padding=3)
        self.sa = SpatialAttentionModule()
        self.up = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1)
        # 可学习缩放因子，初始为 0 保证初期主要保留原始信息
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_down = self.down(x)
        x3 = self.conv3(x_down)
        x5 = self.conv5(x_down)
        x7 = self.conv7(x_down)
        x_ms = x3 + x5 + x7
        # 应用空间注意力
        x_ms = x_ms * self.sa(x_ms)
        x_ms = self.up(x_ms)
        # 残差融合：用可学习缩放系数控制增强信息的贡献
        return x +  self.alpha*x_ms
