import torch.nn as nn
import torch


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(nn.ReLU()(x1))
        x3 = self.conv3(nn.ReLU()(torch.cat((x, x1, x2), dim=1)))  # torch.cat(x, x1, x2) C=16+32+16=64 为何dim=1
        return 0.2 * x + x3


class Model(nn.Module):
    """
    待定义模型，输入大小和patch_size一致，输出大小是4倍的patch_size
    """

    def __init__(self):
        super().__init__()

        self.feature_extraction_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.res_block_layers = nn.Sequential(
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=45, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.upsampling_layers = nn.Sequential(
            nn.PixelShuffle(4)
        )

    def forward(self, x):
        x1 = self.feature_extraction_layers(x)
        x2 = self.res_block_layers(x1)
        x3 = self.conv_last(x2)
        x4 = self.upsampling_layers(torch.cat((x, x3), dim=1))  # C=3+45=48
        return x4
