import torch.nn as nn


class SRG(nn.Module):
    def __init__(self):
        super(SRG, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=9, stride=1, padding=4),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            # 最后一般不会用全连接层，因为全连接层要先将图片flatten，这样会丢失空间信息，不利于重建图像
            # 一般最后都是卷积层
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, img):
        return self.layers(img)
