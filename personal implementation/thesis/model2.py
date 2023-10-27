import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ResNet, MixViTBlocks, MixAttentionBlock
from utils import split_to_windows, back_to_images


class SuperResolutionModel(nn.Module):
    def __init__(self, image_size=120, patch_size=6, in_channels_conv1=3, out_channels_conv1=8, out_channels_conv2=128, blocks_num=2,
                 upscale_factor=4):
        super(SuperResolutionModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels_conv1, out_channels=out_channels_conv1, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.patch_size = patch_size
        self.out_channels_conv1 = out_channels_conv1

        self.blocks = MixViTBlocks(sequence_num=(image_size//patch_size)**2,
                                   sequence_length=(patch_size**2)*out_channels_conv1, blocks_num=blocks_num)
        self.backbone = ResNet(main=self.blocks, has_projection=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels_conv1, out_channels=out_channels_conv2, kernel_size=3, stride=1,
                               padding=2, bias=True)
        self.upscale = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.conv3 = nn.Conv2d(in_channels=out_channels_conv2 // (upscale_factor ** 2), out_channels=in_channels_conv1,
                               kernel_size=3, stride=1, padding=1, bias=True)

        self.color_correction = nn.Parameter(torch.randn(1, in_channels_conv1, 1, 1))

    def forward(self, LR_images):
        x1 = self.conv1(LR_images)
        patches = split_to_windows(images=x1, patch_size=self.patch_size)
        x2 = self.backbone(patches)
        x3 = back_to_images(patches=x2, channels_num=self.out_channels_conv1)
        x4 = F.leaky_relu(self.conv2(x3))
        x5 = self.upscale(x4)
        x6 = F.leaky_relu(self.conv3(x5))
        HR_images = x6 + self.color_correction

        return HR_images
