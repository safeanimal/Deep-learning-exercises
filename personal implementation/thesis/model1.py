from utils import split_to_windows, back_to_images
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ChannelAttention, SpatialAttention, VisionTransformer


class SuperResolutionModel(nn.Module):
    def __init__(self, image_size=120, in_channels=3, out_channels_after_conv1=16, out_channels_after_conv2=64,
                 patch_sizes=[4, 8, 12],
                 num_blocks=[2, 1, 1], upscale_factor=4):
        super(SuperResolutionModel, self).__init__()
        assert len(patch_sizes) == len(num_blocks), 'The length of patch_sizes and num_blocks must be the same'

        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_sizes = patch_sizes
        self.patch_nums = [(image_size // patch_size) ** 2 for patch_size in patch_sizes]
        self.patch_lengths = [out_channels_after_conv1 * patch_size * patch_size for patch_size in patch_sizes]
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels_after_conv1, kernel_size=3, stride=1, padding=1)
        self.vits = [nn.Sequential(
            *[VisionTransformer(sequence_num=self.patch_nums[i], sequence_length=self.patch_lengths[i]) for j in
              range(self.num_blocks[i])]) for i in range(len(self.patch_sizes))]

        self.planes_num = len(self.patch_sizes) * out_channels_after_conv1 + in_channels
        self.layer_norm1 = nn.LayerNorm(normalized_shape=[self.planes_num, self.image_size, self.image_size], elementwise_affine=False)
        self.channel_attention = ChannelAttention(planes=self.planes_num, ratio=4)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=[self.planes_num, self.image_size, self.image_size], elementwise_affine=False)
        self.spatial_attention = SpatialAttention(scope=3)

        self.conv2 = nn.Conv2d(in_channels=self.planes_num + in_channels, out_channels=out_channels_after_conv2,
                               kernel_size=3,
                               stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.conv3 = nn.Conv2d(in_channels=out_channels_after_conv2 // upscale_factor ** 2, out_channels=3,
                               kernel_size=3, stride=1, padding=1)
        self.color_correction = nn.Parameter(torch.randn(1, 3, 1, 1))

    def forward(self, images):
        """
        images: (B, C, H, W)
        """
        x = self.conv1(images)  # (B, 16, H, W)

        patches_list = []
        for patch_size in self.patch_sizes:
            patches = split_to_windows(x, patch_size=patch_size)
            patches_list.append(patches)

        feature_maps = []
        for i in range(len(self.num_blocks)):
            feature_map = self.vits[i](patches_list[i])
            feature_map = back_to_images(feature_map, channels_num=x.shape[1])
            feature_maps.append(feature_map)

        feature_maps.append(images)  # residual connection
        y = torch.cat(feature_maps, dim=1)
        z = self.channel_attention(y)
        z = self.spatial_attention(z)
        o = torch.cat((z, images), dim=1)
        o = F.leaky_relu(self.conv2(o))
        o = self.pixel_shuffle(o)
        o = self.conv3(o)
        o = o + self.color_correction
        return o

    def to(self, *args, **kwargs):
        self = super(SuperResolutionModel, self).to(*args, **kwargs)
        self.vits = [vit.to(*args, **kwargs) for vit in self.vits]
        return self

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# saved_model = SuperResolutionModel().to(device)
# x = torch.randn((10, 3, 240, 240), device=device)
# out = saved_model(x)
# print(out.shape)
