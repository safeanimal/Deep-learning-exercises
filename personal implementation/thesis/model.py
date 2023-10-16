import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # Using a convolution layer to produce a 2D spatial attention map
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=64, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PBMSA(nn.Module):
    def __init__(self, patch_size, embed_dim, num_heads, channels=64):
        super(PBMSA, self).__init__()

        self.channels = channels
        # Define the patch size
        self.patch_size = patch_size

        # Linear layer to project the patches into the desired embedding dimension
        self.patch_embedding = nn.Linear(patch_size * patch_size * channels, embed_dim)

        # Multi-head self-attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)

        self.restore_embedding = nn.Linear(embed_dim, patch_size * patch_size * channels)

    def restore_original(self, x, batch_size, channels, height, width):
        """
        x: Tensor of shape (batch_size, height // patch_size * width // patch_size, embed_dim)
        Returns: Tensor of shape (batch_size, channels, height, width)
        """
        # Project back to patch representation
        patches = self.restore_embedding(x)
        # Reshape to get patches
        patches = patches.reshape(batch_size, height // self.patch_size, width // self.patch_size, self.channels,
                                  self.patch_size,
                                  self.patch_size)

        # Use fold operation to combine patches
        restored = patches.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, self.channels, height, width)

        return restored

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        # Extract patches from the image
        # Shape: (batch_size, channels, height // patch_size, width // patch_size, patch_size, patch_size)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Reshape patches to have shape (batch_size, height // patch_size * width // patch_size, patch_size * patch_size * channels)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(x.size(0), -1,
                                                            self.patch_size * self.patch_size * self.channels)

        # Embed patches using the linear layer
        # Shape: (batch_size, height // patch_size * width // patch_size, embed_dim)
        embedded_patches = self.patch_embedding(patches)
        # Transpose for multi-head attention
        # Shape: (height // patch_size * width // patch_size, batch_size, embed_dim)
        query = key = value = embedded_patches

        # Apply self-attention
        attn_output, _ = self.self_attention(query, key, value)
        return self.restore_original(attn_output, batch_size, channels, height, width)


class AttentionBlock(nn.Module):
    def __init__(self, CA_in_planes, img_size=480, patch_size=8, embed_dim=256, num_heads=8):
        super(AttentionBlock, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_planes=CA_in_planes)
        self.pbmsa = PBMSA(patch_size, embed_dim, num_heads, channels=CA_in_planes)
        self.ln = nn.LayerNorm(normalized_shape=[CA_in_planes, img_size, img_size])
        # self.mlp = nn.Sequential(
        #     nn.Linear(in_features=17, out_features=128),
        #     nn.ReLU(),
        #     nn.Linear(in_features=128, out_features=3)
        # )
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=1+2*CA_in_planes, out_channels=CA_in_planes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        sa_out = self.sa(x)
        # print('sa_out', sa_out.shape)
        ca_out = self.ca(x)
        # print('ca_out', ca_out.shape)
        pbmsa_out = self.pbmsa(x)
        # print('pbmsa_out', pbmsa_out.shape)
        concat_out = torch.cat([sa_out, ca_out, pbmsa_out], dim=1)  # (batch_size, 17, ...)
        # print('concat_out', concat_out.shape)
        # mlp_out = self.mlp(concat_out)
        out = self.ln(self.conv(self.relu(concat_out)))
        return out


class SuperResolutionModel(nn.Module):
    def __init__(self, num_blocks, img_size, first_conv_out_channels=16):
        super(SuperResolutionModel, self).__init__()

        self.feature_extractor = nn.Conv2d(in_channels=3, out_channels=first_conv_out_channels, kernel_size=3,
                                           padding=1)
        self.blocks = nn.Sequential(*[AttentionBlock(CA_in_planes=first_conv_out_channels, img_size=img_size) for _ in range(num_blocks)])
        self.conv1 = nn.Conv2d(in_channels=first_conv_out_channels, out_channels=256, kernel_size=3,
                               padding=1)
        self.act1 = nn.LeakyReLU()
        self.pixel_shuffle1 = nn.PixelShuffle(upscale_factor=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(upscale_factor=2)
        self.act2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3,
                               padding=1)

    def forward(self, x):
        x1 = self.feature_extractor(x)
        x2 = self.blocks(x1)
        x3 = self.act1(self.conv1(x2))
        x4 = self.pixel_shuffle1(x3)
        x5 = self.act2(self.conv2(x4))
        x6 = self.pixel_shuffle2(x5)
        hr_image = self.conv3(x6)
        return hr_image
