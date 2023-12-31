import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # Using a convolution layer to produce a 2D spatial attention map
        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        min_out, _ = torch.min(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([min_out, avg_out, max_out], dim=1)
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
        self.ln = nn.LayerNorm(normalized_shape=[CA_in_planes, img_size, img_size], elementwise_affine=True)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(in_channels=1 + 2 * CA_in_planes, out_channels=CA_in_planes, kernel_size=3, stride=1,
                              padding=1)

        self.gn1 = nn.GroupNorm(8, CA_in_planes)
        self.gn2 = nn.GroupNorm(1, 1 + 2 * CA_in_planes)

    def forward(self, x):
        x = self.gn1(x)
        sa_out = self.sa(x)
        # print('sa_out', sa_out.shape)
        ca_out = self.ca(x)
        # print('ca_out', ca_out.shape)
        pbmsa_out = self.pbmsa(x)
        # print('pbmsa_out', pbmsa_out.shape)
        concat_out = torch.cat([sa_out, ca_out, pbmsa_out], dim=1)  # (batch_size, 17, ...)
        # print('concat_out', concat_out.shape)
        # mlp_out = self.mlp(concat_out)
        out = self.ln(self.conv(self.act(self.gn2(concat_out))))
        return out


class CoreBlock(nn.Module):
    def __init__(self, in_features, feature_size):
        super().__init__()
        self.blocks1 = nn.Sequential(
            *[AttentionBlock(CA_in_planes=in_features, img_size=feature_size, patch_size=s)
              for s in [2, 4]])
        self.blocks2 = nn.Sequential(
            *[AttentionBlock(CA_in_planes=in_features, img_size=feature_size, patch_size=s)
              for s in [4, 4]])

    def forward(self, x):
        x1 = self.blocks1(x)
        x2 = self.blocks2(x1)
        return x + x2


class ResCoreBlocks(nn.Module):
    def __init__(self, in_features, feature_size):
        super().__init__()
        self.coreBlock1 = CoreBlock(in_features, feature_size)
        self.coreBlock2 = CoreBlock(in_features, feature_size)
        self.coreBlock3 = CoreBlock(in_features, feature_size)
        self.coreBlock4 = CoreBlock(in_features, feature_size)
        self.coreBlock5 = CoreBlock(in_features, feature_size)
        self.coreBlock6 = CoreBlock(in_features, feature_size)

    def forward(self, x):
        x1 = self.coreBlock1(x)
        x2 = self.coreBlock2(x1)
        x3 = self.coreBlock3(x+x2)
        x4 = self.coreBlock4(x3)
        x5 = self.coreBlock5(x3+x4)
        x6 = self.coreBlock6(x5)
        return x5+x6


class SuperResolutionModel(nn.Module):
    def __init__(self, img_size, first_conv_out_channels=32):
        super(SuperResolutionModel, self).__init__()

        self.feature_extractor = nn.Conv2d(in_channels=3, out_channels=first_conv_out_channels, kernel_size=3,
                                           padding=1)

        self.backbone = ResCoreBlocks(in_features=first_conv_out_channels, feature_size=img_size)

        self.conv1 = nn.Conv2d(in_channels=first_conv_out_channels, out_channels=256, kernel_size=3,
                               padding=1)

        self.pixel_shuffle1 = nn.PixelShuffle(upscale_factor=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pixel_shuffle2 = nn.PixelShuffle(upscale_factor=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # transform lr_up to 3D data to satisfy the input size of quantizer
        b, c, h, w = x.shape
        # x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)  # (b, h, w, c)->(b, h*w, c)
        # # Finite-Scalar Quatization
        # x, _ = self.quantizer(x)
        # # transform orig_image_quantized back to orig_image's size
        # x = x.view(b, h, w, c).permute(0, 3, 1, 2)

        x1 = self.feature_extractor(x)
        x2 = self.backbone(x1)

        x3 = F.gelu(self.conv1(x1 + x2))
        x4 = self.pixel_shuffle1(x3)
        x5 = F.gelu(self.conv2(x4))
        x6 = self.pixel_shuffle2(x5)
        x7 = self.conv3(x6)

        return x7
