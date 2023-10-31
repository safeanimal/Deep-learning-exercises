import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import SpatialAttention, ChannelAttention, ResNet
from common import TemporalEmbedding


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
    def __init__(self, CA_in_planes, img_size=512, patch_size=8, embed_dim=256, num_heads=8):
        super(AttentionBlock, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(planes=CA_in_planes, ratio=4)
        self.pbmsa = PBMSA(patch_size, embed_dim, num_heads, channels=CA_in_planes)
        self.ln = nn.LayerNorm(normalized_shape=[CA_in_planes, img_size, img_size], elementwise_affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=3 * CA_in_planes, out_channels=CA_in_planes, kernel_size=3, stride=1,
                              padding=1)

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


class GenerativeModel(nn.Module):
    def __init__(self, image_size=512, first_conv_out_channels=16):
        super(GenerativeModel, self).__init__()
        self.embedding1 = TemporalEmbedding(dim_emb=1024, dim_out=3)
        self.feature_extractor = nn.Conv2d(in_channels=3, out_channels=first_conv_out_channels, kernel_size=3,
                                           padding=1)
        self.embedding2 = TemporalEmbedding(dim_emb=1024, dim_out=first_conv_out_channels)
        self.block1 = ResNet(AttentionBlock(CA_in_planes=first_conv_out_channels, img_size=image_size))
        self.block2 = ResNet(AttentionBlock(CA_in_planes=first_conv_out_channels, img_size=image_size))
        self.block3 = ResNet(AttentionBlock(CA_in_planes=first_conv_out_channels, img_size=image_size))
        self.block4 = ResNet(AttentionBlock(CA_in_planes=first_conv_out_channels, img_size=image_size))

        self.conv1 = nn.Conv2d(in_channels=first_conv_out_channels, out_channels=256, kernel_size=3,
                               padding=1)
        self.embedding3 = TemporalEmbedding(dim_emb=1024, dim_out=256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.embedding4 = TemporalEmbedding(dim_emb=1024, dim_out=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3,
                               padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.embedding1(x, t)
        x1 = self.feature_extractor(x)
        x1 = self.embedding2(x1, t)
        x2 = self.block1(x1)
        x2 = self.embedding2(x2, t)

        x3 = self.block2(x2)
        x3 = self.embedding2(x3, t)

        x4 = self.block3(x3)
        x4 = self.embedding2(x4, t)

        x5 = self.block4(x4)
        x5 = self.embedding2(x5, t)

        x6 = F.relu((self.conv1(x5)))
        x6 = self.embedding3(x6, t)

        x7 = F.relu((self.conv2(x6)))
        x7 = self.embedding4(x7, t)

        denoised_image = self.conv3(x7)
        return denoised_image
