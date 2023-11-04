import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class SpatialAttention(nn.Module):
    def __init__(self, scope=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=scope, stride=1, padding=scope // 2)

    def forward(self, x):
        mean_out = torch.mean(x, dim=1, keepdim=True)
        # print(mean_out.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print(max_out.shape)
        min_out, _ = torch.min(x, dim=1, keepdim=True)
        # print(min_out.shape)
        y = torch.cat((mean_out, max_out, min_out), dim=1)
        z = torch.sigmoid(self.conv(y))

        return x * z


class ChannelAttention(nn.Module):
    def __init__(self, planes, ratio):
        super(ChannelAttention, self).__init__()
        assert planes // ratio != 0, 'planes//ratio must be greater than 0'
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(planes, planes // ratio),
            nn.LeakyReLU(inplace=True),
            nn.Linear(planes // ratio, planes),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        z = self.layers(y).view(B, C, 1, 1)

        return x * z


class MSA(nn.Module):
    def __init__(self, sequence_length, num_heads=8):
        """
        Multi-head Self-attention：输入大小为[B,sequence_num,sequence_length]
        num_heads: 注意力的头数
        """
        super(MSA, self).__init__()
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.head_dim = self.sequence_length // self.num_heads  # 每一头注意力对应的向量长度，这里head_dim*num_heads=sequence_length

        self.Wq = nn.Linear(in_features=self.head_dim, out_features=self.head_dim)  # 变换矩阵Wq，用于产生queries
        self.Wk = nn.Linear(in_features=self.head_dim, out_features=self.head_dim)  # 变换矩阵Wk，用于产生keys
        self.Wv = nn.Linear(in_features=self.head_dim, out_features=self.head_dim)  # 变换矩阵Wv，用于产生values
        # 变换矩阵，用于将最终输出变换成和输入相同的大小
        self.fc = nn.Linear(in_features=self.sequence_length, out_features=self.sequence_length)

    def add_mask(self, x):
        """
        添加掩码，目的是让energy矩阵的上三角部分全变为负无穷，这样经由softmax后最终得到的注意力分数矩阵的上三角部分近似为0
        直观上，注意力分数矩阵和值矩阵相乘会导致每个序列只和自身之前（包括自身）的序列“交互”，而不会和后面的序列“交互”
        x: (B, sequence_num, sequence_length)
        """

    def forward(self, queries, keys, values):
        """
        queries/keys/values: (B, sequence_num, sequence_length)，在自注意力的情况下，三者传入的张量应一样
        """
        B, sequence_num, sequence_length = queries.shape

        # 注意reshape会返回一个新的向量，不会改变原有的queries/keys/values
        queries = queries.reshape(B, sequence_num, self.num_heads, self.head_dim)
        keys = keys.reshape(B, sequence_num, self.num_heads, self.head_dim)
        values = values.reshape(B, sequence_num, self.num_heads, self.head_dim)

        queries = self.Wq(queries)
        keys = self.Wk(keys)
        values = self.Wv(values)

        # TODO: 检查下面代码是否正确
        # Einstein乘法
        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])
        # 得到注意力分数
        attention = F.softmax(energy / self.sequence_length ** (1 / 2), dim=3)
        # 注意力分数和值相乘，得到输出
        out = torch.einsum("bhqk,bkhd->bqhd", [attention, values]).reshape(B, sequence_num, -1)
        # 将输出线性变换成和输入一样的形状
        out = self.fc(out)

        return out


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, sequence_length, dropout, sequence_num):
        """
        sequence_length: d
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, sequence_num, sequence_length))

        Y = torch.arange(sequence_num, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, sequence_length, 2, dtype=torch.float32) / sequence_length)
        self.P[:, :, 0::2] = torch.sin(Y)
        self.P[:, :, 1::2] = torch.cos(Y)

    def forward(self, X):
        X = X + self.P.to(X.device)
        return self.dropout(X)


class VisionTransformer(nn.Module):
    def __init__(self, sequence_num, sequence_length, expand=4):
        """
        ViT (VisionTransformer)：将原始图片切成sequence_num个块，每个块有sequence_length个像素点。将每个块添加上位置编码后，输入到ViT中。
        """
        super(VisionTransformer, self).__init__()
        self.positional_encoding = PositionalEncoding(sequence_length, dropout=0.5, sequence_num=sequence_num)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=[sequence_num, sequence_length])
        self.msa = MSA(sequence_length=sequence_length)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=[sequence_num, sequence_length])
        self.mlp = nn.Sequential(
            nn.Linear(sequence_length, sequence_length * expand),
            nn.LeakyReLU(inplace=True),
            nn.Linear(sequence_length * expand, sequence_length),
        )

    def forward(self, x):
        """
        x: (B, sequence_num, sequence_length)
        """
        x = self.positional_encoding(x)
        x1 = self.layer_norm1(x)
        x2 = self.msa(x1, x1, x1)
        x3 = self.layer_norm2(x + x2)
        x4 = self.mlp(x3)

        return x4 + x + x2


class ResNet(nn.Module):
    def __init__(self, main, has_projection=False):
        """
        构建一个残差网络
        main: 主干模块，即残差掠过的模块（输入和输出的最后一个维度必须相等）
        has_projection：表示残差是否经过映射
        """
        super(ResNet, self).__init__()
        self.main = main
        self.has_projection = has_projection
        if has_projection:
            self.project = nn.Linear(in_features=main.shape[-1], out_features=main.shape[-1])

    def forward(self, x):
        y = F.relu(self.main(x))
        if self.has_projection:
            x = self.project(x)
        return x + y


class MixAttentionBlock(nn.Module):
    def __init__(self, feature_map_size, scope, planes, ratio):
        super(MixAttentionBlock, self).__init__()

        self.sq1 = nn.Sequential(nn.LayerNorm([planes, feature_map_size, feature_map_size], elementwise_affine=False),
                                 ChannelAttention(planes=planes, ratio=ratio))
        self.res1 = ResNet(main=self.sq1, has_projection=False)

        self.sq2 = nn.Sequential(nn.LayerNorm([planes, feature_map_size, feature_map_size], elementwise_affine=False),
                                 SpatialAttention(scope=scope))
        self.res2 = ResNet(main=self.sq2, has_projection=False)

        self.sq = nn.Sequential(self.res1, self.res2)
        self.res = ResNet(main=self.sq, has_projection=False)

    def forward(self, x):
        return self.res(x)


class MixAttentionResBlocks(nn.Module):
    def __init__(self, feature_map_size, scope, planes, ratio, blocks_num=6):
        super(MixAttentionResBlocks, self).__init__()
        self.blocks = nn.Sequential(
            *[ResNet(MixAttentionBlock(feature_map_size, scope, planes, ratio)) for i in range(blocks_num)])

    def forward(self, x):
        return self.blocks(x)


class MixViTBlocks(nn.Module):
    def __init__(self, sequence_num, sequence_length, blocks_num=6):
        super(MixViTBlocks, self).__init__()
        self.blocks = nn.Sequential(
            *[ResNet(VisionTransformer(sequence_num, sequence_length)) for i in range(blocks_num)])

    def forward(self, x):
        return self.blocks(x)


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class CropAndConcat(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat([x, contracting_x], dim=1)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down_conv = nn.ModuleList(
            [DoubleConvolution(i, o) for i, o in [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
        self.middle_conv = DoubleConvolution(512, 1024)
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.up_conv = nn.ModuleList(
            [DoubleConvolution(i, o) for i, o in [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        pass_through = []
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            pass_through.append(x)
            x = self.down_sample[i](x)
        x = self.middle_conv(x)
        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x, pass_through.pop())
            x = self.up_conv[i](x)
        x = self.final_conv(x)
        return x