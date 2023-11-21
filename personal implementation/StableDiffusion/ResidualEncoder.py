from sd_models.RRDBNet import RRDB
from sd_models.SwinIR import SwinTransformerBlock, PatchEmbed
import torch.nn as nn
import torch.nn.functional as F


class ResidualEncoder(nn.Module):
    def __init__(self, nf=3, dim=3, input_resolution=(128, 128), num_heads=1, d_cond=768, n_cod=77, window_size=8):
        super().__init__()
        self.L = input_resolution[0]*input_resolution[1]
        self.d_cond = d_cond
        self.n_cond = n_cod
        # TODO: 输入为(B, C, H, W)
        self.rrdb = RRDB(nf=nf)
        self.patch_embed = PatchEmbed(img_size=input_resolution[0])
        # TODO: 输入为(B, Ph*Pw, C)
        #   需要x = self.patch_embed(x)
        self.stl = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.ln = nn.Linear(in_features=dim, out_features=d_cond)
        self.conv = nn.Conv2d(in_channels=self.L, out_channels=n_cod, kernel_size=3, stride=1, padding=1)

        #   TODO: 输出应该为(B, n_cond, d_cond)，CLIP_embedding默认是输出(B, 77, 768)
    def forward(self, x, x_size):
        """
        x: (B, C, H, W)
        """
        x1 = self.rrdb(x)
        x2 = self.patch_embed(x1) # (B, Ph*Pw, C)
        x3 = self.stl(x2, x_size) # (B, Ph*Pw, C)
        x4 = F.gelu(self.ln(x3))
        B, L, _ = x4.shape
        x4 = x4.view(B, L, 24, 32) # the last two dimensions' production must equal to d_cond
        x5 = self.conv(x4)
        return x5.view(B, self.n_cond, self.d_cond)


# if __name__ == "__main__":
#     x_size = (128, 128)
#     res_encoder = ResidualEncoder(input_resolution=x_size).to('cuda:0')
#     x = torch.randn((16, 3, *x_size), device='cuda:0')
#     x = res_encoder(x, x_size)
#     print(x.shape)


