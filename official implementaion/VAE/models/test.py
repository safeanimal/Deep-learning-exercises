import torch
import torch.nn as nn

B = 16
H = 32
W = 32
K = 4096
D = 3

embedding = nn.Embedding(K, D)

latents = torch.randn(B, D, H, W)
latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
latents_shape = latents.shape
flat_latents = latents.view(-1, D)  # [BHW x D]

term1 = torch.sum(flat_latents ** 2, dim=1, keepdim=True)
term2 = torch.sum(embedding.weight ** 2, dim=1)
term3 = 2 * torch.matmul(flat_latents, embedding.weight.t())  # [BHW x K]
dist = term1 + term2 + term3
