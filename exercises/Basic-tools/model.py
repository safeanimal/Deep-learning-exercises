import torch.nn as nn


class Model(nn.Module):
    """
    待定义模型
    """
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return self.layers(z)
