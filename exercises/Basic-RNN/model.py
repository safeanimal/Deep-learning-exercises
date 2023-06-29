import torch.nn as nn
import torch
from torch import Tensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    """
    待定义模型
    """
    def __init__(self):
        super().__init__()
        # num_layers指的是堆叠多少层，rnn的长度根据序列长度自动推断
        # 单向rnn的情况下，输出序列和输入序列长度一致
        # 输入数据（单个像素）的维度为3（RGB），设定隐变量的维度为16
        self.rnn = nn.RNN(input_size=3, hidden_size=16, num_layers=1, batch_first=True)
        self.fcl = nn.Linear(16, 3)

    def forward(self, z):
        h0 = torch.zeros((1, 16, 16)).to(device)
        outputs, _ = self.rnn(z, h0)
        return self.fcl(outputs)
