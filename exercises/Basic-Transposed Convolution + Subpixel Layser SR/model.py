import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=256, kernel_size=33, stride=1, padding=16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.layers(x)


class SRG2(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('conv_1', nn.Conv2d(in_channels=3, out_channels=16, kernel_size=13, stride=1, padding=6))
        self.layers.add_module('leaky_relu1', nn.LeakyReLU(inplace=True))
        self.layers.add_module('conv_2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=9, stride=1, padding=4))
        self.layers.add_module('leaky_relu2', nn.LeakyReLU(inplace=True))
        self.layers.add_module('conv_3', nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1))
        self.layers.add_module('leaky_relu3', nn.LeakyReLU(inplace=True))
        self.layers.add_module('conv_4', nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3, stride=1, padding=1))
        self.layers.add_module('leaky_relu4', nn.LeakyReLU(inplace=True))

        # As for ConvTransposed2d, output_size = (input_size - 1) * stride - 2 * padding + kernel_size + out_padding
        # state: 12*64*64
        self.layers.add_module('conv_transposed1', nn.ConvTranspose2d(in_channels=12, out_channels=8, kernel_size=6,
                                                                      stride=2, padding=2, bias=False))
        self.layers.add_module('leaky_relu5', nn.LeakyReLU(inplace=True))
        # state: 8*128*128
        self.layers.add_module('conv_transposed2', nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=6,
                                                                      stride=2, padding=2, bias=False))
        # state: 3*256*256

    def forward(self, img):
        return self.layers(img)
