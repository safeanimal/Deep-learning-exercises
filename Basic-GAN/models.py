import torch.nn as nn

nc = 3
nz = 100
ngf = 64
ndf = 64


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                # transposed convolution's output size is
                # (input_size - 1) * stride - 2 * padding + kernel_size + out_padding
                # input: Z (nz, 1, 1) = (100, 1 , 1)
                nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0,
                                   bias=False),
                # output: (ngf*8, 4, 4), because (1 - 1) * 1 - 2 * 0 + 4 + 0 = 4
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # input: (ngf*8, 4, 4)
                nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1,
                                   bias=False),
                # output: (ngf*4, 8, 8), because (4 - 1) * 2 - 2 * 1 + 4 + 0 = 8
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # input: (ngf*4, 8, 8)
                nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1,
                                   bias=False),
                # output: (ngf*2, ), because (8 - 1) * 2 - 2 * 1 + 4 + 0 = 16
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # input: (ngf*2, 16, 16)
                nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1,
                                   bias=False),
                # output: (ngf, 32, 32), because (16 - 1) * 2 - 2 * 1 + 4 + 0 = 32
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # input: (ngf, 32, 32)
                nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
                # output: (nc, 64, 64), because (32 - 1) * 2 - 2 * 1 + 4 + 0 = 32
                nn.Tanh()
            )

    # z: (512, 1, 1)
    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                # normal convolution's output size is
                # (input_size + 2 * padding - kernel_size) / stride + 1
                # input: G(Z) (nc, 64, 64)
                nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
                # output: (ndf, 32, 32), because (64 + 2 * 1 - 4 ) / 2 + 1 = 32
                nn.LeakyReLU(0.2, inplace=True),
                # input: (ndf, 32, 32)
                nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                # output: (ndf*2, 16, 16), because (32 + 2 * 1 - 4 ) / 2 + 1 = 16
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # input: (ndf*2, 16, 16)
                nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
                # output: (ndf*4, 8, 8), because (16 + 2 * 1 - 4 ) / 2 + 1 = 8
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # input: (ndf*4, 8, 8)
                nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
                # output: (ndf*8, 4, 4), because (8 + 2 * 1 - 4 ) / 2 + 1 = 4
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # input: (ndf*8, 4, 4)
                nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
                # output: (1, 1, 1), because (4 + 2 * 0 - 4 ) / 1 + 1 = 1
                nn.Sigmoid()
            )

    # img: (3, 256, 256)
    def forward(self, img):
        return self.layers(img)
