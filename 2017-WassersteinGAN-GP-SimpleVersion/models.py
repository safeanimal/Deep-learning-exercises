import torch.nn as nn

nc = 3
nz = 100
ngf = 12
ndf = 64


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # abandon ConvTransposed2d to avoid checkerboard issue
        self.layers = nn.Sequential(
                # transposed convolution's output size is
                # (input_size - 1) * stride - 2 * padding + kernel_size + out_padding
                # input: Z (nz, 1, 1) = (100, 1 , 1)
                nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 64, kernel_size=4, stride=1, padding=0,
                                   bias=False),
                # output: (ngf*64, 4, 4)
                nn.BatchNorm2d(ngf * 64),
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=ngf * 64, out_channels=ngf * 32, kernel_size=4, stride=2, padding=1,
                                   bias=False),
                # nn.PixelShuffle(2),
                # output: (ngf*32, 8, 8)
                nn.BatchNorm2d(ngf * 32),
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=ngf * 32, out_channels=ngf * 16, kernel_size=4, stride=2, padding=1,
                                   bias=False),
                # nn.PixelShuffle(2),
                # output: (ngf*16, 16, 16)
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=ngf * 16, out_channels=ngf*4, kernel_size=4, stride=2, padding=1,
                                   bias=False),
                # nn.PixelShuffle(2),
                # output: (ngf*4, 32, 32)
                nn.BatchNorm2d(ngf*4),
                nn.ReLU(True),

                # nn.ConvTranspose2d(in_channels=ngf*8, out_channels=ngf, kernel_size=4, stride=2, padding=1
                #                    , bias=False),
                nn.PixelShuffle(2),
                # output: (ngf, 64, 64)
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),


                nn.PixelShuffle(2),
                # output: (ngf//4=3, 128, 128)
                nn.Tanh()
            )

    # z: (512, 1, 1)
    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # delete the final sigmoid layer and batchnorm compared to normal GAN
        self.layers = nn.Sequential(
                # normal convolution's output size is
                # (input_size + 2 * padding - kernel_size) / stride + 1
                # input: G(Z) (nc, 128, 128)
                nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # status: (ndf, 64, 64), because (128 + 2 * 1 - 4 ) / 2 + 1 = 64
                nn.Conv2d(in_channels=ndf, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
                # status: (ndf, 32, 32), because (64 + 2 * 1 - 4 ) / 2 + 1 = 32
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                # status: (ndf*2, 16, 16), because (32 + 2 * 1 - 4 ) / 2 + 1 = 16
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
                # status: (ndf*4, 8, 8), because (16 + 2 * 1 - 4 ) / 2 + 1 = 8
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
                # status: (ndf*8, 4, 4), because (8 + 2 * 1 - 4 ) / 2 + 1 = 4
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
                # output: (1, 1, 1), because (4 + 2 * 0 - 4 ) / 1 + 1 = 1
            )

    # img: (3, 256, 256)
    def forward(self, img):
        predicts = self.layers(img)
        return predicts.view(len(predicts), 1)
