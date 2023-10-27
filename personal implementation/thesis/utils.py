from PIL import Image
from torch import Tensor
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf
import torchvision.utils as tvu
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torchvision.models as models
import math
import torch
import torch.nn.functional as F


def fw_hook(module, output):
    print(f"Shape of output to {module} is {output.shape}.")


def check_tensor_shapes_in_model(model: nn.Module):
    for name, layer in model.named_modules():
        layer.register_forward_hook(fw_hook)


def split_to_windows(images, patch_size):
    B, C, H, W = images.shape
    assert H == W, f'Height and width must be equal, but got width = {W}, height= {H}'
    # assert C == 3, f'This must be a 3 channel images, but got {C} channels'
    # 将图片划分为patches, 每个patches为一个向量
    x1 = images.unfold(dimension=2, size=patch_size, step=patch_size)
    x2 = x1.unfold(dimension=3, size=patch_size, step=patch_size)
    x3 = x2.permute(0, 3, 2, 1, 4, 5)
    a, b, c, d, e, f = x3.shape
    patch_num = b * c
    patch_length = d * e * f

    # (B, patch_num, patch_length)
    patches = x3.reshape(a, patch_num, patch_length)
    return patches


def back_to_images(patches, channels_num=3):
    size = patches.shape
    a = size[0]
    b = c = int(math.sqrt(size[1]))
    d = channels_num
    e = f = int(math.sqrt(size[-1] // d))
    y1 = patches.reshape(a, b, c, d, e, f)
    y2 = y1.permute(0, 3, 2, 1, 4, 5)
    y3 = y2.permute(0, 1, 2, 4, 3, 5)

    g, h, i, j, _, _ = y3.shape
    y4 = y3.reshape(g, h, i, j, -1)
    original_images = y4.reshape(g, h, i * j, -1)

    return original_images


# images = torch.randn(16, 16, 120, 120)
# patch_size = 12
# patches = split_to_windows(images, patch_size)
# print(patches.shape)
# original = back_to_images(patches, 16)
# print(original.shape)

# 将一个PIL图像Resize后并转换为Tensor
def resize2tensor(image: Image, size: int) -> Tensor:
    transform = tvt.Compose([
        tvt.Resize(size, interpolation=tvf.InterpolationMode.BICUBIC),
        tvt.ToTensor()
    ])
    return transform(image)


def show_images_tensor(images_per_row: int, images_tensor: Tensor):
    """
    展示图片
    :param images_per_row: 每行展示的图片数量
    :param images_tensor: B*W*H*C
    :return:
    """
    # Create a grid of images using make_grid
    grid = tvu.make_grid(images_tensor, nrow=images_per_row, padding=2, normalize=True)

    # Convert the grid tensor to a numpy array and transpose the dimensions for displaying
    grid_np = grid.numpy().transpose((1, 2, 0))

    # Display the grid of images using matplotlib
    plt.figure(figsize=(15, 15))
    plt.imshow(np.clip(grid_np, 0, 1))
    plt.axis("off")
    plt.show()
