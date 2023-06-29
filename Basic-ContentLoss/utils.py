from PIL import Image
from torch import Tensor
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf
import torchvision.utils as tvu
import matplotlib.pyplot as plt
import numpy as np


# 将一个Tensor图像进行分块
def crop_patches(image: Tensor, patch_size: int) -> list:
    """
    将一个图片切成若干块
    :param image: Tensor图像(C*H*W)
    :param patch_size: 块大小
    :return: 包含图像块的列表
    """
    C, H, W = image.size()
    patches = []
    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            patch = image[:, y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    return patches


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
