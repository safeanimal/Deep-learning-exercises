from torchvision.utils import make_grid


# 显示数据集中的图片
def show_dataloader_images(dataloader, num):
    patch_pairs = dataloader[:num]

