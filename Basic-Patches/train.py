from dataset import ImageDataset
from torch.utils.data import DataLoader, random_split

if __name__ == '__main__':
    # 设置一些参数
    image_folder_path = "D:/Materials/dataset/anime-face-256by256/animefaces256cleaner"
    image_size = 256
    patch_size = 32
    split_ratio = 0.8
    upsampling_scale = 2

    # 生成full_dataset
    full_dataset = ImageDataset(image_folder_path, image_size, patch_size, upsampling_scale)

    # 计算full_dataset的划分长度，以便切分为train_dataset和val_dataset
    full_dataset_length = len(full_dataset)
    train_dataset_length = int(full_dataset_length * split_ratio)

    # 将full_dataset划分为train_dataset和val_dataset
    train_dataset, val_dataset = \
        random_split(full_dataset, [train_dataset_length, full_dataset_length-train_dataset_length])

    # 将Dataset加载为Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=6)

    # 显示Dataloader中的图片

    # 开始训练

    # 将图像分块输入到模型中
