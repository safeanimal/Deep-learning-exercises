import argparse
import math

from torch.utils.data import DataLoader, random_split
from dataset import ImageDataset
import utils
from tqdm import tqdm
import wandb
from model import Model
from logger import AverageLogger
import copy
import os
import torch
from torch.backends import cudnn
from torch import Tensor


def set_parameters():
    parser = argparse.ArgumentParser(description='Super resolution')
    parser.add_argument('img_folder', nargs='?', type=str,
                        default='D:/Materials/dataset/train/Anime face/images')
    parser.add_argument('img_size', nargs='?', type=int, default=64)
    parser.add_argument('ratio', nargs='?', type=float, default=0.8, help='the ratio of samples used for training')
    parser.add_argument('batch_size', nargs='?', type=int, default=16)
    parser.add_argument('num_workers', nargs='?', type=int, default=6)
    parser.add_argument('lr', nargs='?', type=float, default=0.001)
    parser.add_argument('epoch', nargs='?', type=int, default=20)
    parser.add_argument('save_path', nargs='?', type=str, default='save', help='the path in which the sr_models is saved')
    parser.parse_args()
    return parser


def init_wandb(project_name: str, args):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,

        # track hyperparameters and run metadata
        config=vars(args)
    )


# 将图片(batch_size, 3, img_size, img_size)转为像素序列(batch_size, img_size * img_size, 3)，每个像素是一个rgb三维的元素
def transform_to_pixel_seq(imgs: Tensor):
    imgs = imgs.permute(0, 2, 3, 1)  # (batch_size, img_size, img_size, 3)
    pixel_seq = imgs.reshape(imgs.shape[0], -1, 3)  # (batch_size, img_size * img_size, 3)

    return pixel_seq


# 将像素序列(batch_size, img_size * img_size, 3)转为图片(batch_size, 3, img_size, img_size)
def transform_to_imgs(pixel_seq):
    batch_size = pixel_seq.shape[0]
    img_size = int(math.sqrt(pixel_seq.shape[1]))

    imgs = pixel_seq.reshape(batch_size, img_size, img_size, 3)
    imgs = imgs.permute(0, 3, 1, 2)

    return imgs


if __name__ == '__main__':
    parser = set_parameters()
    args = parser.parse_args()  # 可以通过args.x访问x参数
    full_dataset = ImageDataset(img_size=args.img_size, img_folder=args.img_folder, min_size=args.img_size)

    samples_size = len(full_dataset)
    # 将数据集按照args.ratio分割为training和dev两部分
    training_samples_size = int(args.ratio * samples_size)
    dev_samples_size = samples_size - training_samples_size
    training_dataset, dev_dataset = random_split(full_dataset, [training_samples_size, dev_samples_size])

    # 生成dataloader
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size,
                                     num_workers=args.num_workers, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    # 展示dataloader里的数据（图片）
    for batch in training_dataloader:
        input_images, target_images = batch
        utils.show_images_tensor(4, input_images)
        utils.show_images_tensor(4, target_images)
        break

    # 选择GPU为device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = Model().to(device)

    # 开启benchmark
    cudnn.benchmark = True

    # 设置criterion
    criterion = torch.nn.MSELoss()

    # 设置optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 下面变量用于记录最好的指标和epoch
    best_epoch = 0
    best_metric = 0

    # 开始训练
    for epoch in range(args.epoch):
        # 开启训练模式
        model.train()

        # 设置本轮的logger
        loss_logger = AverageLogger()

        # 设置进度条
        with tqdm(total=training_samples_size, ncols=100) as p_bar:
            # 设置进度条描述
            p_bar.set_description(f'Epoch: {epoch + 1}')
            # 分批次训练
            for batch in dev_dataloader:
                input_images, target_images = batch
                input_images = input_images.to(device)
                target_images = target_images.to(device)

                # 得到输入数据
                inputs = transform_to_pixel_seq(input_images)

                # 得到目标数据
                targets = transform_to_pixel_seq(target_images)

                # 前向传播，得到模型输出
                outputs = model(inputs)  # 注意RNN返回两个参数，第一个是outputs，第二个是最后一个cell的hidden states

                # 计算损失
                loss = criterion(outputs, targets)

                # 记录损失
                loss_logger.update(loss.item())

                # 反向传播，计算并累积梯度
                loss.backward()

                # 更新模型参数
                optimizer.step()

                # 清空模型梯度
                optimizer.zero_grad()

                # 更新进度条&显示损失及指标
                p_bar.update(args.batch_size)
                p_bar.set_postfix({'loss': loss_logger.avg})

        # 开启评估模式
        model.eval()

        # 设置进度条
        with tqdm(total=dev_samples_size, ncols=100) as p_bar:
            # 设置进度条描述
            p_bar.set_description(f'Val: {epoch + 1}')
            # 分批次
            for batch in dev_dataloader:
                input_images, target_images = batch
                input_images = input_images.to(device)
                target_images = target_images.to(device)

                # 取消梯度的计算
                with torch.no_grad():
                    # 得到输入数据
                    inputs = transform_to_pixel_seq(input_images)

                    # 得到目标数据
                    targets = transform_to_pixel_seq(target_images)

                    # 前向传播，得到模型输出
                    outputs = model(inputs)

                    output_imgs = transform_to_imgs(outputs)
                    utils.show_images_tensor(4, output_imgs.cpu())

                # 更新进度条&显示指标
                p_bar.update(args.batch_size)

    last_weights = copy.deepcopy(model.state_dict())
    torch.save(last_weights, os.path.join(args.save_path, 'last.pth'))

    print(f'Complete!')
