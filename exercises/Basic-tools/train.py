import argparse
from torch.utils.data import DataLoader, Subset, random_split
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


def set_parameters():
    parser = argparse.ArgumentParser(description='Super resolution')
    parser.add_argument('img_folder', nargs='?', type=str,
                        default='D:/Materials/dataset/anime-face-256by256/animefaces256cleaner')
    parser.add_argument('img_size', nargs='?', type=int, default=256)
    parser.add_argument('patch_size', nargs='?', type=int, default=64)
    parser.add_argument('upsampling_scale', nargs='?', choices=[2, 4, 8], default=4)
    parser.add_argument('samples_size', nargs='?', type=int, default=20000)
    parser.add_argument('ratio', nargs='?', type=float, default=0.8, help='the ratio of samples used for training')
    parser.add_argument('batch_size', nargs='?', type=int, default=16)
    parser.add_argument('num_workers', nargs='?', type=int, default=6)
    parser.add_argument('lr', nargs='?', type=float, default=0.001)
    parser.add_argument('epoch', nargs='?', type=int, default=20)
    parser.add_argument('save_path', nargs='?', type=str, default='save', help='the path in which the model is saved')
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


def get_inputs():
    return None


def get_targets():
    return None


def compute_metric():
    return None


def compute_loss(outputs, targets):
    return None


if __name__ == '__main__':
    parser = set_parameters()
    args = parser.parse_args()  # 可以通过args.x访问x参数
    full_dataset = ImageDataset(img_size=args.img_size, upsampling_scale=args.upsampling_scale,
                                img_folder=args.img_folder, samples_size=args.samples_size)

    # 将数据集按照args.ratio分割为training和dev两部分
    training_samples_size = int(args.ratio * args.samples_size)
    dev_samples_size = args.samples_size - training_samples_size
    training_dataset, dev_dataset = random_split(full_dataset, [training_samples_size, dev_samples_size])

    # 生成dataloader
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # 展示dataloader里的数据（图片）
    for batch in training_dataloader:
        lr_images, hr_images = batch
        utils.show_images_tensor(4, lr_images)
        utils.show_images_tensor(4, hr_images)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 下面变量用于记录最好的指标和epoch
    best_epoch = 0
    best_metric = 0

    # 开始训练
    for epoch in range(args.epoch):
        # 开启训练模式
        model.train()

        # 设置本轮的logger
        metric_logger = AverageLogger()
        loss_logger = AverageLogger()

        # 设置进度条
        with tqdm(total=training_samples_size, ncols=100) as p_bar:
            # 设置进度条描述
            p_bar.set_description(f'Epoch: {epoch + 1}')
            # 分批次训练
            for batch in dev_dataloader:
                lr_images, hr_images = batch
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)


                # 得到输入数据
                inputs = get_inputs()

                # 得到目标数据
                targets = get_targets()

                # 前向传播，得到模型输出
                outputs = model(inputs)

                # 计算指标
                metric = compute_metric()

                # 记录指标
                metric_logger.update(metric.item())

                # 计算损失
                loss = compute_loss(outputs, targets)

                # 记录损失
                loss_logger.update(loss.item())

                # 反向传播，计算并累积梯度
                loss.backward()

                # 更新模型参数
                optimizer.step()

                # 清空模型梯度
                model.zero_grad()

                # 更新进度条&显示损失及指标
                p_bar.update(args.batch_size)
                p_bar.set_postfix({'loss': loss_logger.avg, 'metric': metric_logger.avg})

        # 开启评估模式
        model.eval()
        # 清空相应logger
        metric_logger.clear()
        # 设置进度条
        with tqdm(total=dev_samples_size, ncols=100) as p_bar:
            # 设置进度条描述
            p_bar.set_description(f'Val: {epoch + 1}')
            # 分批次
            for batch in dev_dataloader:
                lr_images, hr_images = batch
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)

                # 取消梯度的计算
                with torch.no_grad():
                    # 得到输入数据
                    inputs = get_inputs()

                    # 得到目标数据
                    targets = get_targets()

                    # 前向传播，得到模型输出
                    outputs = model(inputs)

                    # 计算指标
                    metric = compute_metric()

                    # 记录指标
                    metric_logger.update(metric.item())

                # 更新进度条&显示指标
                p_bar.update(args.batch_size)
                p_bar.set_postfix({'metric': metric_logger.avg})

            # if current epoch has the best metric, update it
            if metric_logger.avg > best_metric:
                best_metric = metric_logger.avg
                best_epoch = epoch + 1
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, os.path.join(args.save_path, 'best.pth'))

    print(f'Complete! \n The best epoch is {best_epoch} \n The best metric is {best_metric}')
