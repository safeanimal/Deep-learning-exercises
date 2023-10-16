import argparse
from torch.utils.data import DataLoader, Subset, random_split
from dataset import ImageDataset
import utils
from tqdm import tqdm
import wandb
from model import SuperResolutionModel
from logger import AverageLogger
import copy
import os
import torch
from metric import peak_signal_to_noise_ratio
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR


def set_parameters():
    parser = argparse.ArgumentParser(description='Super resolution')
    parser.add_argument('hr_img_folder', nargs='?', type=str,
                        default='D:/Materials/dataset/DIV2K+Flickr2K/HR_patches')
    parser.add_argument('lr_img_folder', nargs='?', type=str,
                        default='D:/Materials/dataset/DIV2K+Flickr2K/LR_unknown_x4_patches')
    parser.add_argument('img_size', nargs='?', type=int, default=256)
    parser.add_argument('patch_size', nargs='?', type=int, default=120)
    parser.add_argument('upsampling_scale', nargs='?', choices=[2, 4, 8], default=4)
    parser.add_argument('samples_size', nargs='?', type=int, default=10000)
    parser.add_argument('ratio', nargs='?', type=float, default=0.8, help='the ratio of samples used for training')
    parser.add_argument('batch_size', nargs='?', type=int, default=64)
    parser.add_argument('num_workers', nargs='?', type=int, default=8)
    parser.add_argument('lr', nargs='?', type=float, default=0.00005)
    parser.add_argument('epoch', nargs='?', type=int, default=500)
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


if __name__ == '__main__':
    parser = set_parameters()
    args = parser.parse_args()  # 可以通过args.x访问x参数
    full_dataset = ImageDataset(lr_img_folder=args.lr_img_folder, hr_img_folder=args.hr_img_folder)

    total_samples_num = len(full_dataset)
    # 将数据集按照args.ratio分割为training和dev两部分
    training_samples_size = int(args.ratio * total_samples_num)
    dev_samples_size = total_samples_num - training_samples_size
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
    model = SuperResolutionModel(num_blocks=8, img_size=args.patch_size).to(device)

    # 开启benchmark
    cudnn.benchmark = True

    # 设置criterion
    criterion = torch.nn.L1Loss()

    # 设置optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=60, gamma=0.5)  # Decay LR by a factor of 0.6 every 30 epochs

    # 下面变量用于记录最好的指标和epoch
    best_epoch = 0
    best_metric = 0
    iteration = 0

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
            for batch in training_dataloader:
                iteration = iteration + 1
                lr_images, hr_images = batch
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)

                # 前向传播，得到模型输出
                outputs = model(lr_images)

                # 计算指标
                metric = peak_signal_to_noise_ratio(hr_images, outputs)

                # 记录指标
                metric_logger.update(metric.item())

                # 计算损失
                loss = criterion(outputs, hr_images)

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
                p_bar.set_postfix({'loss': loss_logger.avg, 'PSNR': metric_logger.avg, 'iteration': iteration})

        scheduler.step()

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
                    # 前向传播，得到模型输出
                    outputs = model(lr_images)

                    # 计算指标
                    metric = peak_signal_to_noise_ratio(hr_images, outputs)

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
