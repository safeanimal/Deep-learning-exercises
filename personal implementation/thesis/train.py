from torch.utils.data import DataLoader, ConcatDataset
from dataset import ImageDataset
import utils
from tqdm import tqdm
import wandb
from model0 import SuperResolutionModel
from logger import AverageLogger
import copy
import os
import torch
from metric import peak_signal_to_noise_ratio
from torch.backends import cudnn
from torch.optim.lr_scheduler import MultiStepLR
import yaml
from torchsummary import summary
from loss import PerceptualLoss, EdgeLoss
from SwinIR import SwinIR

def init_wandb(project_name: str, args):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,

        # track hyperparameters and run metadata
        config=args
    )


if __name__ == '__main__':
    with open('config.yml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    dataset_config = config['dataset']
    hyperparameters_config = config['hyperparameters']

    lr_train_img_folders = dataset_config['lr_train_img_folders']
    hr_train_img_folders = dataset_config['hr_train_img_folders']

    lr_val_img_folder = dataset_config['lr_val_img_folder']
    hr_val_img_folder = dataset_config['hr_val_img_folder']

    save_path = dataset_config['save_path']

    lr = hyperparameters_config['lr']
    batch_size = hyperparameters_config['batch_size']
    num_workers = hyperparameters_config['num_workers']
    epoch = hyperparameters_config['epoch']
    lr_decay_schedule = hyperparameters_config['lr_decay_schedule']

    loss_name = hyperparameters_config['loss_name']
    add_perceptual_loss = hyperparameters_config['add_perceptual_loss']
    add_edge_loss = hyperparameters_config['add_edge_loss']

    optimizer_name = hyperparameters_config['optimizer_name']
    coefficient_perceptual_loss = hyperparameters_config['coefficient_perceptual_loss']
    coefficient_edge_loss = hyperparameters_config['coefficient_edge_loss']

    model_num = hyperparameters_config['model_num']

    # 训练集
    training_dataset1 = ImageDataset(lr_img_folder=lr_train_img_folders[0], hr_img_folder=hr_train_img_folders[0])
    training_dataset2 = ImageDataset(lr_img_folder=lr_train_img_folders[1], hr_img_folder=hr_train_img_folders[1])
    training_dataset = ConcatDataset([training_dataset1, training_dataset2])
    # 训练集大小
    training_samples_size = len(training_dataset)
    # 验证集
    val_dataset = ImageDataset(lr_img_folder=lr_val_img_folder,
                               hr_img_folder=hr_val_img_folder)
    # 验证集大小
    val_samples_size = len(val_dataset)

    # 生成dataloader
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    # 展示dataloader里的数据（图片）
    for batch in training_dataloader:
        lr_images, gt_images = batch
        utils.show_images_tensor(4, lr_images)
        utils.show_images_tensor(4, gt_images)
        break

    # 选择GPU为device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model_parameters = config['model'+str(model_num)]
    image_size = model_parameters['image_size']
    if model_num == 0:
        num_blocks = model_parameters['num_blocks']
        model = SuperResolutionModel(num_blocks=num_blocks, img_size=image_size).to(device)
    elif model_num == 1:
        in_channels = model_parameters['in_channels']
        out_channels_after_conv1 = model_parameters['out_channels_after_conv1']
        out_channels_after_conv2 = model_parameters['out_channels_after_conv2']
        patch_sizes = model_parameters['patch_sizes']
        num_blocks = model_parameters['num_blocks']
        upscale_factor = model_parameters['upscale_factor']
        model = SuperResolutionModel(in_channels=in_channels, out_channels_after_conv1=out_channels_after_conv1,
                                     out_channels_after_conv2=out_channels_after_conv2, patch_sizes=patch_sizes,
                                     num_blocks=num_blocks, upscale_factor=upscale_factor).to(device)
    elif model_num == 2:
        model = SuperResolutionModel().to(device)
    elif model_num == -1: # 对比SwinIR的效果
        upscale = 4
        window_size = 8
        height = (image_size // upscale // window_size + 1) * window_size
        width = (image_size // upscale // window_size + 1) * window_size
        model = SwinIR(upscale=4, img_size=(height, width),
                       window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                       embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect').to(device)


    # 开启benchmark
    cudnn.benchmark = True

    # 设置loss_function_base
    if loss_name == 'MSE':
        loss_function_base = torch.nn.MSELoss()
    elif loss_name == 'L1':
        loss_function_base = torch.nn.L1Loss()

    # 添加perceptual loss function
    if add_perceptual_loss:
        loss_function_perceptual = PerceptualLoss().to(device)
    # 添加edge loss function
    if add_edge_loss:
        loss_function_edge = EdgeLoss().to(device)
    # 设置optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = MultiStepLR(optimizer, lr_decay_schedule, gamma=0.2, last_epoch=-1,
                            verbose=False)

    init_wandb(project_name='train5', args=config)

    # 下面变量用于记录最好的指标和epoch
    best_epoch = 0
    best_metric = 0
    iteration = 0

    # 开始训练
    for epoch in range(epoch):
        # 开启训练模式
        model.train()

        # 设置进度条
        with tqdm(total=training_samples_size, ncols=100) as p_bar:
            # 设置进度条描述
            p_bar.set_description(f'Epoch: {epoch + 1}')
            # 分批次训练
            for batch in training_dataloader:
                iteration = iteration + 1
                lr_images, gt_images = batch
                lr_images = lr_images.to(device)
                gt_images = gt_images.to(device)

                # 前向传播，得到模型输出
                hr_images = model(lr_images)

                # 计算指标
                metric = peak_signal_to_noise_ratio(gt_images, hr_images)

                # 计算损失
                loss = loss_function_base(hr_images, gt_images)
                if add_perceptual_loss:
                    loss += coefficient_perceptual_loss * loss_function_perceptual(hr_images, gt_images)
                if add_edge_loss:
                    loss += coefficient_edge_loss * loss_function_edge(hr_images, gt_images)
                # 反向传播，计算并累积梯度
                loss.backward()

                # 更新模型参数
                optimizer.step()

                # 清空模型梯度
                optimizer.zero_grad()

                # 更新进度条&显示损失及指标
                p_bar.update(batch_size)
                p_bar.set_postfix({'loss': loss.item(), 'PSNR': metric.item()})
                wandb.log({"loss": loss.item(), "PSNR": metric.item()})

        scheduler.step()

        # 开启评估模式
        model.eval()
        # 设置本轮的logger
        metric_logger = AverageLogger()
        # 设置进度条
        with tqdm(total=val_samples_size, ncols=100) as p_bar:
            # 设置进度条描述
            p_bar.set_description(f'Val: {epoch + 1}')
            # 分批次
            for batch in val_dataloader:
                lr_images, gt_images = batch
                lr_images = lr_images.to(device)
                gt_images = gt_images.to(device)

                # 取消梯度的计算
                with torch.no_grad():
                    # 前向传播，得到模型输出
                    hr_images = model(lr_images)

                    # 计算指标
                    metric = peak_signal_to_noise_ratio(gt_images, hr_images)

                    # 记录指标
                    metric_logger.update(metric.item())

                # 更新进度条&显示指标
                p_bar.update(batch_size)
                p_bar.set_postfix({'metric': metric_logger.avg})

            # if current epoch has the best metric, update it
            if metric_logger.avg > best_metric:
                best_metric = metric_logger.avg
                best_epoch = epoch + 1
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, os.path.join(save_path, 'best_dormitory.pth'))

    print(f'Complete! \n The best epoch is {best_epoch} \n The best metric is {best_metric}')
