import random
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_dataset import ImageDataset
from utils import show_images_tensor
from diffusion_model import GenerativeModel



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    # 读取diffusion的配置
    with open('../diffusion_config.yml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    image_folder = config['image_folder']

    batch_size = config['batch_size']
    num_workers = config['num_workers']
    lr = config['lr']
    epoch = config['epoch']
    save_path = config['save_path']

    image_size = config['image_size']
    channels = 3
    image_shape = (-1, channels, image_size, image_size)

    T = config['T']
    beta_min = config['beta_min']
    beta_max = config['beta_max']

    # 注意下标从0开始
    first_element = torch.Tensor([0]).to(device)
    beta = torch.linspace(beta_min, beta_max, T).to(device)
    # 在首位添加元素0，这样time step就和下标一一对应
    beta = torch.cat((first_element, beta))
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    sigma = torch.sqrt(beta)

    # 打印数据以供检查
    print('beta[0:10]:', beta[0:11])
    print('alpha[0:10]:', alpha[0:11])
    print('alpha_bar[0:10]', alpha_bar[0:11])
    print('sigma[0:10]', sigma[0:11])


    # 给定x0、时刻t计算噪声eps，并按一定方式添加到x0中得到加噪图像xt
    # Checked
    def get_xt_given_x0_and_t(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_sqrt_alpha_bar_t = torch.sqrt(1 - alpha_bar[t]).view(-1, 1, 1, 1)
        mean = sqrt_alpha_bar_t * x0
        std = sqrt_one_minus_sqrt_alpha_bar_t
        xt = mean + std * eps
        return xt


    def show_diffusion(dataloader: DataLoader):
        diffusion = []
        step = 50  # 扩散步长
        # 随机选择一张图片
        idx = random.randint(0, batch_size)
        for batch in dataloader:
            x0 = batch[idx].to(device)
            # 扩散过程
            for t in range(1, T + 1, step):
                eps = torch.randn_like(x0).to(device)
                xt = get_xt_given_x0_and_t(x0, t, eps)
                diffusion.append(xt)
            # 还原为向量
            diffusion = torch.stack(diffusion).reshape(*image_shape)
            break
        show_images_tensor(5, diffusion.cpu())


    # show_diffusion()

    def sampling(xt, T, model, start, end, step):
        """
        xt是LR经缩放得到的HR图像，然后该HR要进一步进行扩散得到最终图像
        xt: (batch_size, channels, hr_image_size, hr_image_size)
        """
        sampling_results = []
        # 计算x_{T-1},...,x0
        for i in range(T, 0, -1):
            z = torch.randn_like(xt).to(device)
            # print('z', z.shape)
            t = (torch.zeros((1,)) + i).long().to(device)
            # print('t', t.shape)
            # ---------------这里模型应该输出一个高分辨率图像--------------------
            eps_theta = model(xt, t).reshape(*image_shape)
            # print('eps_theta', eps_theta.shape)
            assert not torch.any(torch.isnan(eps_theta)), 'in function `sampling`, eps_theta is nan!'

            coeff = 1. / torch.sqrt(alpha[i])
            coeff_eps_theta = (1 - alpha[i]) / torch.sqrt(1 - alpha_bar[i])

            if i == 1:
                z = 0
            xt = coeff * (xt - coeff_eps_theta * eps_theta) + sigma[i] * z
            # print('xt', xt.shape)
            # 存放结果，在[start, end]中每step取一个样本
            # if i >= start and i <= end and (i-1) % step == 0 or i == end:
            # sampling_results.append(xt)

        # sampling_results = torch.stack(sampling_results) # (?, 1, 28, 28)

        return xt  # sampling_results


    def DDIM_sampling(xt, T, model):
        """
        xt: (batch_size, channels, hr_image_size, hr_image_size)
        """
        step_num = 20
        for i in range(T, 0, -step_num):
            one_minus_alpha_tao_minus_one = (1 - alpha_bar[i - step_num]).to(device)
            one_minus_alpha_tao = (1 - alpha_bar[i]).to(device)

            eta = 1  # =1为DDPM；=0为DDIM
            sig_tao = eta * torch.sqrt(one_minus_alpha_tao_minus_one / one_minus_alpha_tao) * torch.sqrt(
                1 - alpha_bar[i] / alpha_bar[i - step_num])
            assert not torch.any(torch.isnan(sig_tao)), 'in function `DDIM_sampling`, sig_tao is nan!'
            t = (torch.zeros((1,)) + i).long().to(device)
            eps_theta = model(xt, t).reshape(*image_shape)
            assert not torch.any(torch.isnan(eps_theta)), 'in function `DDIM_sampling`, eps_theta is nan!'
            f_theta = (xt - torch.sqrt(one_minus_alpha_tao) * eps_theta) / torch.sqrt(alpha_bar[i])
            assert not torch.any(torch.isnan(f_theta)), 'in function `DDIM_sampling`, f_theta is nan!'
            first = torch.sqrt(alpha_bar[i - step_num]) * f_theta
            assert not torch.any(torch.isnan(first)), 'in function `DDIM_sampling`, first is nan!'
            second = torch.sqrt(one_minus_alpha_tao_minus_one - sig_tao ** 2) * eps_theta
            assert not torch.any(torch.isnan(second)), 'in function `DDIM_sampling`, second is nan!'
            z = torch.randn_like(xt)
            third = sig_tao * z
            assert not torch.any(torch.isnan(third)), 'in function `DDIM_sampling`, third is nan!'
            if i > 0:
                xt = first + second + third
            else:
                xt = first + third
        return xt

    # 训练集
    training_dataset = ImageDataset(image_folder=image_folder, image_size=image_size)
    # 训练集大小
    training_samples_size = len(training_dataset)

    # 生成dataloader
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # 展示dataloader里的数据（图片）
    for batch in training_dataloader:
        images = batch
        show_images_tensor(4, images)
        break

    model = GenerativeModel(image_size=image_size).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

    for i in range(epoch):
        with tqdm(total=training_samples_size, ncols=100) as p_bar:
            # 设置进度条描述
            p_bar.set_description(f'Epoch: {i + 1}')
            for input_images in training_dataloader:
                x0 = batch.to(device)
                # print(x0.shape)

                eps = torch.randn_like(x0).to(device)

                t = torch.randint(1, T + 1, (batch_size,)).long().to(device)

                xt = get_xt_given_x0_and_t(x0, t, eps)

                eps_theta = model(xt, t).reshape(*image_shape)

                # assert not torch.isnan(eps_theta).any(), "NaN values found in eps_theta"

                loss = criterion(eps_theta, eps)
                loss.backward()
                # 记录损失
                p_bar.set_postfix({'loss': loss.item()})

                # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # 打印梯度
                # for name, param in model.named_parameters():
                # if param.grad is not None:
                # print(name, torch.max(param.grad))

                # nn.utils.clip_grad_norm_(model.parameters(), 2, norm_type=2) # 梯度裁剪

                optimizer.step()
                optimizer.zero_grad()

                # 更新进度条&显示损失及指标
                p_bar.update(x0.shape[0])
        torch.save(model.state_dict(), save_path+"/model.pth")

    model.eval() # 开启评估模式！
    with torch.no_grad():
        noise = torch.randn(batch_size, *image_shape[1:]).to(device)
        sampling_results = sampling(noise, T, model, 1, T, 50)
    sampling_results = sampling_results.detach().cpu()
    show_images_tensor(5, sampling_results.reshape(*image_shape).cpu())
