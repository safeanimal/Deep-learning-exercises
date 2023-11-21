import torch
import yaml

from utils import show_images_tensor
from diffusion_model import GenerativeModel

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# 读取diffusion的配置
with open('../diffusion_config.yml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

image_folder = config['image_folder']

batch_size = config['batch_size']
image_size = config['image_size']
channels = 3

T = config['T']
beta_min = config['beta_min']
beta_max = config['beta_max']

first_element = torch.Tensor([0]).to(device)
beta = torch.linspace(beta_min, beta_max, T).to(device)
# 在首位添加元素0，这样time step就和下标一一对应
beta = torch.cat((first_element, beta))
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
sigma = torch.sqrt(beta)
image_shape = (16, 3, image_size, image_size)


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


model = GenerativeModel(image_size=image_size).to(device)

model.load_state_dict(torch.load('saved_model/sr_models.pth'))
model.eval()  # 开启评估模式！
with torch.no_grad():
    noise = torch.randn(*image_shape).to(device)
    sampling_results = sampling(noise, T, model, 1, T, 50)
sampling_results = sampling_results.detach().cpu()
show_images_tensor(5, sampling_results.reshape(*image_shape).cpu())
