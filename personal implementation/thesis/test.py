import argparse

import yaml

import utils
from PIL import Image
from model2 import SuperResolutionModel
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf
import torch
import copy
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('input_image_path', nargs='?', type=str,
                    default='D:/Materials/dataset/DIV2K 2018/DIV2K_valid_LR_unknown_X4_sub/')
parser.add_argument('input_image_name', nargs='?', type=str, default='0831_s013.png')
parser.add_argument('target_image_path', nargs='?', type=str, default='test')
parser.add_argument('state_dict_path', nargs='?', default='save/best.pth')
args = parser.parse_args()

with open('config.yml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# 选择GPU为device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

hyperparameters_config = config['hyperparameters']
model_num = hyperparameters_config['model_num']

# 加载模型
model_parameters = config['saved_model'+str(model_num)]
image_size = model_parameters['image_size']
if model_num == 0:
    num_blocks = model_parameters['num_blocks']
    model = SuperResolutionModel(num_blocks=num_blocks, image_size=image_size).to(device)
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

model.load_state_dict(torch.load(args.state_dict_path))

model.eval()

original_image = Image.open(args.input_image_path+args.input_image_name)

input_transform = tvt.Compose([
    tvt.ToTensor()
])

# 图像预处理
original_image = input_transform(original_image)
lr_image = copy.deepcopy(original_image)
lr_image = lr_image.to(device)
lr_image = lr_image.reshape(1, *lr_image.shape)
hr_image = model(lr_image)

lr_image = lr_image.cpu()
hr_image = hr_image.cpu()
utils.show_images_tensor(1, lr_image)
utils.show_images_tensor(1, hr_image)


save_image(lr_image, 'test/num8/lr_image_'+args.input_image_name+'.png')
save_image(hr_image, 'test/num8/hr_image_'+args.input_image_name+'.png')
save_image(original_image.squeeze(0), 'test/num8/original_image_'+args.input_image_name+'.png')