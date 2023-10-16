import argparse
import utils
from PIL import Image
from model import SuperResolutionModel
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf
import torch
import copy
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('input_image_path', nargs='?', type=str,
                    default='D:/Materials/dataset/train/DIV2K 2018/DIV2K_valid_LR_unknown_X4_sub/')
parser.add_argument('input_image_name', nargs='?', type=str, default='0802_s024.png')
parser.add_argument('target_image_path', nargs='?', type=str, default='test')
parser.add_argument('img_size', nargs='?', type=int, default=480)
parser.add_argument('patch_size', nargs='?', type=int, default=120)
parser.add_argument('upsampling_scale', nargs='?', choices=[2, 4, 8], default=4)
parser.add_argument('state_dict_path', nargs='?', default='save/8blocks.pth')
args = parser.parse_args()

# 选择GPU为device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = SuperResolutionModel(num_blocks=8, img_size=args.patch_size).to(device)
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