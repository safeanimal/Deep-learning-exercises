import argparse
import utils
from PIL import Image
from model import Model
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf
import torch

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('input_image_path', nargs='?', type=str,
                    default='D:/Materials/dataset/anime-face-256by256/animefaces256cleaner/19304_result.jpg')
parser.add_argument('target_image_path', nargs='?', type=str, default='test')
parser.add_argument('img_size', nargs='?', type=int, default=64)
parser.add_argument('state_dict_path', nargs='?', default='save/best.pth')
args = parser.parse_args()

# 选择GPU为device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Model().to(device)
model.load_state_dict(torch.load(args.state_dict_path))

model.eval()

lr_image = Image.open(args.input_image_path)

input_transform = tvt.Compose([
    tvt.Resize((args.img_size // args.upsampling_scale, args.img_size // args.upsampling_scale),
               interpolation=tvf.InterpolationMode.BICUBIC),
    tvt.ToTensor()
])

# 图像预处理
lr_image = input_transform(lr_image)
lr_image = lr_image.to(device)

# 分块
lr_image_patches = torch.stack(utils.crop_patches(lr_image, args.patch_size))

patches_num_one_row = args.img_size // args.patch_size

# hr_image_patches形状为[patches_num_one_row*patches_num_one_row]
# 是一个patches_num_one_row*patches_num_one_row个patches组成的，一个patches的大小为3*patch_size*patch_size
# 要将它变为3*W*H的Tensor
hr_image_patches = model(lr_image_patches)
hr_image_patches = hr_image_patches.reshape(patches_num_one_row, patches_num_one_row, 3, args.patch_size, args.patch_size)

# 还原为原始图像
hr_image = hr_image_patches.permute(0, 1, 2, 3, 4) \
    .reshape(3, patches_num_one_row * args.patch_size, patches_num_one_row * args.patch_size)

hr_image = hr_image.cpu()
utils.show_images_tensor(1, hr_image)
