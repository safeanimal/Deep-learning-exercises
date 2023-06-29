import os

import torch
from torchvision.transforms import transforms
import torchvision.transforms.functional as ttf
from model import SRG2
from tqdm import tqdm
from PIL import Image
from utils import AverageLogger, calculate_psnr


def preprocessing(img):
    """

    :param img: PIL Image
    :return: Tensor
    """
    pre_transform = transforms.Compose([
        transforms.Resize((64, 64), interpolation=ttf.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    return pre_transform(img)


def postprocessing(img):
    """

    :param img: Tensor
    :return: PIL image
    """
    return transforms.ToPILImage()(img)


def scale_bicubic(img):
    """

    :param img: PIL image
    :return: Tensor
    """
    bicubic_transform = transforms.Compose([
        transforms.Resize((64, 64), interpolation=ttf.InterpolationMode.BICUBIC),
        transforms.Resize((256, 256), interpolation=ttf.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    return bicubic_transform(img)


test_inputs_dir = 'C:\\Users\\Administrator\\OneDrive\\Documents\\Deep learning\\datasets\\train\\Anime face\\images'
test_outputs_dir = 'test_outputs'
weights_dir = 'outputs/best.pth'
test_samples_size = 10

model = SRG2()
model.load_state_dict(torch.load(weights_dir))

test_img_names = os.listdir(test_inputs_dir)[:test_samples_size]

model.eval()

with tqdm(len(test_img_names), ncols=100) as pbar:
    # logger for the psnr of the reconstructed HR images and bicubic interpolated images
    hr_psnr_logger = AverageLogger()
    bicubic_psnr_logger = AverageLogger()

    for img_name in test_img_names:
        pbar.set_description(f'Processing image {img_name}')

        img_dir = os.path.join(test_inputs_dir, img_name)
        input_img = Image.open(img_dir).convert('RGB')  # PIL image
        input_img_copy = input_img.copy()   # PIL image

        # unsqueeze to add one dimension (1, channels, height, width)
        input_img = preprocessing(input_img).unsqueeze(0)   # Tensor

        with torch.no_grad():
            output_img = model(input_img)   # Tensor

        pbar.set_postfix({'psnr': 0})
        pbar.update(1)

        hr_img = postprocessing(output_img.squeeze(0))  # PIL image
        bicubic_img = scale_bicubic(input_img_copy)
        bicubic_img = postprocessing(bicubic_img)

        hr_img_path = os.path.join(test_outputs_dir, f'x4 {img_name}')
        bicubic_img_path = os.path.join(test_outputs_dir, f'x4 bicubic {img_name}')
        
        hr_img.save(hr_img_path)
        bicubic_img.save(bicubic_img_path)

    print(f'psnr for reconstructed HR image: {hr_psnr_logger.avg}')
    print(f'psnr for bicubic image: {bicubic_psnr_logger.avg}')