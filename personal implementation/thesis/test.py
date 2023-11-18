import os
import yaml
from metric import peak_signal_to_noise_ratio
from PIL import Image
from model0 import SuperResolutionModel
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf
import torch
import copy
from torchvision.utils import save_image
from logger import AverageLogger
import datetime

with open('test_config.yml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

test_dirs = config['test_dirs']
test_type = config['test_type']

log_path = config['log_path']

model_parameters_path = config['model_parameters_path']

# 选择GPU为device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = SuperResolutionModel()
model.load_state_dict(torch.load(model_parameters_path))

# 开启推理模式
model.eval()

input_transform = tvt.Compose([
    tvt.ToTensor()
])

# 创建记录器
avg_logger = AverageLogger()

if not os.path.exists(log_path):
    os.mkdir(log_path)

with open('test_log.txt', mode='a', encoding='uft-8') as file:
    file.writelines('Date:', datetime.datetime.now())
    # 以此测试每一个图片集
    for test_dir in test_dirs:
        image_names = os.listdir(test_dir)
        LR_image_paths = [os.path.join(test_dir, image_name) for image_name in image_names if image_name[-6:-4] == test_type]
        GT_image_paths = [os.path.join(test_dir, image_name) for image_name in image_names if image_name[-6:-4] == 'HR']
        print(LR_image_paths)
        print(GT_image_paths)
        for i in range(len(LR_image_paths)):
            lr_image = Image.open(LR_image_paths[i]).convert('RGB')
            lr_image = input_transform(lr_image).to(device)

            gt_image = Image.open(GT_image_paths[i]).convert('RGB')
            gt_image = input_transform(gt_image).to(device)

            hr_image = model(lr_image)

            psnr = peak_signal_to_noise_ratio(hr_image, gt_image)
            avg_logger.update(psnr)

        file.writelines(f'For {test_dir} \n\tPSNR (avg):{avg_logger.avg}\n')
        avg_logger.clear()

# original_image = Image.open(args.input_image_path+args.input_image_name)
#
# # 图像预处理
# original_image = input_transform(original_image)
# lr_image = copy.deepcopy(original_image)
# lr_image = lr_image.to(device)
# lr_image = lr_image.reshape(1, *lr_image.shape)
# hr_image = sr_models(lr_image)
#
# lr_image = lr_image.cpu()
# hr_image = hr_image.cpu()
# utils.show_images_tensor(1, lr_image)
# utils.show_images_tensor(1, hr_image)
#
#
# save_image(lr_image, 'test/num8/lr_image_'+args.input_image_name+'.png')
# save_image(hr_image, 'test/num8/hr_image_'+args.input_image_name+'.png')
# save_image(original_image.squeeze(0), 'test/num8/original_image_'+args.input_image_name+'.png')