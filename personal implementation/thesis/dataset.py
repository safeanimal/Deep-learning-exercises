from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import torchvision.transforms.functional as tvf
import os


class ImageDataset(Dataset):
    """
    read [samples_size] [img_size] images from [img_folder]
    these images are preprocessed with [upsampling_scale]
    """

    def __init__(self, lr_img_folder, hr_img_folder):
        super().__init__()
        self.lr_img_names = sorted(os.listdir(lr_img_folder))
        self.hr_img_names = sorted(os.listdir(hr_img_folder))
        self.lr_img_folder = lr_img_folder
        self.hr_img_folder = hr_img_folder
        # if 0 < samples_size <= len(self.hr_img_names):
        #     self.img_names = self.hr_img_names[:samples_size]
        # else:
        #     raise ValueError('the samples_size should be in the range of [1, img_number]')

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        lr_img_name = self.lr_img_names[item]
        lr_img_path = os.path.join(self.lr_img_folder, lr_img_name)
        lr_img = Image.open(lr_img_path).convert('RGB')

        lr_img = self.transform(lr_img)

        hr_img_name = self.hr_img_names[item]
        hr_img_path = os.path.join(self.hr_img_folder, hr_img_name)
        hr_img = Image.open(hr_img_path).convert('RGB')

        hr_img = self.transform(hr_img)
        return lr_img, hr_img

    def __len__(self):
        return len(self.lr_img_names)
