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

    def __init__(self, img_size, upsampling_scale, img_folder, samples_size):
        super().__init__()
        self.img_names = sorted(os.listdir(img_folder))
        self.img_folder = img_folder
        if 0 < samples_size <= len(self.img_names):
            self.img_names = self.img_names[:samples_size]
        else:
            raise ValueError('the samples_size should be in the range of [1, img_number]')

        self.input_transform = transforms.Compose([
            transforms.Resize((img_size // upsampling_scale, img_size // upsampling_scale),
                              interpolation=tvf.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        self.target_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=tvf.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        img_name = self.img_names[item]
        img_path = os.path.join(self.img_folder, img_name)
        img = Image.open(img_path).convert('RGB')

        input_img = self.input_transform(img)
        target_img = self.target_transform(img)

        return input_img, target_img

    def __len__(self):
        return len(self.img_names)
