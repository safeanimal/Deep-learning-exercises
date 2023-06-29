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

    def __init__(self, img_size, img_folder, min_size):
        super().__init__()
        self.all_img_names = sorted(os.listdir(img_folder))
        self.img_folder = img_folder

        self.img_names = []
        # 将低像素的图片过滤掉
        for img_name in self.all_img_names:
            width, height = Image.open(os.path.join(self.img_folder, img_name)).size
            if width >= min_size and height >= min_size:
                self.img_names.append(img_name)

        self.input_transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=tvf.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        self.target_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=tvf.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        img_name = self.all_img_names[item]
        img_path = os.path.join(self.img_folder, img_name)
        img = Image.open(img_path).convert('RGB')

        input_img = self.input_transform(img)
        target_img = self.target_transform(img)

        return input_img, target_img

    def __len__(self):
        return len(self.img_names)
