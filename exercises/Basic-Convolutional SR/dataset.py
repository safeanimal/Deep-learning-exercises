from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import os
from PIL import Image


class ImgDataset(Dataset):
    """
    训练图片均裁剪为256*256

    __init__:
        img_folder: 训练集的路径
        scale_factor: 图片应该放大的倍数

    __getitem__: 产生训练对256//scale_factor * 256//scale_factor <-> 256 * 256
    """

    def __init__(self, img_folder, scale_factor, num_samples=None):
        super().__init__()
        self.img_folder = img_folder
        self.image_files = sorted(os.listdir(img_folder))

        if num_samples is not None:
            if 0 < num_samples <= len(self.image_files):
                self.image_files = self.image_files[:num_samples]
            else:
                raise ValueError("num_samples must be between 0 and len(dataset)")

        self.input_transform = transforms.Compose([
            transforms.Resize((256 // scale_factor, 256 // scale_factor), interpolation=TF.InterpolationMode.BICUBIC,
                              antialias=True),
        ])
        self.target_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=TF.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path = os.path.join(self.img_folder, self.image_files[index])
        img = Image.open(img_path).convert('RGB')
        target_img = self.target_transform(img)
        input_img = self.input_transform(target_img)
        return input_img, target_img

    def __len__(self):
        return len(self.image_files)
