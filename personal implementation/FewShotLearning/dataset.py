import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, image1_path, image2_path):
        super().__init__()
        self.image1 = Image.open(image1_path).convert('RGB')
        self.image2 = Image.open(image2_path).convert('RGB')

    def __getitem__(self, item):
        return self.image1, self.image2

    def __len__(self):
        return 1
