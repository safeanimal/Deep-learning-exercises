from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    """
    """

    def __init__(self, image_folder, image_size=128):
        super().__init__()
        self.image_names = sorted(os.listdir(image_folder))
        self.image_folder = image_folder
        # if 0 < samples_size <= len(self.hr_img_names):
        #     self.img_names = self.hr_img_names[:samples_size]
        # else:
        #     raise ValueError('the samples_size should be in the range of [1, img_number]')

        self.transform = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        image_name = self.image_names[item]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_names)
