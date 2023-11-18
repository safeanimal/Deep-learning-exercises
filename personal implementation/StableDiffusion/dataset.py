from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms, InterpolationMode
import os
from PIL import Image


class SRDataset(Dataset):
    """
    return the residual between GT image and upsampled LR image and the GT image
    """

    def __init__(self, lr_img_folder: str, gt_img_folder: str):
        super(SRDataset, self).__init__()
        self.lr_img_names = sorted(os.listdir(lr_img_folder))
        self.gt_img_names = sorted(os.listdir(gt_img_folder))
        self.lr_img_folder = lr_img_folder
        self.gt_img_folder = gt_img_folder
        # if 0 < samples_size <= len(self.gt_img_names):
        #     self.img_names = self.gt_img_names[:samples_size]
        # else:
        #     raise ValueError('the samples_size should be in the range of [1, img_number]')

        self.transform = transforms.Compose([

            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        gt_img_name = self.gt_img_names[item]
        gt_img_path = os.path.join(self.gt_img_folder, gt_img_name)
        gt_img = Image.open(gt_img_path).convert('RGB')

        gt_img = self.transform(gt_img)

        lr_img_name = self.lr_img_names[item]
        lr_img_path = os.path.join(self.lr_img_folder, lr_img_name)
        lr_img = Image.open(lr_img_path).convert('RGB')

        lr_img = self.transform(lr_img)

        hr_img = transforms.Resize(size=gt_img.shape[1:], interpolation=InterpolationMode.BICUBIC)(lr_img)

        res_img = gt_img - hr_img
        return lr_img, gt_img, res_img

    def __len__(self):
        return len(self.lr_img_names)


# if __name__ == "__main__":
#     dataset = SRDataset(lr_img_folder="H:/dataset/DIV2K/DIV2K_train_LR_unknown_X4_sub", gt_img_folder="H:/dataset/DIV2K/DIV2K_train_HR_sub")
#     data = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
#     for batch in data:
#         print(batch[0].shape, batch[1].shape)
#         break

