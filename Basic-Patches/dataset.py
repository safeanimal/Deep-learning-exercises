from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tvt
from torchvision.transforms.functional import InterpolationMode
import os


class ImageDataset(Dataset):
    def __init__(self, folder_path, image_size, patch_size, upsampling_scale):
        """

        :param folder_path: images folder's path
        :param image_size: the height/width of the image
        :param patch_size: the height/width of the patch, it should be divisible by patch size
        :param upsampling_scale: 上采样倍数
        """
        super().__init__()
        self.path_size = patch_size
        self.upsampling_scale = upsampling_scale
        self.patch_num_in_one_image = (image_size // patch_size) ** 2

        image_names = os.listdir(folder_path)
        self.images_path_list = [os.path.join(folder_path, image_name) for image_name in image_names]

    def __getitem__(self, item):
        """
        获取一个(lr,hr)图像块组合
        :param item: indicates which image the patch return from
        :return: a patch of current image
        """
        # 每self.patch_num_in_one_image轮便新加载一批图像块
        if item % self.patch_num_in_one_image == 0:
            image_path = self.images_path_list[item // self.patch_num_in_one_image]
            self.patch_pairs = self.pre_process(image_path, self.path_size)

        current_patch_pair = self.patch_pairs[item % self.patch_num_in_one_image]
        return current_patch_pair[0], current_patch_pair[1]

    def pre_process(self, image_path, patch_size):
        """
        将单个hr图片分块，并与对应的lr图像块组合，返回一个(lr,hr)图像对的列表
        :param image_path: 图像路径
        :param patch_size: 块大小
        :return: 一个含有(lr,hr)图像对的列表
        """
        hr_image = Image.open(image_path)
        hr_image = tvt.ToTensor()(hr_image)
        hr_image_patches = self.crop_patches(hr_image, patch_size)

        lr_image_size = patch_size // self.upsampling_scale
        lr_image_patches = [tvt.Resize(size=lr_image_size, interpolation=InterpolationMode.BICUBIC)(hr_image_patch)
                            for hr_image_patch in hr_image_patches]
        patch_pairs = [(lr_image_patches[i], hr_image_patches[i]) for i in range(len(lr_image_patches))]
        return patch_pairs

    # Function to crop an image into patches
    def crop_patches(self, image, patch_size):
        """
        将一个图片切成若干块
        :param image: 图像
        :param patch_size: 块大小
        :return: 包含图像块的列表
        """
        C, H, W = image.size()
        patches = []
        for y in range(0, H, patch_size):
            for x in range(0, W, patch_size):
                patch = image[:, y:y + patch_size, x:x + patch_size]
                patches.append(patch)
        return patches

    def __len__(self):
        """
        获取图像块的总数
        :return:
        """
        return self.patch_num_in_one_image * len(self.images_path_list)
