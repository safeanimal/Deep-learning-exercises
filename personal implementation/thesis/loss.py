import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225], device=device)


# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


# TODO:测试在不同的图像SOTA模型下的percetual loss的效果
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)
        self.vgg19_features = models.vgg19(weights=models.VGG19_Weights).features.eval()
        for parameter in self.vgg19_features.parameters():
            parameter.requires_grad = False

    def forward(self, HR, GT):
        HR = self.normalization(HR)
        GT = self.normalization(GT)

        HR_features = self.vgg19_features(HR)
        GT_features = self.vgg19_features(GT)

        perceptual_loss = F.mse_loss(HR_features, GT_features)

        return perceptual_loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def edge_detection(self, images):
        # Convert RGB images to grayscale
        images_gray = torch.mean(images, dim=1, keepdim=True)

        # Define Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float, device=images.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float, device=images.device).unsqueeze(0).unsqueeze(0)

        # Apply Sobel filters to grayscale images
        edges_x = F.conv2d(images_gray, sobel_x, padding=1)
        edges_y = F.conv2d(images_gray, sobel_y, padding=1)

        # Combine results
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        return edges

    def forward(self, HR_images, GT_images):
        HR_edge_map = self.edge_detection(HR_images)
        GT_edge_map = self.edge_detection(GT_images)

        # Visualize the edge maps
        # self.visualize_edge_map(HR_edge_map, title="HR Edge Map")
        # self.visualize_edge_map(GT_edge_map, title="GT Edge Map")

        return F.mse_loss(HR_edge_map, GT_edge_map)

    def visualize_edge_map(self, edge_map, title="Edge Map"):
        # Convert tensor to numpy array
        edge_map_np = edge_map.squeeze().detach().cpu().numpy()
        plt.imshow(edge_map_np, cmap='gray')
        plt.title(title)
        plt.show()


# # Load images
# transform = transforms.Compose([transforms.ToTensor()])
# HR_images = transform(Image.open('D:/Materials/dataset/DIV2K 2018/DIV2K_valid_LR_unknown_X4_sub/0831_s013.png')).unsqueeze(
#     0)
# GT_images = transform(Image.open('D:/Materials/dataset/DIV2K 2018/DIV2K_valid_LR_unknown_X4_sub/0831_s013.png')).unsqueeze(
#     0)
#
# # Create an instance of the EdgeLoss class
# edge_loss_model = EdgeLoss()
#
# # Compute the loss (this will also display the edge maps)
# loss = edge_loss_model(HR_images, GT_images)
# print(loss)
