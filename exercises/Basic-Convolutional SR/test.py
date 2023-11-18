import torch
from PIL import Image
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF

from model import SRG

# Load the pre-trained sr_models
model_path = 'outputs\\best.pth'
model = SRG()
model.load_state_dict(torch.load(model_path))
model.eval()

# When you use ToTensor(), the pixel values are automatically normalized to the range [0, 1]. The ToPILImage()
# function is designed to convert the tensors back to the original range of [0, 255] for the PIL Image. So,
# you don't need to multiply the pixel values by 255 manually.

"""The reason for using unsqueeze and squeeze is because your sr_models expects a 4-dimensional input tensor with the 
shape (batch_size, channels, height, width), while the output of the preprocess_image function is a 3-dimensional 
tensor with the shape (channels, height, width). unsqueeze(0) is used to add an extra dimension at the beginning of 
the tensor, effectively converting it into a batch of size 1. This way, the input tensor will have the required shape 
(1, channels, height, width) that the sr_models expects. After passing the input tensor through the sr_models, the output 
tensor will also have the shape (1, channels, height, width). However, to convert it back to an image using the 
postprocess_image function, we need a 3-dimensional tensor with the shape (channels, height, width). That's where 
squeeze(0) comes in – it removes the first dimension (batch_size) from the tensor, leaving us with the required 
3-dimensional tensor."""


# Define a function to preprocess the input image
def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.Resize((256 // 4, 256 // 4), interpolation=TF.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    return preprocess(img)


# Define a function to post-process the output image
def postprocess_image(tensor):
    return transforms.ToPILImage()(tensor)


# Load and preprocess the input image
input_image_path = 'C:/Users/41181/OneDrive/Documents/Deep learning/datasets/train/Anime face/images/239_2000.jpg'
input_image = Image.open(input_image_path).convert('RGB')
input_tensor = preprocess_image(input_image).unsqueeze(0)

# Pass the input image through the sr_models
with torch.no_grad():
    output_tensor = model(input_tensor)

# Post-process the output image and save it
output_image = postprocess_image(output_tensor.squeeze(0))
output_image_path = 'outputs\\4x 239_2000.jpg'
output_image.save(output_image_path)
