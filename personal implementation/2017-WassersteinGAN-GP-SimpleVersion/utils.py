import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


def show_images(images_per_row, images_tensor):
    # Create a grid of images using make_grid
    grid = vutils.make_grid(images_tensor, nrow=images_per_row, padding=2, normalize=True)

    # Convert the grid tensor to a numpy array and transpose the dimensions for displaying
    grid_np = grid.numpy().transpose((1, 2, 0))

    # Display the grid of images using matplotlib
    plt.figure(figsize=(15, 15))
    plt.imshow(np.clip(grid_np, 0, 1))
    plt.axis("off")
    plt.show()
