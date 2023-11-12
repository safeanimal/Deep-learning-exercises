from typing import Any
import torch
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from unet import UNetModel
from autoencoder import Autoencoder

class SuperResolution(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNetModel() # backbone
        # TODO: need a encoder and a decoder which are suitable for super-resolution task,
        #   insight: pre-train a swinIR, take the reconstruction part as decoder
        #   take the remaining part as encoder

    def training_step(self, batch, batch_idx):

    # TODO: freeze the parameters of unet, only train the encoder and decoder
    #  encode LR_image as x_T
    #   for t = T to 1:
    #       x_{t-1} = denoise using model(x_t, t)
    #   decode x_1 as HR_image
    #   compute loss = MSE(HR_image, GT_image)
    #   backward

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer

    def validation_step(self, batch, batch_idx):

