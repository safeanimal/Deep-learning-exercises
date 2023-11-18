import random

import torch
import lightning as L
from ddim import DDIMSampler
from ddpm import DDPMSampler
from super_resolution import SuperResolution
import torch.nn.functional as F
from dataset import SRDataset
from torch.utils.data import DataLoader


class SuperResolutionLightning(L.LightningModule):
    def __init__(self,
                 sampler_name: str,
                 auto_encoder_checkpoint: str,
                 unet_checkpoint: str,
                 ddim_eta: float = 0.0,
                 n_steps: int = 1000,
                 n_steps_ddim: int = 50,
                 input_resolution: tuple = [128, 128]
                 ):
        super().__init__()
        self.model = SuperResolution(linear_start=0.00085, linear_end=0.0120, n_steps=n_steps,
                                     latent_scaling_factor=0.18215, input_resolution=input_resolution)
        # TODO: need a encoder and a decoder which are suitable for super-resolution task,
        #   insight: pre-train a swinIR, take the reconstruction part as decoder
        #   take the remaining part as encoder
        self.n_steps = n_steps
        # Initialize [sampler](../sampler/index.html)
        if sampler_name == 'ddim':
            self.sampler = DDIMSampler(self.model,
                                       n_steps=n_steps_ddim,
                                       ddim_eta=ddim_eta)
        elif sampler_name == 'ddpm':
            self.sampler = DDPMSampler(self.model)

    def training_step(self, batch, batch_idx):
        # TODO: freeze the parameters of unet, only train the encoder and decoder
        #  encode LR_image as x_T
        #   for t = T to 1:
        #       x_{t-1} = denoise using sr_models(x_t, t)
        #   decode x_1 as HR_image
        #   compute loss = MSE(HR_image, GT_image)
        #   backward

        loss = self.get_loss(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)

        max_pixel_value = 1.0
        psnr = 10 * torch.log10((max_pixel_value ** 2) / loss)
        return psnr

    def get_loss(self, batch, batch_idx):
        lr, gt, res = batch
        lr = lr.to('cuda:0')
        gt = gt.to('cuda:0')
        res = res.to('cuda:0')

        lr_size = tuple(lr.shape[-2:])
        cond = self.model.get_res_conditioning(lr, lr_size)

        z_T = self.model.autoencoder_encode(res)
        z_0 = self.sampler.sample(cond=cond,
                                  shape=lr.shape,
                                  uncond_scale=7.5,
                                  x_last=z_T)
        hr = self.model.autoencoder_decode(z_0)
        loss = F.mse_loss(gt, hr)
        return loss

if __name__ == '__main__':
    train_dataset = SRDataset(lr_img_folder='H:/dataset/DIV2K/DIV2K_train_LR_unknown_X4_sub', gt_img_folder='H:/dataset/DIV2K/DIV2K_train_HR_sub')
    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=4, shuffle=True)

    val_dataset = SRDataset(lr_img_folder='H:/dataset/DIV2K/DIV2K_valid_HR_sub', gt_img_folder='H:/dataset/DIV2K/DIV2K_valid_LR_unknown_X4_sub')
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=4, shuffle=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = SuperResolutionLightning(sampler_name='ddim',
                                     auto_encoder_checkpoint='G:/checkpoints/stable-diffusion-v-1-4-original/sd-v1-4-auto-encoder.ckpt',
                                     unet_checkpoint='G:/checkpoints/stable-diffusion-v-1-4-original/sd-v1-4-unet.ckpt').to(
        device)
    # trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    for batch_idx, batch in enumerate(train_loader):
        print(model.get_loss(batch, batch_idx).shape)
        break
