from typing import Any

import lightning as L
import torch.optim
from sr_quantization import SRQuantization
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import ImageDataset


class SRQuantizationLightning(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SRQuantization(ddim_steps=50)

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log('loss', value=loss, prog_bar=True)
        print(f'loss: {loss}')
        return loss

    def validation_step(self, batch, batch_idx):
        mse = self.get_loss(batch, batch_idx)
        max_pixel_value = 1.0
        psnr = 10 * torch.log10((max_pixel_value ** 2) / mse)
        self.log('PSNR', value=psnr, prog_bar=True)
        return psnr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-5)
        return optimizer

    def get_loss(self, batch, batch_idx):
        lr, gt = batch
        hr = self.model(lr)
        loss = F.mse_loss(hr, gt)
        return loss


train_dataset = ImageDataset(lr_img_folder='D:/Materials/dataset/DIV2K 2018/DIV2K_train_LR_unknown_X4_sub',
                             gt_img_folder='D:/Materials/dataset/DIV2K 2018/DIV2K_train_HR_sub')
val_dataset = ImageDataset(lr_img_folder='D:/Materials/dataset/DIV2K 2018/DIV2K_valid_LR_unknown_X4_sub',
                           gt_img_folder='D:/Materials/dataset/DIV2K 2018/DIV2K_valid_HR_sub')

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

model_lightning = SRQuantizationLightning().to('cuda:0')
model_lightning.model.load(unet_checkpoint_path='E:/AI/checkpoints/stable-diffusion-v-1-4-original/sd-v1-4-unet.ckpt',
                           auto_encoder_checkpoint_path='E:/AI/checkpoints/stable-diffusion-v-1-4-original/sd-v1-4-auto-encoder.ckpt')
trainer = L.Trainer(max_epochs=1)
trainer.fit(model_lightning, train_dataloaders=train_loader, val_dataloaders=val_loader)
