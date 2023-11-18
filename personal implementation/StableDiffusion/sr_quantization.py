import torch
import torch.nn as nn
from unet import UNetModel
from autoencoder import Autoencoder, Encoder, Decoder
from vector_quantize_pytorch import FSQ
from ddim import DDIMSampler
from latent_diffusion import LatentDiffusion
from clip_embedder import CLIPTextEmbedder
import torch.nn.functional as F
from util import save_images, load_img

class SRQuantization(nn.Module):
    def __init__(self, ddim_steps: int = 50):
        super().__init__()
        self.unet = UNetModel(in_channels=4,
                              out_channels=4,
                              channels=320,
                              attention_levels=[0, 1, 2],
                              n_res_blocks=2,
                              channel_multipliers=[1, 2, 4, 4],
                              n_heads=8,
                              tf_layers=1,
                              d_cond=768)
        self.levels_before = [8, 6, 5] # this is for 3-dimension image, if 4-diemsion, set to [8, 5, 5, 5]

        self.quantizer_before = FSQ(self.levels_before)
        self.encoder = Encoder(z_channels=4,
                               in_channels=3,
                               channels=128,
                               channel_multipliers=[1, 2, 4, 4],
                               n_resnet_blocks=2)

        self.levels_after = [8, 5, 5, 5] # this is for 4-dimension image, if 3-diemsion, set to [8, 6, 5]
        self.quantizer_after = FSQ(self.levels_after)
        self.decoder = Decoder(out_channels=3,
                               z_channels=4,
                               channels=128,
                               channel_multipliers=[1, 2, 4, 4],
                               n_resnet_blocks=2)

        self.autoencoder = Autoencoder(emb_channels=4,
                                       encoder=self.encoder,
                                       decoder=self.decoder,
                                       z_channels=4)
        self.clip_text_embedder = CLIPTextEmbedder()

        self.latent_diffusion = LatentDiffusion(unet_model=self.unet,
                                                autoencoder=self.autoencoder,
                                                clip_embedder=self.clip_text_embedder,
                                                linear_start=0.00085,
                                                linear_end=0.0120,
                                                n_steps=1000,
                                                latent_scaling_factor=0.18215)
        self.sampler = DDIMSampler(self.latent_diffusion, n_steps=ddim_steps)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.ddim_steps = ddim_steps

    def forward(self, lr, uncond_scale: float = 1.0, strength: float = 0.9999):
        lr_up = F.interpolate(lr, mode='bicubic', scale_factor=4)
        b, c, h, w = lr_up.shape
        assert (h, w) == (512, 512), f'h and w must equal to 512, but got h={h}, w={w}'
        prompt = ''
        # Make a batch of prompts
        prompts = b * [prompt]

        # Load image
        orig_image = lr_up.to(self.device)
        # transform lr_up to 3D data to satisfy the input size of quantizer
        orig_q = orig_image.permute(0, 2, 3, 1).contiguous().view(b, -1, c) # (b, h, w, c)->(b, h*w, c)
        # Finite-Scalar Quatization
        orig_image_quantized, _ = self.quantizer_before(orig_q)
        # transform orig_image_quantized back to orig_image's size
        orig_image_quantized = orig_image_quantized.view(b, h, w, c).permute(0, 3, 1, 2)

        # Encode the image in the latent space and make `batch_size` copies of it
        orig = self.latent_diffusion.autoencoder_encode(orig_image_quantized).repeat(b, 1, 1, 1)

        # Get the number of steps to diffuse the original
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_index = int(strength * self.ddim_steps)
        # # AMP auto casting
        # with torch.cuda.amp.autocast():
        # In unconditional scaling is not $1$ get the embeddings for empty prompts (no conditioning).
        if uncond_scale != 1.0:
            un_cond = self.latent_diffusion.get_text_conditioning(b * [""])
        else:
            un_cond = None
        # Get the prompt embeddings
        cond = self.latent_diffusion.get_text_conditioning(prompts)

        # Add noise to the original image
        x = self.sampler.q_sample(orig, t_index)
        # Reconstruct from the noisy image
        x = self.sampler.paint(x, cond, t_index,
                               uncond_scale=uncond_scale,
                               uncond_cond=un_cond)
        # Decode the image from the [autoencoder](../model/autoencoder.html)
        # transform x to 3D data to satisfy the input size of quantizer
        bx, cx, hx, wx = x.shape
        x_q = x.permute(0, 2, 3, 1).contiguous().view(bx, -1, cx)  # (bx, hx, wx, cx)->(bx, hx*wx, cx)
        # Finite-Scalar Quatization
        x_quantized, _ = self.quantizer_after(x_q)
        # transform x_quantized back to x's size
        x_quantized = x_quantized.view(bx, hx, wx, cx).permute(0, 3, 1, 2)
        hr = self.latent_diffusion.autoencoder_decode(x_quantized)
        return hr

    def load(self, unet_checkpoint_path: str, auto_encoder_checkpoint_path: str):
        self.unet.load_state_dict(torch.load(unet_checkpoint_path))
        self.autoencoder.load_state_dict(torch.load(auto_encoder_checkpoint_path))

        self.unet.requires_grad_(False)
        # self.autoencoder.requires_grad_(False)
        self.clip_text_embedder.requires_grad_(False)


# model = SRQuantization().to('cuda:0')
# model.load(unet_checkpoint_path='E:/AI/checkpoints/stable-diffusion-v-1-4-original/sd-v1-4-unet.ckpt',
#            auto_encoder_checkpoint_path='E:/AI/checkpoints/stable-diffusion-v-1-4-original/sd-v1-4-auto-encoder.ckpt')
# lr = load_img('D:/Materials/dataset/DIV2K 2018/DIV2K_valid_LR_unknown_X4_sub/0855_s018.png')
# hr = model(lr)
# save_images(hr, dest_path='outputs', prefix='sr_img_')
# levels = [8,5,5,5] # see 4.1 and A.4.1 in the paper
# quantizer = FSQ(levels)
#
#
#
# x = torch.randn(1, 1024, 4) # 4 since there are 4 levels
# xhat, indices = quantizer(x)
#
# print(xhat.shape)    # (1, 1024, 4) - (batch, seq, dim)
# print(indices.shape) # (1, 1024)    - (batch, seq)
#
# assert xhat.shape == x.shape
# assert torch.all(xhat == quantizer.indices_to_codes(indices))