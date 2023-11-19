from unet import UNetModel
from autoencoder import Autoencoder, Encoder, Decoder
from ResidualEncoder import ResidualEncoder
import torch.nn as nn
import torch


class SuperResolution(nn.Module):
    def __init__(self,
                 latent_scaling_factor: float,
                 n_steps: int,
                 linear_start: float,
                 linear_end: float,
                 input_resolution: tuple = (128, 128)):
        super().__init__()
        self.encoder = Encoder(z_channels=4,
                               in_channels=3,
                               channels=128,
                               channel_multipliers=[1, 2, 4, 4],
                               n_resnet_blocks=2)

        self.decoder = Decoder(out_channels=3,
                               z_channels=4,
                               channels=128,
                               channel_multipliers=[1, 2, 4, 4],
                               n_resnet_blocks=2)
        self.cond_stage_model = ResidualEncoder(input_resolution=input_resolution)
        self.first_stage_model = Autoencoder(emb_channels=4, encoder=self.encoder, decoder=self.decoder, z_channels=4)
        self.model = UNetModel(in_channels=4,
                               out_channels=4,
                               channels=320,
                               attention_levels=[0, 1, 2],
                               n_res_blocks=2,
                               channel_multipliers=[1, 2, 4, 4],
                               n_heads=8,
                               tf_layers=1,
                               d_cond=768)  # backbone
        self.latent_scaling_factor = latent_scaling_factor
        # Number of steps $T$
        self.n_steps = n_steps

        # $\beta$ schedule
        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64) ** 2
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        # $\alpha_t = 1 - \beta_t$
        alpha = 1. - beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

    @property
    def device(self):
        """
        ### Get sr_models device
        """
        return next(iter(self.model.parameters())).device

    def get_res_conditioning(self, x: torch.Tensor, x_size: tuple):
        """
        ### Get [CLIP embeddings](sr_models/clip_embedder.html) for a list of text prompts
        """
        return self.cond_stage_model(x, x_size)

    def autoencoder_encode(self, image: torch.Tensor):
        """
        ### Get scaled latent space representation of the image

        The encoder output is a distribution.
        We sample from that and multiply by the scaling factor.
        """
        return self.latent_scaling_factor * self.first_stage_model.encode(image).sample()

    def autoencoder_decode(self, z: torch.Tensor):
        """
        ### Get image from the latent representation

        We scale down by the scaling factor and then decode.
        """
        return self.first_stage_model.decode(z / self.latent_scaling_factor)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        """
        ### Predict noise

        Predict noise given the latent representation $x_t$, time step $t$, and the
        conditioning context $c$.

        $$\epsilon_\text{cond}(x_t, c)$$
        """
        return self.model(x, t, context)

    def load(self, auto_encoder_checkpoint: str, unet_checkpoint: str):
        self.first_stage_model.load_state_dict(torch.load(auto_encoder_checkpoint))
        self.first_stage_model.requires_grad_(False)

        self.model.load_state_dict(torch.load(unet_checkpoint))
        self.model.requires_grad_(False)

# When you call .to('cuda:0') on a PyTorch model, it recursively moves all the model's parameters and buffers to the specified device
# model = SuperResolution(linear_start=0.00085, linear_end=0.0120, n_steps=1000, latent_scaling_factor=0.18215).to('cuda:0')
# model.load(auto_encoder_checkpoint='G:/checkpoints/stable-diffusion-v-1-4-original/sd-v1-4-auto-encoder.ckpt',
#            unet_checkpoint='G:/checkpoints/stable-diffusion-v-1-4-original/sd-v1-4-unet.ckpt')
# x_size = (128, 128)
# x = torch.randn(16, 3, 128, 128).to('cuda:0')
# cond = model.get_res_conditioning(x, x_size)
# print(cond.shape)
