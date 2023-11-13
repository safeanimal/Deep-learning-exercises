"""
---
title: Generate images using stable diffusion with a prompt from a given image
summary: >
 Generate images using stable diffusion with a prompt from a given image
---

# Generate images using [stable diffusion](../index.html) with a prompt from a given image
"""
# TODO: 有一个想法，保持image->image，但是添加一个条件，也就是使用GT-HR的残差，编码后加入模型中进行训练，目标函数为L1或MSE（GT和HR）
#   写一个Dataset，每次获取HR (插值得到), GT, Res，模型将prompts置空，输入HR（当作img2img中的输入img）和Res（编码后加入stable diffusion），期望获得GT
#   Res通过RRDB或者SwinTransformerLayer进行提取特征，然后加入Unet
import argparse
from pathlib import Path

import torch

from labml import lab, monit
from ddim import DDIMSampler
from util import load_model, load_img, save_images, set_seed


class Img2Img:
    """
    ### Image to image class
    """

    def __init__(self, *, checkpoint_path: Path,
                 ddim_steps: int = 50,
                 ddim_eta: float = 0.0):
        """
        :param checkpoint_path: is the path of the checkpoint
        :param ddim_steps: is the number of sampling steps
        :param ddim_eta: is the [DDIM sampling](../sampler/ddim.html) $\eta$ constant
        """
        self.ddim_steps = ddim_steps

        # Load [latent diffusion model](../latent_diffusion.html)
        self.model = load_model(checkpoint_path)
        # Get device
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # Move the model to device
        self.model.to(self.device)

        # Initialize [DDIM sampler](../sampler/ddim.html)
        self.sampler = DDIMSampler(self.model,
                                   n_steps=ddim_steps,
                                   ddim_eta=ddim_eta)

    @torch.no_grad()
    def __call__(self, *,
                 dest_path: str,
                 orig_img: str,
                 strength: float,
                 batch_size: int = 3,
                 prompt: str,
                 uncond_scale: float = 5.0,
                 ):
        """
        :param dest_path: is the path to store the generated images
        :param orig_img: is the image to transform
        :param strength: specifies how much of the original image should not be preserved
        :param batch_size: is the number of images to generate in a batch
        :param prompt: is the prompt to generate images with
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        """
        # Make a batch of prompts
        prompts = batch_size * [prompt]
        # Load image
        orig_image = load_img(orig_img).to(self.device)
        # Encode the image in the latent space and make `batch_size` copies of it
        orig = self.model.autoencoder_encode(orig_image).repeat(batch_size, 1, 1, 1)

        # Get the number of steps to diffuse the original
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_index = int(strength * self.ddim_steps)

        # AMP auto casting
        with torch.cuda.amp.autocast():
            # In unconditional scaling is not $1$ get the embeddings for empty prompts (no conditioning).
            if uncond_scale != 1.0:
                un_cond = self.model.get_text_conditioning(batch_size * [""])
            else:
                un_cond = None
            # Get the prompt embeddings
            # TODO: 这里的文本cond要变成HR-GT的Res编码cond
            cond = self.model.get_text_conditioning(prompts)
            # Add noise to the original image
            x = self.sampler.q_sample(orig, t_index)
            # Reconstruct from the noisy image
            x = self.sampler.paint(x, cond, t_index,
                                   uncond_scale=uncond_scale,
                                   uncond_cond=un_cond)
            # Decode the image from the [autoencoder](../model/autoencoder.html)
            images = self.model.autoencoder_decode(x)

        # Save images
        save_images(images, dest_path, 'img_')


def main():
    """
    ### CLI
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a blue hair anime girl",
        help="the prompt to render"
    )

    parser.add_argument(
        "--orig-img",
        type=str,
        nargs="?",
        default='D:/Materials/dataset/anime-face-512by512/portraits/18255660.jpg',
        help="path to the input image"
    )

    parser.add_argument("--batch_size", type=int, default=4, help="batch size", )
    parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps")

    parser.add_argument("--scale", type=float, default=5.0,
                        help="unconditional guidance scale: "
                             "eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")

    parser.add_argument("--strength", type=float, default=0.75,
                        help="strength for noise: "
                             " 1.0 corresponds to full destruction of information in init image")

    opt = parser.parse_args()
    set_seed(42)

    img2img = Img2Img(checkpoint_path=Path("E:/AI/checkpoints/stable-diffusion-v-1-4-original/sd-v1-4.ckpt"),
                      ddim_steps=opt.steps)

    with monit.section('Generate'):
        img2img(
            dest_path='outputs',
            orig_img=opt.orig_img,
            strength=opt.strength,
            batch_size=opt.batch_size,
            prompt=opt.prompt,
            uncond_scale=opt.scale)


#
if __name__ == "__main__":
    main()
