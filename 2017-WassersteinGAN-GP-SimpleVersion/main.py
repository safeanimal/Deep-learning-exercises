import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.backends import cudnn
from torchvision.transforms import transforms
import torchvision.datasets as tvd
import torchvision.transforms.functional as tvtf
from models import Generator, Discriminator
from tqdm import tqdm
import wandb
import utils

if __name__ == '__main__':
    nz = 100
    img_size = 128

    folder_path = 'D:/Materials/dataset/anime-face-256by256'
    # folder_path = 'D:/Materials/dataset/celeba'
    samples_num = 92219

    # 创建dataset
    full_dataset = tvd.ImageFolder(folder_path, transform=transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=tvtf.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ]))

    indices = torch.randperm(len(full_dataset))[:samples_num]
    train_dataset = Subset(dataset=full_dataset, indices=indices)

    # 创建dataloader
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    # 查看部分待训练图片
    train_imgs, _ = next(iter(train_dataloader))
    train_imgs = train_imgs[:16]
    utils.show_images(4, train_imgs)

    # 申请设备变量
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # 自定义的模型参数初始化函数
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    # 定义模型并转移到设备
    G = Generator().to(device)
    D = Discriminator().to(device)

    # 用自定义的模型初始化函数初始化模型参数
    G.apply(weights_init)
    D.apply(weights_init)

    lr = 0.0001
    betas = (0.5, 0.999)
    # 定义优化器
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    # 开启cudnn.benchmark
    cudnn.benchmark = True

    # 设置epoch_num
    epoch_num = 15

    # 设置一个epoch中Discriminator训练几次
    dis_train_num = 2

    wandb.init(
        # set the wandb project where this run will be logged
        project="Basic GAN",

        # track hyperparameters and run metadata
        config={
            "architecture": "WGAN",
            "dataset": "another anime face dataset",
            "samples_num": samples_num,
            "learning_rate": lr,
            "Generator": str(G),
            "Discriminator": str(D),
            "epochs": epoch_num,
            "dis_train_num": dis_train_num,
            "batch_size": batch_size,
            "optimizer_G": optimizer_G.__class__,
            "betas_G": betas,
            "optimizer_D": optimizer_D.__class__,
            "betas_D": betas
        }
    )

    # 开始训练
    for epoch in range(epoch_num):
        D.train()
        G.train()
        with tqdm(total=samples_num, ncols=100) as p_bar:
            p_bar.set_description(f'Epoch: {epoch + 1}')
            for data in train_dataloader:
                real_imgs, _ = data
                real_imgs = real_imgs.to(device)

                for i in range(dis_train_num):
                    # generate noise
                    noise_samples = torch.randn((len(real_imgs), nz, 1, 1)).to(device)

                    # generate fake images from these noise
                    fake_imgs = G(noise_samples).to(device)

                    # ---------train Discriminator-----------------#
                    # clear the gradients of Discriminator
                    D.zero_grad()

                    # mix images
                    epsilon = torch.randn_like(real_imgs)
                    mix_imgs = epsilon * fake_imgs + (1 - epsilon) * real_imgs
                    mix_scores = D(mix_imgs)

                    # compute the gradient of D(mix_imgs) with respect to mix_imgs
                    gradient = torch.autograd.grad(
                        # take the gradient of outputs with respect to inputs.
                        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
                        inputs=mix_imgs,
                        outputs=mix_scores,
                        # These other parameters have to do with the pytorch autograd engine works
                        grad_outputs=torch.ones_like(mix_scores),
                        create_graph=True,
                        retain_graph=True,
                    )[0]

                    # Flatten the gradients so that each row captures one image
                    gradient = gradient.view(len(gradient), -1)

                    # Calculate the magnitude of every row
                    gradient_norm = gradient.norm(2, dim=1)

                    # set lambda
                    lmbda = 5
                    loss_D = D(fake_imgs.detach()).mean() - D(real_imgs).mean() + lmbda * (
                                (gradient_norm - 1) ** 2).mean()

                    # compute gradients
                    loss_D.backward()
                    # clip the gradients
                    # torch.nn.utils.clip_grad_norm_(D.parameters(), 0.01)
                    # update parameters
                    optimizer_D.step()

                # ------------train Generator-----------------#
                # clear gradients
                G.zero_grad()

                # generate noise
                noise_samples2 = torch.randn((len(real_imgs), nz, 1, 1)).to(device)
                fake_imgs2 = G(noise_samples2)
                loss_G = -D(fake_imgs2).mean()

                # compute gradients
                loss_G.backward()
                # clip the gradients
                # torch.nn.utils.clip_grad_norm_(G.parameters(), 0.01)
                # optimize parameters
                optimizer_G.step()

                # log loss to wandb
                wandb.log({'loss_D': loss_D.item(), 'loss_G': loss_G.item(), 'D(G(z))': -loss_G.item()})

                p_bar.update(len(real_imgs))

        D.eval()
        G.eval()
        # 训练完成后生成结果
        with torch.no_grad():
            noise_samples = torch.randn((16, nz, 1, 1)).to(device)
            fake_imgs = G(noise_samples).cpu()

        utils.show_images(4, fake_imgs)
