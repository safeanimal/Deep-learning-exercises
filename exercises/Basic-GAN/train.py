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

if __name__ == '__main__':
    nz = 100

    folder_path = 'D:/Materials/dataset/anime-face-256by256'
    # folder_path = 'D:/Materials/dataset/celeba'
    samples_num = 50000

    # 创建dataset
    full_dataset = tvd.ImageFolder(folder_path, transform=transforms.Compose([
        transforms.Resize((64, 64), interpolation=tvtf.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ]))

    indices = torch.randperm(len(full_dataset))[:samples_num]
    train_dataset = Subset(dataset=full_dataset, indices=indices)

    # 创建dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)

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

    # 定义损失函数
    # 对于BCELoss，如果target==1，则其表达式为log(x)。如果target==0，其表达式为log(1-x)。
    # loss_G = -E[log(D(G(z))] loss_G越小就代表D(G(z))越接近1，也就是产生的图片糊弄住了Discriminator，让它觉得是真的
    # loss_D = E[log(D(x))]+E(log(1-D(G(z))) loss_D越小代表log(D(x))和log(1-D(G(z)))越小，也就是D(x)越接近1，D(G(z))越接近0；即
    # Discriminator很容易给真实的图片x打高分，给产生的图片G(z)打低分
    # 故loss_G = criterion(D(G(z)), 1)
    # loss_D = criterion(D(x), 1) + criterion(D(G(z)), 0)
    criterion = torch.nn.BCELoss()

    lr = 0.0001
    betas = (0.5, 0.999)
    # 定义优化器
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    # 开启cudnn.benchmark
    cudnn.benchmark = True

    # 设置epoch_num
    epoch_num = 10

    wandb.init(
        # set the wandb project where this run will be logged
        project="Basic GAN",

        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "optimizer_G": optimizer_G.__class__,
            "betas_G": betas,
            "optimizer_D": optimizer_D.__class__,
            "betas_D": betas,
            "architecture": "DCGAN",
            "dataset": "another anime face dataset",
            "epochs": 10,
        }
    )

    # 开始训练
    for epoch in range(epoch_num):
        with tqdm(total=samples_num, ncols=100) as p_bar:
            p_bar.set_description(f'Epoch: {epoch + 1}')
            for data in train_dataloader:
                real_imgs, _ = data
                real_imgs = real_imgs.to(device)

                # generate noise
                noise_samples = torch.randn((len(real_imgs), nz, 1, 1)).to(device)

                # generate fake images from these noise
                fake_imgs = G(noise_samples).to(device)

                # ---------train Discriminator-----------------#
                # clear the gradients of Discriminator
                D.zero_grad()

                # preprocess labels and images
                real_labels = torch.ones((len(real_imgs), 1)).to(device)
                fake_labels = torch.zeros((len(fake_imgs), 1)).to(device)

                # compute loss_D_fake
                D_fake = D(fake_imgs.detach()).view(fake_labels.size())
                loss_D_fake = criterion(D_fake, fake_labels)
                loss_D_fake.backward()

                # compute loss_D_real
                D_real = D(real_imgs).view(real_labels.size())
                loss_D_real = criterion(D_real, real_labels)
                loss_D_real.backward()

                loss_D = loss_D_fake + loss_D_real
                # backpropagation
                optimizer_D.step()

                # ------------train Generator-----------------#
                # clear gradients
                G.zero_grad()

                # compute loss_G
                Dgz_new = D(fake_imgs).view(-1)
                loss_G = criterion(Dgz_new, torch.ones_like(Dgz_new))

                # backpropagation
                loss_G.backward()
                optimizer_G.step()

                # log loss to wandb
                wandb.log({'loss_D': loss_D.item(), 'loss_G': loss_G.item(), 'D(G(z))': Dgz_new.mean()})

                p_bar.update(len(real_imgs))

        # 训练完成后生成结果
        with torch.no_grad():
            noise_samples = torch.randn((4, nz, 1, 1)).to(device)
            fake_imgs = G(noise_samples).cpu()
            # 转成PIL图片
            fake_imgs_PIL = [transforms.ToPILImage()(fake_img) for fake_img in fake_imgs]

        # 显示图片
        import matplotlib.pyplot as plt

        plt.figure()

        n_rows = 2
        n_cols = 2
        for i, fake_img_PIL in enumerate(fake_imgs_PIL):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(fake_img_PIL)
        # 显示图片
        plt.show()
        plt.close()
