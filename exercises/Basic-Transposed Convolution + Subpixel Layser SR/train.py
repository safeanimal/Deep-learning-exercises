import os.path

import torch
from torch.utils.data import DataLoader, random_split
from model import SRG2
from dataset import ImgDataset
from tqdm import tqdm
from utils import AverageLogger, calculate_psnr
from torch.backends import cudnn
import copy
import wandb

if __name__ == '__main__':
    # specify outputs dir
    outputs_dir = 'outputs'
    # calculate train and val dataset's samples num
    split_ratio = 0.8
    samples_size = 1000
    num_train_samples = int(samples_size*0.8)
    num_val_samples = samples_size - num_train_samples

    # get the dataset
    img_folder = 'D:/Materials/dataset/anime-face-256by256/animefaces256cleaner'
    full_dataset = ImgDataset(256, 4, img_folder, samples_size)

    # split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [num_train_samples, num_val_samples])

    # load the dataset
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

    # select the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create model and move it to specified device
    model = SRG2().to(device)

    # enable the cudnn.benchmark to accelerate the training
    cudnn.benchmark = True

    # select criterion and optimization method
    criterion = torch.nn.MSELoss()
    optimization = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epoch = 10
    best_epoch = 0
    best_psnr = 0
    best_weights = copy.deepcopy(model.state_dict())

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Transposed Convolution SR Basic",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "architecture": str(model),
            "dataset": "another anime face dataset",
            "epochs": num_epoch,
            "criterion": criterion.__class__.__name__,
            "optimization": optimization.__class__.__name__,
        }
    )

    for epoch in range(num_epoch):
        # set the train mode
        model.train()
        # use logger to log loss
        loss_loger = AverageLogger()

        with tqdm(total=len(train_dataset), ncols=100) as pbar:
            pbar.set_description(f'Epoch{epoch + 1}')
            for data_batch in train_dataloader:
                inputs, targets = data_batch
                # move tensors to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                # predict the outputs
                outputs = model(inputs)
                # calculate loss
                loss = criterion(outputs, targets)
                # backpropagation and optimization
                optimization.zero_grad()
                loss.backward()
                optimization.step()
                # log the loss
                loss_loger.update(loss.item(), len(inputs))
                # update the pbar
                pbar.update(len(inputs))
                pbar.set_postfix({'loss': loss_loger.avg})

        wandb.log({'loss': loss_loger.avg})

        # set the model to eval mode
        model.eval()
        # use logger to log PSNR
        psnr_logger = AverageLogger()
        with tqdm(total=len(val_dataset)) as pbar:
            pbar.set_description(f'Val{epoch+1}')
            for data_batch in val_dataloader:
                inputs, targets = data_batch
                # move tensors to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                # turn off the auto-grad in eval process
                with torch.no_grad():
                    # predict the outputs
                    outputs = model(inputs)
                # calculate PSNR
                PSNR = calculate_psnr(targets, outputs)
                # log the PSNR
                psnr_logger.update(PSNR)
                # update the progress bar
                pbar.update(len(inputs))
                pbar.set_postfix({'psnr': psnr_logger.avg})

        wandb.log({'psnr': psnr_logger.avg})
        # if current epoch has the best psnr, update it
        if psnr_logger.avg > best_psnr:
            best_psnr = psnr_logger.avg
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(outputs_dir, 'best.pth'))

    print(f'Complete! \n The best epoch is {best_epoch} \n The best psnr is {best_psnr}')
    wandb.finish()