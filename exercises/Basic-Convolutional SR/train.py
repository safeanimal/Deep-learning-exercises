import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from model import SRG
from dataset import ImgDataset
import copy
from utils import psnr, AverageMeter


if __name__ == '__main__':
    outputs_dir = "C:\\Users\\41181\\OneDrive\\Documents\\Deep learning\\Super Resolution\\Models\\Test1\\outputs"
    img_folder = 'C:\\Users\\41181\\OneDrive\\Documents\\Deep learning\\datasets\\train\\animefaces256cleaner'
    scale_factor = 4
    num_samples = 1000
    dataset = ImgDataset(img_folder, scale_factor, num_samples)

    # 80% images for training, 20% for validation
    split_ratio = 0.8
    # calculate the lengths
    dataset_length = len(dataset)
    train_dataset_length = int(dataset_length * split_ratio)
    val_dataset_length = dataset_length - train_dataset_length
    # split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_dataset_length, val_dataset_length])
    # load the data
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # select device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # initiate the sr_models, loss_function, and optimizer
    model = SRG().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # set the benchmark
    cudnn.benchmark = True
    # deep copy the sr_models's parameters
    best_weight = copy.deepcopy(model.state_dict())

    num_epoch = 50

    best_epoch = 0
    best_psnr = 0

    for epoch in range(num_epoch):
        # training

        # start the train mode
        model.train()
        # used to track and calculate the average loss
        loss_meter = AverageMeter()
        # total is the progress bar's total number
        with tqdm(total=len(train_dataset)) as t:
            t.set_description(f'epoch:{epoch}/{num_epoch-1}')
            # process each batch
            for data in train_loader:
                # one batch contains 16 (input img and target img) pairs
                inputs, targets = data
                # move the tensor to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                # predict
                predicts = model(inputs)
                # compute the loss
                loss = criterion(predicts, targets)
                # backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # log the loss
                loss_meter.update(loss.item(), len(inputs))

                t.set_postfix(loss='{:.6f}'.format(loss_meter.avg))
                # update the progress bar
                t.update(len(inputs))

        # Save the sr_models's weights for the current epoch.
        torch.save(model.state_dict(), os.path.join(outputs_dir, 'epoch_{}.pth'.format(epoch)))

        # eval

        # start the eval mode
        model.eval()
        # used to track and calculate the average psnr
        psnr_meter = AverageMeter()

        # process each batch
        for data in val_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                predicts = model(inputs)
            # one batch contains 16 samples
            psnr_meter.update(psnr(targets, predicts), len(inputs))

        print(f'psnr:{psnr_meter.avg}')
        # if current epoch has the best result, keep it
        if psnr_meter.avg > best_psnr:
            best_epoch = epoch
            best_psnr = psnr_meter.avg
            best_weight = copy.deepcopy(model.state_dict())

        torch.save(best_weight, os.path.join(outputs_dir, 'best.pth'))
        print(f'best epoch:{best_epoch}, best psnr:{best_psnr}')
