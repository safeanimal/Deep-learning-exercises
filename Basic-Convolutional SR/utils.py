import torch


def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def psnr(y_true, y_pred, max_pixel_value=1.0):
    mse_value = mse(y_true, y_pred)
    if mse_value == 0:
        return float("inf")
    return 20 * torch.log10(max_pixel_value / torch.sqrt(mse_value))


class AverageMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.best = 0

    def update(self, val, n=1):
        self.cnt += n
        self.sum += val
        self.avg = self.sum / self.cnt
