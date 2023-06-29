import torch


def calculate_psnr(inputs, outputs, max_val=1.0):
    """
    calculate the mean PSNR between inputs and targets

    :param outputs: (batch_size, channels, height, width)
    :param max_val: the max pixel value in these images
    :param inputs: (batch_size, channels, height, width)
    :return: mean PSNR
    """
    # torch.mean compute the mean of all the elements
    mse = torch.mean((inputs - outputs) ** 2)
    return 10 * torch.log10((max_val ** 2) / mse)


class AverageLogger(object):
    def __init__(self):
        super().__init__()
        self.avg = 0
        self.cnt = 0
        self.sum = 0

    def update(self, x, n=1):
        self.sum += x
        self.cnt += n
        self.avg = self.sum / self.cnt
