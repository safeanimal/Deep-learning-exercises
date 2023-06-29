import torch
from torch import Tensor


def mean_square_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    计算真实值和预测值之间的均方误差
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: 均方误差
    """
    return torch.mean((y_true - y_pred) ** 2)


def peak_signal_to_noise_ratio(y_true: Tensor, y_pred: Tensor, max_pixel_value=1.0) -> Tensor:
    """
    计算峰值信噪比
    :param y_true: 真实值（一批图片组成的Tensor, B*W*H*C）
    :param y_pred: 预测值（一批图片组成的Tensor, B*W*H*C）
    :param max_pixel_value: 最大像素值
    :return: 峰值信噪比的值
    """
    mse_value = mean_square_error(y_true, y_pred)
    if mse_value == 0:
        return float("inf")
    return 20 * torch.log10(max_pixel_value / torch.sqrt(mse_value))


def print_structural_similarity():
    """
    https://github.com/VainF/pytorch-msssim
    1. Basic Usage
    from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
    # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
    # Y: (N,3,H,W)

    # calculate ssim & ms-ssim for each image
    ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
    ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)

    # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
    ssim_loss = 1 - ssim( X, Y, data_range=255, size_average=True) # return a scalar
    ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )

    # reuse the gaussian kernel with SSIM & MS_SSIM.
    ssim_module = SSIM(data_range=255, size_average=True, channel=3) # channel=1 for grayscale images
    ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

    ssim_loss = 1 - ssim_module(X, Y)
    ms_ssim_loss = 1 - ms_ssim_module(X, Y)

    2.Normalized input If you need to calculate MS-SSIM/SSIM on normalized images, please denormalize them to the
    range of [0, 1] or [0, 255] first.
    # X: (N,3,H,W) a batch of normalized images (-1 ~ 1)
    # Y: (N,3,H,W)
    X = (X + 1) / 2  # [-1, 1] => [0, 1]
    Y = (Y + 1) / 2
    ms_ssim_val = ms_ssim( X, Y, data_range=1, size_average=False ) #(N,)

    3. Enable non-negative_ssim For ssim, it is recommended to set non-negative_ssim=True to avoid negative results.
    However, this option is set to False by default to keep it consistent with tensorflow and skimage.

    For ms-ssim, there is no non-negative_ssim option and the ssim responses is forced to be non-negative to avoid
    NaN results.

    :return:
    """
    print("看示例用法")

