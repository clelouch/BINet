import torch
import math
import torch.nn.functional as F
from torch import nn


def gaussian(window_size, sigma):
    import math
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def get_weight(data, window_size=31, sigma=4, channel=1, weight_type='average'):
    if weight_type == 'average':
        weight = torch.abs(F.avg_pool2d(data, kernel_size=window_size, stride=1, padding=window_size // 2) - data)
    elif weight_type == 'normal':
        x = torch.Tensor([math.exp(-(i - window_size // 2) ** 2 / float(2 * sigma ** 2)) for i in range(window_size)])
        x = x.unsqueeze(1) / x.sum()
        normal_kernel = x.mm(x.t()).unsqueeze(0).unsqueeze(0)
        normal_kernel = normal_kernel.expand(channel, 1, window_size, window_size)
        normal_kernel = normal_kernel.cuda()
        weight = torch.abs(F.conv2d(data, normal_kernel, stride=1, padding=window_size // 2) - data)
    return weight


def iou_loss(pred, mask, weight):
    inter = ((pred * mask) * weight).sum(dim=(2, 3))
    union = ((pred + mask) * weight).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou


def _iou(pred, target, size_average=True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


class IOU(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)


def bce_iou_loss(pred, target):
    bce_loss = nn.BCELoss(size_average=True)
    iou_loss_f = IOU(size_average=True)
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss_f(pred, target)

    loss = bce_out + iou_out

    return loss
