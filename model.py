import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
import numpy as np
from attention import ChannelAttention as CA
from deeplab_resnet import resnet50
from vgg import vgg16

config_vgg = {'convert': [[128, 256, 512, 512, 512], [64, 128, 256, 512, 512]],
              'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False],
                            [True, True, True, False]], 'score': 128}  # no convert layer, no conv6

config_resnet = {'convert': [[64, 256, 512, 1024, 2048], [64, 64, 64, 64, 64]],
                 'score': [64, 64, 64, 64, 64]}


def upsample_like(x, target):
    return F.interpolate(x, target.shape[2:], mode='bilinear', align_corners=True)


class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up = []
        for i in range(len(list_k[0])):
            up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False),
                                    nn.BatchNorm2d(list_k[1][i]),
                                    nn.ReLU(inplace=True)))
        self.convert0 = nn.ModuleList(up)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl


class BasicConv(nn.Module):
    def __init__(self, channel, stride, padding=1, dilate=1):
        super(BasicConv, self).__init__()
        self.channel = channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, 3, stride=stride, padding=padding, dilation=dilate, bias=False),
            nn.BatchNorm2d(self.channel), )
        # nn.ReLU()

    def forward(self, x):
        return self.conv(x)


################# FIM1 #########################
class USRM3(nn.Module):
    def __init__(self, channel):
        super(USRM3, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv(self.channel, 2, 1, 1)
        self.conv2 = BasicConv(self.channel, 2, 1, 1)
        self.conv3 = BasicConv(self.channel, 1, 2, 2)

        self.conv_rev1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev2 = BasicConv(self.channel, 1, 1, 1)

        self.conv_sum = BasicConv(self.channel, 1, 1, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)

        # debug
        # print(x.shape, y1.shape, y2.shape, y3.shape)

        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev1(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev2(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)

        # debug
        # print(y.shape, y1.shape, y2.shape)
        return self.conv_sum(F.relu(x + y))


################ FIM2 ##########################
class USRM4(nn.Module):
    def __init__(self, channel):
        super(USRM4, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv(self.channel, 1, 2, 2)
        self.conv2 = BasicConv(self.channel, 2, 1, 1)
        self.conv3 = BasicConv(self.channel, 2, 1, 1)
        self.conv4 = BasicConv(self.channel, 1, 2, 2)

        self.conv_rev1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev2 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev3 = BasicConv(self.channel, 1, 1, 1)

        self.conv_sum1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_sum2 = BasicConv(self.channel, 1, 1, 1)

    def forward(self, x, h1):
        # gi means global information
        y1 = self.conv1(x)
        y1 = y1 + F.interpolate(h1, y1.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)

        # debug
        # print(x.shape, y1.shape, y2.shape, y3.shape, y4.shape)

        y3up = F.interpolate(y4, y3.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv_rev1(y3 + y3up)
        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev2(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        # debug
        # print("In USRM4, h1.shape: ", h1.shape, ", y1up.shape: ", y1up.shape)
        y1 = self.conv_rev3(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum1(F.relu(x + y)), self.conv_sum2(F.relu(h1 + upsample_like(y1up, h1)))


################ FIM3 ##########################
class USRM5(nn.Module):
    def __init__(self, channel):
        super(USRM5, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv(self.channel, 2, 1, 1)
        self.conv2 = BasicConv(self.channel, 1, 2, 2)
        self.conv3 = BasicConv(self.channel, 2, 1, 1)
        self.conv4 = BasicConv(self.channel, 2, 1, 1)
        self.conv5 = BasicConv(self.channel, 1, 2, 2)

        self.conv_rev1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev2 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev3 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev4 = BasicConv(self.channel, 1, 1, 1)

        self.conv_sum1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_sum2 = BasicConv(self.channel, 1, 1, 1)
        self.conv_sum3 = BasicConv(self.channel, 1, 1, 1)

    def forward(self, x, h1, h2):
        # gi means global information
        y1 = self.conv1(x)
        y1 = y1 + F.interpolate(h1, y1.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv2(y1)
        y2 = y2 + F.interpolate(h2, y2.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)

        # debug
        # print(x.shape, y1.shape, y2.shape, y3.shape, y4.shape)

        y4up = F.interpolate(y5, y4.shape[2:], mode='bilinear', align_corners=True)
        y4 = self.conv_rev1(y4 + y4up)
        y3up = F.interpolate(y4, y3.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv_rev2(y3 + y3up)
        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev3(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev4(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum1(F.relu(x + y)), self.conv_sum2(F.relu(h1 + upsample_like(y1up, h1))), \
               self.conv_sum3(F.relu(h2 + upsample_like(y2up, h2)))


################ FIM4 ##########################
class USRM5_2(nn.Module):
    def __init__(self, channel):
        super(USRM5_2, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv(self.channel, 2, 1, 1)
        self.conv2 = BasicConv(self.channel, 2, 1, 1)
        self.conv3 = BasicConv(self.channel, 1, 2, 2)
        self.conv4 = BasicConv(self.channel, 2, 1, 1)
        self.conv5 = BasicConv(self.channel, 1, 2, 2)

        self.conv_rev1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev2 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev3 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev4 = BasicConv(self.channel, 1, 1, 1)

        self.conv_sum1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_sum2 = BasicConv(self.channel, 1, 1, 1)
        self.conv_sum3 = BasicConv(self.channel, 1, 1, 1)
        self.conv_sum4 = BasicConv(self.channel, 1, 1, 1)

    def forward(self, x, h1, h2, h3):
        # gi means global information
        y1 = self.conv1(x)
        y1 = y1 + F.interpolate(h1, y1.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv2(y1)
        y2 = y2 + F.interpolate(h2, y2.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv3(y2)
        y3 = y3 + F.interpolate(h3, y3.shape[2:], mode='bilinear', align_corners=True)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)

        # debug
        # print(x.shape, y1.shape, y2.shape, y3.shape, y4.shape)

        y4up = F.interpolate(y5, y4.shape[2:], mode='bilinear', align_corners=True)
        y4 = self.conv_rev1(y4 + y4up)
        y3up = F.interpolate(y4, y3.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv_rev2(y3 + y3up)
        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev3(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev4(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum1(F.relu(x + y)), self.conv_sum2(F.relu(h1 + upsample_like(y1up, h1))), self.conv_sum3(
            F.relu(h2 + upsample_like(y2up, h2))), self.conv_sum4(F.relu(h3 + upsample_like(y3up, h3)))


################ FIM5 ##########################
class USRM5_3(nn.Module):
    def __init__(self, channel):
        super(USRM5_3, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv(self.channel, 2, 1, 1)
        self.conv2 = BasicConv(self.channel, 2, 1, 1)
        self.conv3 = BasicConv(self.channel, 2, 1, 1)
        self.conv4 = BasicConv(self.channel, 1, 2, 2)
        self.conv5 = BasicConv(self.channel, 1, 2, 2)

        self.conv_rev1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev2 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev3 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev4 = BasicConv(self.channel, 1, 1, 1)

        self.conv_sum1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_sum2 = BasicConv(self.channel, 1, 1, 1)
        self.conv_sum3 = BasicConv(self.channel, 1, 1, 1)
        self.conv_sum4 = BasicConv(self.channel, 1, 1, 1)
        self.conv_sum5 = BasicConv(self.channel, 1, 1, 1)

    def forward(self, x, h1, h2, h3, h4):
        # gi means global information
        y1 = self.conv1(x)
        y1 = y1 + F.interpolate(h1, y1.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv2(y1)
        y2 = y2 + F.interpolate(h2, y2.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv3(y2)
        y3 = y3 + F.interpolate(h3, y3.shape[2:], mode='bilinear', align_corners=True)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)

        # debug
        # print(x.shape, y1.shape, y2.shape, y3.shape, y4.shape)

        y4up = F.interpolate(y5, y4.shape[2:], mode='bilinear', align_corners=True)
        y4 = self.conv_rev1(y4 + y4up)
        y3up = F.interpolate(y4, y3.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv_rev2(y3 + y3up)
        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev3(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev4(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum1(F.relu(x + y)), self.conv_sum2(F.relu(h1 + upsample_like(y1up, h1))), self.conv_sum3(
            F.relu(h2 + upsample_like(y2up, h2))), self.conv_sum4(F.relu(h3 + upsample_like(y3up, h3))), self.conv_sum5(
            F.relu(h4 + upsample_like(y4up, h4)))


################ BID ##########################
class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.channel = channel
        self.usrm3 = USRM3(self.channel)
        self.usrm4 = USRM4(self.channel)
        self.usrm5_1 = USRM5(self.channel)
        self.usrm5_2 = USRM5_2(self.channel)
        self.usrm5_3 = USRM5_3(self.channel)

        self.attention1 = CA(self.channel * 5, self.channel)
        self.attention2 = CA(self.channel * 4, self.channel)
        self.attention3 = CA(self.channel * 3, self.channel)
        self.attention4 = CA(self.channel * 2, self.channel)
        # self.attention5 = CA(self.channel * 2, self.channel)

    def forward(self, C1, C2, C3, C4, C5):
        C5 = self.usrm3(C5)
        C4_1, C5_1 = self.usrm4(C4, C5)
        C3_2, C4_2, C5_2 = self.usrm5_1(C3, C4_1, C5_1)
        C2_3, C3_3, C4_3, C5_3 = self.usrm5_2(C2, C3_2, C4_2, C5_2)
        C1_4, C2_4, C3_4, C4_4, C5_4 = self.usrm5_3(C1, C2_3, C3_3, C4_3, C5_3)

        C5 = self.attention1(torch.cat([C5, C5_1, C5_2, C5_3, C5_4], dim=1))
        C4 = self.attention2(torch.cat([C4_1, C4_2, C4_3, C4_4], dim=1))
        C3 = self.attention3(torch.cat([C3_2, C3_3, C3_4], dim=1))
        C2 = self.attention4(torch.cat([C2_3, C2_4], dim=1))
        # C1 = self.attention5(torch.cat([C1, C1_4], dim=1))

        return C1_4, C2, C3, C4, C5


class ScoreLayer(nn.Module):
    def __init__(self, list_k):
        super(ScoreLayer, self).__init__()
        score = []
        for k in list_k:
            score.append(nn.Conv2d(k, 1, 1, 1))
        self.score = nn.ModuleList(score)

    def forward(self, x, x_size=None):
        for i in range(len(x)):
            x[i] = self.score[i](x[i])
            x[i] = F.interpolate(x[i], x_size[2:], mode='bilinear', align_corners=True)
        return x


def extra_layer(base_model_cfg, vgg):
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet
    convert_layers, score_layers = [], []
    convert_layers = ConvertLayer(config['convert'])
    score_layers = ScoreLayer(config['score'])
    return vgg, convert_layers, score_layers


class BINet(nn.Module):
    def __init__(self, base_model_cfg, base, convert_layers, score_layers):
        super(BINet, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base
        self.score = score_layers
        if base_model_cfg == 'vgg':
            self.config = config_vgg
        elif base_model_cfg == 'resnet':
            self.config = config_resnet
        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers
        self.decoder1 = Decoder(64)
        self.decoder2 = Decoder(64)
        self.score_pred = nn.Conv2d(64, 1, 1, 1)

    def forward(self, x):
        x_size = x.size()
        C1, C2, C3, C4, C5 = self.base(x)
        if self.base_model_cfg == 'resnet':
            C1, C2, C3, C4, C5 = self.convert([C1, C2, C3, C4, C5])

        C1, C2, C3, C4, C5 = self.decoder1(C1, C2, C3, C4, C5)
        pred1 = self.score_pred(F.interpolate(C1, x_size[2:], mode='bilinear', align_corners=True))
        C1, C2, C3, C4, C5 = self.decoder2(C1, C2, C3, C4, C5)

        pred2_1, pred2_2, pred2_3, pred2_4, pred2_5 = self.score([C1, C2, C3, C4, C5], x_size)
        return torch.sigmoid(pred1), torch.sigmoid(pred2_1), torch.sigmoid(pred2_2), torch.sigmoid(
            pred2_3), torch.sigmoid(pred2_4), torch.sigmoid(pred2_5)


def build_model(base_model_cfg='resnet'):
    if base_model_cfg == 'vgg':
        return BINet(base_model_cfg, *extra_layer(base_model_cfg, vgg16()))
    elif base_model_cfg == 'resnet':
        return BINet(base_model_cfg, *extra_layer(base_model_cfg, resnet50()))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))


if __name__ == '__main__':
    import time

    model = build_model(base_model_cfg='resnet')
    # print_network(model, 'BINet')
    img = torch.randn(1, 3, 352, 352)
    model = model.cuda()
    img = img.cuda()
    with torch.no_grad():
        start = time.time()
        res = model(img)
        # for i in range(10):
        #     res = model(img)
        torch.cuda.synchronize()
        end = time.time()
        # for k in res:
        #     print(k.shape)
        print(end - start)
