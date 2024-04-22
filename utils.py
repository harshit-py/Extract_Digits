import torch.nn as nn
import torch.nn.functional as F
import torch
# from model import CustomVgg16, SimpleResNet


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]

    if pool:
        layers.append(nn.MaxPool2d(4))

    return nn.Sequential(*layers)


def fscore(output, label, threshold=0.5, beta=1):
    prob = F.sigmoid(output) > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-12)

    return F2.mean(0)


def load_checkpoint(path, model='resnet'):
    if model == 'resnet':
        return SimpleResNet.load_from_checkpoint(path)
    else:
        return CustomVgg16.load_from_checkpoint(path)


def eval_single(model, sample):
    with torch.no_grad():
        output = model(sample.unsqueeze(0))


