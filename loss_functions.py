from dice_loss import IoULoss
from helper import collate_fn, draw_box
from data_helper import UnlabeledDataset, LabeledDataset
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import init
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import time
import sys
import random
import psutil

import json
import copy
import numpy as np
import pandas as pd
from collections import Counter

from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200


random.seed(0)
np.random.seed(0)
torch.manual_seed(0);


def loss_function_iou(x_hat, x, gamma=0.1, device=None):

    iou_loss = IoULoss()
    one_count = x.sum().cpu().item()
    zero_count = x.nelement() - one_count
    weight = torch.tensor([1/np.sqrt(zero_count), 1/np.sqrt(one_count)])
    weight_ = weight[x.data.view(-1).long()].view_as(x).to(device)
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x, reduction='none'
    )
    BCE = (BCE*weight_).mean()

    IOU = iou_loss(x_hat, x)

    return gamma*BCE + (1-gamma)*IOU


def unet_weight_map(y, wc=None, w0=10, sigma=5, device=None):
    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """

    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w


def unet_weight_map(y, wc=None, w0=100, sigma=3, device=None):
    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """
    y = y.cpu()
    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w


def dice_loss(pred, target, smooth=1., device=None):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) /
            (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def dice_loss_weighted(pred, target, smooth=1., device=None):

    weight = torch.tensor([1/np.sqrt(5110000), 1/np.sqrt(10000)])
    weight_ = weight[target.data.view(-1).long()].view_as(target).to(device)

    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (weight_ * pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) /
            (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

# Reconstruction + KL divergence losses summed over all elements and batch


def loss_function_bce_dice(x_hat, x, epoch=None, device=None):
    # only weighted bCE for first 8 epochs
    if epoch != None and epoch < 15:
        # weighted
        weight = torch.tensor([1, 1000])
        weight_ = weight[x.data.view(-1).long()].view_as(x).to(device)
        BCE = nn.functional.binary_cross_entropy(
            x_hat, x, reduction='none'
        )
        BCE = (BCE*weight_).mean()
    else:
        BCE = nn.functional.binary_cross_entropy(
            x_hat, x, reduction='mean'
        )

    DICE = dice_loss(x_hat, x)

    return BCE + DICE


def loss_function_unet(x_hat, x, epoch=None, device=None):
    # only weighted bCE for first 8 epochs
    batch_size = x.size(0)
    weight_ = torch.zeros_like(x)
    for i in range(batch_size):
        weight_[i, 0, :, :] = torch.from_numpy(
            unet_weight_map(x[i, 0, :, :])).to(device)
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x, reduction='none'
    )
    BCE = (BCE*weight_).mean()

    return BCE


def loss_function_weighted(x_hat, x, gamma=0.75, epoch=None, device=None):
    # only weighted bCE for first 8 epochs

    # weighted
    weight = torch.tensor([1/np.sqrt(5110000), 1/np.sqrt(10000)])
    # weight = torch.tensor([1/(5110000), 1/(10000)])
    weight_ = weight[x.data.view(-1).long()].view_as(x).to(device)
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x, reduction='none'
    )
    BCE = (BCE*weight_).mean()

    DICE = dice_loss_weighted(x_hat, x)

    return (1 - gamma) * BCE + gamma * DICE


 def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) /
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.sum()

# Reconstruction + KL divergence losses summed over all elements and batch


def loss_function_CNNVAE(x_hat, x, mu, logvar, epoch=None):
    # only weighted bCE for first 8 epochs
    if epoch != None and epoch < 8:
        # weighted
        weight = torch.tensor([1, 1000])
        weight_ = weight[x.view(-1, 800*800).data.view(-1).long()
                         ].view_as(x.view(-1, 800*800)).to(device)
        BCE = nn.functional.binary_cross_entropy(
            x_hat.view(-1, 800*800), x.view(-1, 800*800), reduction='none'
        )
        BCE = (BCE*weight_).sum()
    else:
        BCE = nn.functional.binary_cross_entropy(
            x_hat.view(-1, 800*800), x.view(-1, 800*800), reduction='sum'
        )

    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

    DICE = dice_loss(x_hat, x)

    return BCE + KLD + DICE
