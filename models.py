from sklearn.metrics import confusion_matrix
from helper import collate_fn, draw_box
from data_helper import UnlabeledDataset, LabeledDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias=True,
                 pool=False,
                 mp_kernel_size=2,
                 mp_stride=2):
        super().__init__()
        if pool:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.LeakyReLU(negative_slope=0.1),  # nn.ReLU(),
                nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride))
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.LeakyReLU(negative_slope=0.1),  # nn.ReLU(),
            )

    def forward(self, x):
        return self.layer(x)


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.1)  # nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class ConvTLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 unpool=False,
                 mp_kernel_size=2,
                 mp_stride=2):
        super().__init__()
        if unpool:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels,
                                   out_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   output_padding=output_padding,
                                   bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.LeakyReLU(negative_slope=0.1),  # nn.ReLU()
                nn.MaxUnpool2d(kernel_size=mp_kernel_size, stride=mp_stride)
            )
        else:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels,
                                   out_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   output_padding=output_padding,
                                   bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.LeakyReLU(negative_slope=0.1),  # nn.ReLU()
            )

    def forward(self, x):
        return self.layer(x)


class EncoderGAN(nn.Module):
    def __init__(self, d=650):
        super().__init__()
        self.conv1 = ConvLayer(3, 96, stride=2)
        self.conv2 = ConvLayer(96, 128, stride=2)
        self.conv3 = ConvLayer(128, 256, stride=2)
        self.conv4 = ConvLayer(256, 512, stride=2)
        self.conv5 = ConvLayer(512, 1024, stride=2)
        self.conv6 = ConvLayer(1024, 2048, stride=2)
        self.lin1 = nn.Linear(2048*3*3, d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # print(x.shape)
        x = self.lin1(x.reshape(-1, 2048*3*3))
        return x


class Discriminator1(nn.Module):
    def __init__(self, d=650):
        super().__init__()
        self.d = d
        self.encoder = EncoderGAN(self.d)
        self.linear = nn.Linear(self.d, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x.reshape(-1, self.d))
        return torch.sigmoid(x)


def random_vector(batch_size, length, device):
    # Sample from a Gaussian distribution
    z_vec = torch.randn(batch_size, length, 1, 1).float()
    if torch.cuda.is_available():
        z_vec = z_vec.to(device)
    return z_vec


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convt1 = ConvTLayer(4096, 2048, stride=2)
        self.convt2 = ConvTLayer(2048, 1024, stride=2, output_padding=(0, 0))
        self.convt3 = ConvTLayer(
            1024, 512, stride=2, padding=(1, 1), output_padding=(1, 1))
        self.convt4 = ConvTLayer(512, 256, stride=2, output_padding=(1, 1))
        self.convt5 = ConvTLayer(256, 128, stride=2, output_padding=(1, 1))
        self.convt6 = ConvTLayer(128, 96, stride=2, output_padding=(1, 1))
        self.convt7 = ConvTLayer(96, 64, stride=2, output_padding=(1, 1))
        self.convt8 = ConvTLayer(64, 32, stride=1, output_padding=(0, 0))
        self.convt9 = ConvTLayer(
            32, 18, stride=1, padding=(1, 1), output_padding=(0, 0))

    def forward(self, z):
        z = self.convt1(z)
        z = self.convt2(z)
        z = self.convt3(z)
        z = self.convt4(z)
        z = self.convt5(z)
        z = self.convt6(z)
        z = self.convt7(z)
        z = self.convt8(z)
        z = self.convt9(z)
        return z


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.convt1 = ConvTLayer(4096, 2048, stride=2)
        self.convt2 = ConvTLayer(2048, 1024, stride=2, output_padding=(1, 1))
        self.convt3 = ConvTLayer(
            1024, 512, stride=2, padding=(1, 1), output_padding=(0, 0))
        self.convt4 = ConvTLayer(512, 256, stride=2, output_padding=(0, 0))
        self.convt5 = ConvTLayer(256, 128, stride=2, output_padding=(0, 0))
        self.convt6 = ConvTLayer(128, 96, stride=2, output_padding=(0, 0))
        self.convt7 = ConvTLayer(96, 3, stride=2, output_padding=(1, 1))

    def forward(self, z):
        z = self.convt1(z)
        z = self.convt2(z)
        z = self.convt3(z)
        z = self.convt4(z)
        z = self.convt5(z)
        z = self.convt6(z)
        z = self.convt7(z)
        return z


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.decoder(x)
        return torch.tanh(x).reshape(6, -1, 3, 256, 256)


class Generator1(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder1()

    def forward(self, x):
        x = self.decoder(x)
        return torch.tanh(x)


class EncoderY(nn.Module):
    def __init__(self,  d):
        super().__init__()
        self.conv1 = ConvLayer(3, 96, stride=2)
        self.conv2 = ConvLayer(96, 128, stride=2)
        self.conv3 = ConvLayer(128, 256, stride=2)
        self.conv4 = ConvLayer(256, 512, stride=2)
        self.conv5 = ConvLayer(512, 1024, stride=2)
        self.conv6 = ConvLayer(1024, 2048, stride=2)
        self.lin1 = nn.Linear(2048*3*3, d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # print(x.shape)
        x = self.lin1(x.reshape(-1, 2048*3*3))
        return x


class EncoderX(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv1 = ConvLayer(1, 16, stride=2)
        self.conv2 = ConvLayer(16, 32, stride=2)
        self.conv3 = ConvLayer(32, 48, stride=2)
        self.conv4 = ConvLayer(48, 64, stride=2)
        self.conv5 = ConvLayer(64, 96, stride=2)
        self.conv6 = ConvLayer(96, 128, stride=2)
        self.conv7 = ConvLayer(128, 256, stride=2)
        self.conv8 = ConvLayer(256, 512, stride=2)
        self.lin1 = nn.Linear(512*2*2, d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        # print(x.shape)
        x = self.lin1(x.reshape(-1, 512*2*2))
        return x


class DecoderX(nn.Module):
    def __init__(self):
        super().__init__()
        self.convt1 = ConvTLayer(4096, 2048, kernel_size=3, stride=2)
        self.convt2 = ConvTLayer(
            2048, 1024, kernel_size=3, stride=3, output_padding=(0, 0))
        self.convt3 = ConvTLayer(
            1024, 512, kernel_size=3, stride=2, padding=(1, 1), output_padding=(0, 0))
        self.convt4 = ConvTLayer(
            512, 256, kernel_size=3, stride=3, padding=(1, 1), output_padding=(0, 0))
        self.convt5 = ConvTLayer(
            256, 128, kernel_size=3, stride=2, output_padding=(0, 0))
        self.convt6 = ConvTLayer(
            128, 96, kernel_size=3, stride=2, output_padding=(0, 0))
        self.convt7 = ConvTLayer(
            96, 64, kernel_size=3, stride=2, output_padding=(0, 0))
        self.convt8 = ConvTLayer(64, 1, kernel_size=3,
                                 stride=2, output_padding=(1, 1))

    def forward(self, z):
        z = self.convt1(z)
        z = self.convt2(z)
        z = self.convt3(z)
        z = self.convt4(z)
        z = self.convt5(z)
        z = self.convt6(z)
        z = self.convt7(z)
        z = self.convt8(z)
        return torch.sigmoid(z)


# Defining the model

class CNN_VAE(nn.Module):
    # hidden_d=196, image_d=650 or hidden_d=286, image_d=625
    def __init__(self, hidden_d=196, image_d=650, device='cpu'):
        super().__init__()

        self.d = hidden_d
        self.id = image_d
        self.device = device

        self.y_encoder = EncoderY(d=self.id)

        self.x_encoder = EncoderX(d=2*self.d)

        self.x_decoder = DecoderX()

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, y):
        mu_logvar = self.x_encoder(x).view(-1, 2, self.d)
        # print(mu_logvar.shape)
        img_enc = [self.y_encoder(img.squeeze()) for img in y]
        mu = mu_logvar[:, 0, :]
        # print(mu.shape)
        logvar = mu_logvar[:, 1, :]
        # print(logvar.shape)
        z = self.reparameterise(mu, logvar)
        img_enc.append(z)
        out = torch.cat(img_enc, axis=1).reshape(-1, 4096, 1, 1)
        return self.x_decoder(out), mu, logvar

    def inference(self, y, mu=None, logvar=None):
        N = y.size(1)
        z = torch.randn((N, self.d)).to(self.device)
        # print('Prior:',z.shape)
        if mu is not None and logvar is not None:
            # print(mu.shape)
            # print(logvar.shape)
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            z = eps.mul(std).add_(mu)
            # print('Post:',z.shape)
        z = z.reshape(-1, 196)
        img_enc = [self.y_encoder(img.squeeze()) for img in y]
        img_enc.append(z)
        out = torch.cat(img_enc, axis=1).reshape(-1, 4096, 1, 1)
        return self.x_decoder(out)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.BatchNorm2d(out_channels),
        # nn.Dropout(0.5),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.BatchNorm2d(out_channels),
        # nn.Dropout(0.5)
    )


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(3, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 48)
        self.dconv_down4 = double_conv(48, 64)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(48 + 64, 48)
        self.dconv_up2 = double_conv(32 + 48, 32)
        self.dconv_up1 = double_conv(32 + 16, 16)

        self.dconv_up0 = double_conv(6*16, 3*16)
        self.dconv_up00 = double_conv(3*16, 16)

        self.conv_last = nn.Conv2d(16, n_class, 1)

    def forward_once(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        return x

    def forward(self, x):
        x = [self.forward_once(y) for y in x]
        x = torch.cat(x, axis=1)

        x = self.upsample(x)
        x = self.dconv_up0(x)

        x = self.upsample(x)
        x = self.dconv_up00(x)

        out = self.conv_last(x)

        return torch.sigmoid(out)


def single_conv(in_channels, out_channels, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=padding),
        nn.LeakyReLU(negative_slope=0.1),
        nn.BatchNorm2d(out_channels),
        # nn.Dropout(0.5)
    )


def double_conv2(in_channels, out_channels, output_padding=0, double=False):
    if double:
        return nn.Sequential(nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            # nn.Dropout(0.5)
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(out_channels),
        )

    else:
        return nn.Sequential(nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            output_padding=output_padding),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(out_channels),
            # nn.Dropout(0.5)
        )


class EncoderCNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv1 = ConvLayer(3, 96, stride=2)
        self.conv2 = ConvLayer(96, 128, stride=2)
        self.conv3 = ConvLayer(128, 256, stride=2)
        self.conv4 = ConvLayer(256, 512, stride=2)
        self.conv5 = ConvLayer(512, 1024, stride=2)
        self.conv6 = ConvLayer(1024, 2048, stride=2)
        self.lin1 = nn.Linear(2048*3*3, d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.lin1(x.reshape(-1, 2048*3*3))
        return x


class Encoder(nn.Module):
    def __init__(self, d=650):
        super().__init__()
        self.encoder = EncoderCNN(d=d)

    def forward(self, x):
        x = self.encoder(x)
        return x


# Defining the model
encoder = Encoder()
pretrained_dict = torch.load(
    'models/rotation_learning_model.pth')
model_dict = encoder.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
encoder.load_state_dict(pretrained_dict)

for param in encoder.parameters():
    param.requires_grad = False

# Defining the model


class CNN_VAE_transfer(nn.Module):
    # hidden_d=196, image_d=650 or hidden_d=286, image_d=625
    def __init__(self, hidden_d=196, image_d=650, device='cpu'):
        super().__init__()

        self.d = hidden_d
        self.id = image_d
        self.device = device

        self.y_encoder = encoder

        self.x_encoder = EncoderX(d=2*self.d)

        self.x_decoder = DecoderX()

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, y):
        mu_logvar = self.x_encoder(x).view(-1, 2, self.d)
        # print(mu_logvar.shape)
        img_enc = [self.y_encoder(img.squeeze()) for img in y]
        mu = mu_logvar[:, 0, :]
        # print(mu.shape)
        logvar = mu_logvar[:, 1, :]
        # print(logvar.shape)
        z = self.reparameterise(mu, logvar)
        img_enc.append(z)
        out = torch.cat(img_enc, axis=1).reshape(-1, 4096, 1, 1)
        return self.x_decoder(out), mu, logvar

    def inference(self, y, mu=None, logvar=None):
        N = y.size(1)
        z = torch.randn((N, self.d)).to(self.device)
        # print('Prior:',z.shape)
        if mu is not None and logvar is not None:
            # print(mu.shape)
            # print(logvar.shape)
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            z = eps.mul(std).add_(mu)
            # print('Post:',z.shape)
        z = z.reshape(-1, 196)
        img_enc = [self.y_encoder(img.squeeze()) for img in y]
        img_enc.append(z)
        out = torch.cat(img_enc, axis=1).reshape(-1, 4096, 1, 1)
        return self.x_decoder(out)


class CNN_ROT_UNET(nn.Module):
    def __init__(self, d=650, output_size=4):
        super().__init__()
        self.dconv_down1 = double_conv(3, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 48)
        self.dconv_down4 = double_conv(48, 64)
        self.maxpool = nn.MaxPool2d(2)
        self.linear = nn.Linear(64*25*25, 4)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        # print(x.shape)
        x = self.linear(x.reshape(-1, 64*25*25))
        return F.log_softmax(x, dim=1)


class CNNVAE_ROT(nn.Module):
    def __init__(self, d=650, output_size=4):
        super(CNN, self).__init__()
        self.encoder = Encoder1(d=d)
        self.linear = nn.Linear(d, 4)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.BatchNorm2d(out_channels),
        # nn.Dropout(0.5),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.BatchNorm2d(out_channels),
        # nn.Dropout(0.5)
    )


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(3, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 48)
        self.dconv_down4 = double_conv(48, 64)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(48 + 64, 48)
        self.dconv_up2 = double_conv(32 + 48, 32)
        self.dconv_up1 = double_conv(32 + 16, 16)

        self.dconv_up0 = double_conv(6*16, 3*16)
        self.dconv_up00 = double_conv(3*16, 16)

        self.conv_last = nn.Conv2d(16, n_class, 1)

    def forward_once(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        return x

    def forward(self, x):
        x = [self.forward_once(y) for y in x]
        x = torch.cat(x, axis=1)

        x = self.upsample(x)
        x = self.dconv_up0(x)

        x = self.upsample(x)
        x = self.dconv_up00(x)

        out = self.conv_last(x)

        return torch.sigmoid(out)


def single_conv(in_channels, out_channels, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=padding),
        nn.LeakyReLU(negative_slope=0.1),
        nn.BatchNorm2d(out_channels),
        # nn.Dropout(0.5)
    )


def double_conv2(in_channels, out_channels, output_padding=0, non_lin=False):
    if non_lin:
        return nn.Sequential(nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            output_padding=output_padding),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(out_channels),
            # nn.Dropout(0.5)
        )
    else:
        return nn.Sequential(nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            output_padding=output_padding),
            # nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(out_channels),
            # nn.Dropout(0.5)
        )


class UNet2(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(3, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 48)
        self.dconv_down4 = double_conv(48, 64)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv2(64, 64, 0, non_lin=True)
        self.dconv_up3 = double_conv2(48 + 64, 48, non_lin=True)
        self.dconv_up2 = double_conv2(32 + 48, 32, non_lin=True)
        self.dconv_up1 = double_conv2(32 + 16, 16, non_lin=True)

        self.dconv_up0 = double_conv2(6*16, 3*16, non_lin=True)
        self.dconv_up00 = double_conv2(3*16, 16, non_lin=True)

        self.conv_last = nn.Conv2d(16, n_class, 1, stride=2)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def forward_once(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        # print(x.shape)
        #x = self.upsample(x)
        x = self.dconv_up4(x)
        # print(x.shape)
        # print(conv3.shape)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        return x

    def forward(self, x):
        x = [self.forward_once(y) for y in x]
        x = torch.cat(x, axis=1)

        x = self.dconv_up0(x)

        x = self.dconv_up00(x)

        out = self.conv_last(x)

        return torch.sigmoid(out)


class UNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.dconv_down1 = double_conv(3, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 48)
        self.dconv_down4 = double_conv(48, 64)

        self.maxpool = nn.MaxPool2d(2)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        return x


unetencoder = UNetEncoder()

pretrained_dict = torch.load(
    'models/rotation_learning_model_unet.pth')
model_dict = unetencoder.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
unetencoder.load_state_dict(pretrained_dict)

for param in unetencoder.parameters():
    param.requires_grad = False


class UNetTransfer(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = unetencoder.dconv_down1
        self.dconv_down2 = unetencoder.dconv_down2
        self.dconv_down3 = unetencoder.dconv_down3
        self.dconv_down4 = unetencoder.dconv_down4

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv2(64, 64, 0, non_lin=True)
        self.dconv_up3 = double_conv2(48 + 64, 48, non_lin=True)
        self.dconv_up2 = double_conv2(32 + 48, 32, non_lin=True)
        self.dconv_up1 = double_conv2(32 + 16, 16, non_lin=True)

        self.dconv_up0 = double_conv2(6*16, 3*16, non_lin=True)
        self.dconv_up00 = double_conv2(3*16, 16, non_lin=True)

        self.conv_last = nn.Conv2d(16, n_class, 1, stride=2)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def forward_once(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        # print(x.shape)
        #x = self.upsample(x)
        x = self.dconv_up4(x)
        # print(x.shape)
        # print(conv3.shape)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        return x

    def forward(self, x):
        x = [self.forward_once(y) for y in x]
        x = torch.cat(x, axis=1)

        x = self.dconv_up0(x)

        x = self.dconv_up00(x)

        out = self.conv_last(x)

        return torch.sigmoid(out)
