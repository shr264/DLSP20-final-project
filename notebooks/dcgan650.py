import os
os.chdir('/scratch/shr264/myjupyter/dl-project-private/code')
import time
import sys
import random

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
from sklearn.metrics import confusion_matrix

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file
image_folder = '../../DLSP20Dataset/data'
annotation_csv = '../../DLSP20Dataset/data/annotation.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(106)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(106, 134)

train_scene_index = np.random.choice(labeled_scene_index, int(np.ceil(0.9*len(labeled_scene_index))))
test_scene_index = labeled_scene_index[np.isin(labeled_scene_index, train_scene_index,invert=True)]


#transform = torchvision.transforms.ToTensor()

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),
                                          torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])

unlabeled_trainset = UnlabeledDataset(image_folder=image_folder, scene_index=unlabeled_scene_index, first_dim='sample', transform=transform)
trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=16, shuffle=True, num_workers=0)

class ConvLayer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=0, 
                 bias = True, 
                 pool=False,
                 mp_kernel_size=2, 
                 mp_stride=2):
        super(ConvLayer, self).__init__()
        if pool:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.LeakyReLU(negative_slope=0.1), ## nn.ReLU(), 
                nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride))
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.LeakyReLU(negative_slope=0.1), ## nn.ReLU(), 
                )
        
    def forward(self, x):
        return self.layer(x)

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearLayer, self).__init__()
        self.layer = nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.1) ## nn.ReLU()
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
        super(ConvTLayer, self).__init__()
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
                nn.LeakyReLU(negative_slope=0.1), ## nn.ReLU()
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
                nn.LeakyReLU(negative_slope=0.1), ## nn.ReLU()
            )        
    def forward(self, x):
        return self.layer(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = ConvLayer(3,16)
        self.conv2 = ConvLayer(96,128)
        self.conv3 = ConvLayer(128,256)
        self.conv4 = ConvLayer(256,512)
        self.conv5 = ConvLayer(512,1024, padding=(1,0))
        
    def forward(self, x):
        x = [y for y in sample]
        x = [self.conv1(y) for y in x]
        x = torch.cat(x,axis=0).reshape(-1,6*16,127,152)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #print(x.shape)
        x = self.conv5(x)
        #print(x.shape)
        return x

class Encoder1(nn.Module):
    def __init__(self, d=650):
        super(Encoder1, self).__init__()
        self.conv1 = ConvLayer(3,96, stride=2)
        self.conv2 = ConvLayer(96,128, stride=2)
        self.conv3 = ConvLayer(128,256, stride=2)
        self.conv4 = ConvLayer(256,512, stride=2)
        self.conv5 = ConvLayer(512,1024, stride=2)
        self.conv6 = ConvLayer(1024,2048, stride=2)
        self.lin1 = nn.Linear(2048*3*3, d)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        #print(x.shape)
        x = self.lin1(x.reshape(-1,2048*3*3))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = Encoder()
        self.linear = nn.Linear(1024*7*7,1)
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.linear(x.reshape(-1,1024*7*7))
        return torch.sigmoid(x)

class Discriminator1(nn.Module):
    def __init__(self, d=650):
        super(Discriminator1, self).__init__()
        self.d = d
        self.encoder = Encoder1()
        self.linear = nn.Linear(self.d,1)
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.linear(x.reshape(-1,self.d))
        return torch.sigmoid(x)

def random_vector(batch_size, length):
    # Sample from a Gaussian distribution
    z_vec = torch.randn(batch_size, length, 1, 1).float()
    if torch.cuda.is_available():
        z_vec = z_vec.to(device)
    return z_vec

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.convt1 = ConvTLayer(4096, 2048, stride=2)
        self.convt2 = ConvTLayer(2048, 1024, stride=2, output_padding=(0,0))
        self.convt3 = ConvTLayer(1024, 512, stride=2, padding=(1,1), output_padding=(1,1))
        self.convt4 = ConvTLayer(512, 256, stride=2, output_padding=(1,1))
        self.convt5 = ConvTLayer(256, 128, stride=2, output_padding=(1,1))
        self.convt6 = ConvTLayer(128, 96, stride=2, output_padding=(1,1))
        self.convt7 = ConvTLayer(96, 64, stride=2, output_padding=(1,1))
        self.convt8 = ConvTLayer(64, 32, stride=1, output_padding=(0,0))
        self.convt9 = ConvTLayer(32, 18, stride=1, padding=(1,1), output_padding=(0,0))
        
    def forward(self,z):
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
        self.convt2 = ConvTLayer(2048, 1024, stride=2, output_padding=(1,1))
        self.convt3 = ConvTLayer(1024, 512, stride=2, padding=(1,1), output_padding=(0,0))
        self.convt4 = ConvTLayer(512, 256, stride=2, output_padding=(0,0))
        self.convt5 = ConvTLayer(256, 128, stride=2, output_padding=(0,0))
        self.convt6 = ConvTLayer(128, 96, stride=2, output_padding=(0,0))
        self.convt7 = ConvTLayer(96, 3, stride=2, output_padding=(1,1))
        
    def forward(self,z):
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
        super(Generator, self).__init__()
        self.decoder = Decoder()
        
    def forward(self,x):
        x = self.decoder(x)
        return torch.tanh(x).reshape(6,-1,3,256,256)

class Generator1(nn.Module):
    def __init__(self):
        super(Generator1, self).__init__()
        self.decoder = Decoder1()
        
    def forward(self,x):
        x = self.decoder(x)
        return torch.tanh(x)

# Build Network
batch_size = 8
z_dim = 4096

D = Discriminator1()
G = Generator1()

# If a gpu is available move all models to gpu
G = G.to(device)
D = D.to(device)

def real_loss(predictions, smooth=False):
    batch_size = predictions.shape[0]
    labels = torch.ones(batch_size)
    # Smooth labels for discriminator to weaken learning
    if smooth:
        labels = labels * 0.9
    # We use the binary cross entropy loss | Model has a sigmoid function
    criterion = nn.BCELoss()
    # Move models to GPU if available
    labels = labels.to(device)
    criterion = criterion.to(device)
    loss = criterion(predictions.squeeze(), labels)
    return loss

def fake_loss(predictions):
    batch_size = predictions.shape[0]
    labels = torch.zeros(batch_size)
    criterion = nn.BCELoss()
    # Move models to GPU if available
    labels = labels.to(device)
    criterion = criterion.to(device)
    loss = criterion(predictions.squeeze(), labels)
    return loss

#-----TRAINING PARAMETERS-----
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
#-----------------------------

# Adam optimizer as trainigs function
d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=[beta1, beta2])
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=[beta1, beta2])

def train_discriminator(generator, discriminator, optimizer, real_data, batch_size, z_size):
    # Set discriminator into training mode and reset gradients
    discriminator.train()
    optimizer.zero_grad()
    # Rescale images into -1 to 1 range
    #real_data = scale(real_data)
    # Train on real data
    real_data_logits = discriminator.forward(real_data)
    loss_real = real_loss(real_data_logits, smooth=True)
    # Train on fake data
    z_vec = random_vector(batch_size, z_size)
    fake_data = generator.forward(z_vec)
    fake_data_logits = discriminator.forward(fake_data)
    loss_fake = fake_loss(fake_data_logits)
    # Calculate total loss, gradients and take optimization step
    total_loss = loss_fake + loss_real
    total_loss.backward()
    optimizer.step()
    return total_loss

def train_generator(generator, discriminator, optimizer, batch_size, z_size):
    # Reset gradients and set model to training mode
    generator.train()
    optimizer.zero_grad()
    # Generate fake data
    z_vec = random_vector(batch_size, z_size)
    fake_data = generator.forward(z_vec)
    # Train generator with output of discriminator
    discriminator_logits = discriminator.forward(fake_data)
    # Reverse labels
    loss = real_loss(discriminator_logits)
    loss.backward()
    optimizer.step()
    return loss


epochs = 250
# After how many batches should generated sample images be shown?
print_every = 1000
# How many images should be shown?
sample_size = 8
# After how many epochs should the loss be plotted?
plot_every = 25
# Create some sample noise
sample_noise = random_vector(sample_size, z_dim)
#-------------------------

# Keep track of losses
d_losses = []
g_losses = []

for e in range(epochs):
    for batch_idx, sample in enumerate(trainloader):
    #for batch_idx, (sample, target, road_image, extra) in enumerate(trainloader):
        # send to device
        #sample = torch.stack(sample).reshape(6,-1,3,256,306)
        sample = sample.to(device)
        batch_size = sample.shape[0]
        
        input_idx = np.random.randint(low=0, high=6, size=1)
        sample = sample[:,input_idx,:,:,:].squeeze()
        
        # Move images to GPU if available
        sample = sample.to(device)
        # Train discriminator
        d_loss = train_discriminator(G, D, d_optimizer, sample, batch_size, z_dim)
        # Train generator
        g_loss = train_generator(G, D, g_optimizer, batch_size, z_dim)
        
        # Keep track of losses
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        
        # Print some sample pictures
        if (batch_idx % print_every == 0):
            print("Epoch: {}, Batch: {}, D-Loss: {}, G-Loss: {}".format(e, batch_idx, d_loss, g_loss))
            # Make sample generation
            G.eval()
            # Generate predictions
            predictions = G.forward(sample_noise)
            plt.imshow(torchvision.utils.make_grid(predictions.reshape(-1,3,256,256)[0]).detach().cpu().numpy().transpose(1, 2, 0))
            plt.savefig("/scratch/shr264/myjupyter/dl-project-private/code/imgs/DCGAN_generator_batch_"+str(batch_idx)+".png", dpi=150)
    if (e % plot_every == 0):
        # Print losses
        plt.plot(d_losses, label="Discriminator", alpha=0.5)
        plt.plot(g_losses, label="Generator", alpha=0.5)
        plt.title("Trainings loss")
        plt.legend()
        plt.savefig("/scratch/shr264/myjupyter/dl-project-private/code/imgs/DCGAN_losses_epoch_"+str(e)+".png", dpi=150)
        torch.save(G.state_dict(), "/scratch/shr264/myjupyter/dl-project-private/code/models/DCGAN_G_generator_epoch_"+str(e)+".pth")
        torch.save(D.state_dict(), "/scratch/shr264/myjupyter/dl-project-private/code/models/DCGAN_D_generator_epoch_"+str(e)+".pth")
