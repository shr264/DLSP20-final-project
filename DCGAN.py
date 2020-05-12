from models import Discriminator1, Generator1
from sklearn.metrics import confusion_matrix
from helper import collate_fn, draw_box
from data_helper import UnlabeledDataset, LabeledDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import random
import sys
import time
import os
os.chdir('/scratch/shr264/myjupyter/dl-project-private/code')

matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

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

train_scene_index = np.random.choice(
    labeled_scene_index, int(np.ceil(0.9*len(labeled_scene_index))))
test_scene_index = labeled_scene_index[np.isin(
    labeled_scene_index, train_scene_index, invert=True)]


#transform = torchvision.transforms.ToTensor()

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ])

unlabeled_trainset = UnlabeledDataset(
    image_folder=image_folder, scene_index=unlabeled_scene_index, first_dim='sample', transform=transform)
trainloader = torch.utils.data.DataLoader(
    unlabeled_trainset, batch_size=16, shuffle=True, num_workers=0)

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


# -----TRAINING PARAMETERS-----
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
# -----------------------------

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
# -------------------------

# Keep track of losses
d_losses = []
g_losses = []

for e in range(epochs):
    for batch_idx, sample in enumerate(trainloader):
        # for batch_idx, (sample, target, road_image, extra) in enumerate(trainloader):
        # send to device
        #sample = torch.stack(sample).reshape(6,-1,3,256,306)
        sample = sample.to(device)
        batch_size = sample.shape[0]

        input_idx = np.random.randint(low=0, high=6, size=1)
        sample = sample[:, input_idx, :, :, :].squeeze()

        # Move images to GPU if available
        sample = sample.to(device)
        # Train discriminator
        d_loss = train_discriminator(
            G, D, d_optimizer, sample, batch_size, z_dim)
        # Train generator
        g_loss = train_generator(G, D, g_optimizer, batch_size, z_dim)

        # Keep track of losses
        d_losses.append(d_loss)
        g_losses.append(g_loss)

        # Print some sample pictures
        if (batch_idx % print_every == 0):
            print("Epoch: {}, Batch: {}, D-Loss: {}, G-Loss: {}".format(e,
                                                                        batch_idx, d_loss, g_loss))
            # Make sample generation
            G.eval()
            # Generate predictions
            predictions = G.forward(sample_noise)
            plt.imshow(torchvision.utils.make_grid(
                predictions.reshape(-1, 3, 256, 256)[0]).detach().cpu().numpy().transpose(1, 2, 0))
            plt.savefig("/scratch/shr264/myjupyter/dl-project-private/code/imgs/DCGAN_generator_batch_" +
                        str(batch_idx)+".png", dpi=150)
    if (e % plot_every == 0):
        # Print losses
        plt.plot(d_losses, label="Discriminator", alpha=0.5)
        plt.plot(g_losses, label="Generator", alpha=0.5)
        plt.title("Trainings loss")
        plt.legend()
        plt.savefig(
            "/scratch/shr264/myjupyter/dl-project-private/code/imgs/DCGAN_losses_epoch_"+str(e)+".png", dpi=150)
        torch.save(G.state_dict(
        ), "/scratch/shr264/myjupyter/dl-project-private/code/models/DCGAN_G_generator_epoch_"+str(e)+".pth")
        torch.save(D.state_dict(
        ), "/scratch/shr264/myjupyter/dl-project-private/code/models/DCGAN_D_generator_epoch_"+str(e)+".pth")
