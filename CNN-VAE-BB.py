from custom_helper import (get_n_params,
                           order_points,
                           arrange_box,
                           iou,
                           map_to_ground_truth,
                           calculate_overlap,
                           one_hot_embedding,
                           poly2mask,
                           convert_to_binary_mask,
                           create_conf_matrix,
                           create_conf_matrix2,
                           classScores,
                           split_list)

from models import CNN_VAE, CNN_VAE_transfer
from loss_functions import loss_function_CNNVAE
from trainers import train_and_test_CNNVAE_BB, train_and_test_CNNVAE_Road
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
from sklearn.metrics import confusion_matrix
from skimage import draw
import os
import time
import sys
import random
import psutil
import math
import json
import copy
import pandas as pd
import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file
image_folder = '../DLSP20Dataset/data'
annotation_csv = '../DLSP20Dataset/data/annotation.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.is_available()

# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(106)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(106, 134)

train_scene_index = np.random.choice(
    labeled_scene_index, int(np.ceil(0.8*len(labeled_scene_index))))

test_scene_index = labeled_scene_index[np.isin(
    labeled_scene_index, train_scene_index, invert=True)]


transform = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ])

batch_size = 8

# The labeled dataset can only be retrieved by sample.
# And all the returned data are tuple of tensors, since bounding boxes may have different size
# You can choose whether the loader returns the extra_info. It is optional. You don't have to use it.
labeled_trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=train_scene_index,
                                  transform=transform,
                                  extra_info=True
                                  )

trainloader = torch.utils.data.DataLoader(labeled_trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0,
                                          collate_fn=collate_fn)

labeled_testset = LabeledDataset(image_folder=image_folder,
                                 annotation_file=annotation_csv,
                                 scene_index=test_scene_index,
                                 transform=transform,
                                 extra_info=True
                                 )

testloader = torch.utils.data.DataLoader(labeled_testset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=0,
                                         collate_fn=collate_fn)


# Defining the model


model = CNN_VAE().to(device)
# Setting the optimiser

learning_rate = 1e-2

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(0.5, 0.999)
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.1,
                                                       patience=5,
                                                       verbose=True)

# Training and testing the VAE

train_and_test_CNNVAE_BB(model,
                         learning_rate,
                         optimizer,
                         scheduler,
                         trainloader,
                         testloader,
                         name='CNNVAE_BB',
                         epochs=25,
                         threshold=0.5,
                         device=device)

model = CNN_VAE_transfer().to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate,
    betas=(0.5, 0.999)
)

train_and_test_CNNVAE_BB(model,
                         learning_rate,
                         optimizer,
                         scheduler,
                         trainloader,
                         testloader,
                         name='CNNVAE_BB_transfer',
                         epochs=25,
                         threshold=0.5,
                         device=device)

model = CNN_VAE_transfer().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(0.5, 0.999)
)

train_and_test_CNNVAE_BB(model,
                         learning_rate,
                         optimizer,
                         scheduler,
                         trainloader,
                         testloader,
                         name='CNNVAE_BB_transfer_ft',
                         epochs=25,
                         threshold=0.5,
                         device=device)
