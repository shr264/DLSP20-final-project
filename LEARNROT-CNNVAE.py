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

from models import CNN_ROT_UNET, CNNVAE_ROT
from trainers import train_and_test_RotNet
from skimage import draw
from random import sample
from sklearn.metrics import confusion_matrix
from helper import collate_fn, draw_box
from data_helper import UnlabeledDataset, LabeledDataset, UnlabeledRotationDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
from PIL import Image
import time
import sys
import copy
import random

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file

# ON PRINCE
image_folder = '../DLSP20Dataset/data'
annotation_csv = '../DLSP20Dataset/data/annotation.csv'


# ON WORK LAPTOP
# image_folder = '/Users/rasy7001/Documents/DeepLearning/competition /data'
# annotation_csv = '/Users/rasy7001/Documents/DeepLearning/competition /data'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.cuda.is_available()


# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(106)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(106, 134)

# training for rotation
train_scene_index = np.random.choice(
    unlabeled_scene_index, int(np.ceil(0.95*len(unlabeled_scene_index))))

# test for rotation
test_scene_index = unlabeled_scene_index[np.isin(
    unlabeled_scene_index, train_scene_index, invert=True)]


# transform = torchvision.transforms.ToTensor()

# transform=torchvision.transforms.Compose([torchvision.transforms.RandomCrop((200,200)),
#                                          torchvision.transforms.Resize((100,100)),
# torchvision.transforms.ToTensor(),
#                             #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ])

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ])

# The dataset class for unlabeled data.


# Get individual image

unlabeled_trainset = UnlabeledDataset(image_folder=image_folder,
                                      scene_index=unlabeled_scene_index,
                                      first_dim='image',
                                      transform=transform)
trainloader = torch.utils.data.DataLoader(
    unlabeled_trainset, batch_size=2, shuffle=True, num_workers=2)

# Rotation Representation Learning

batch_size = 16

unlabeled_trainset = UnlabeledRotationDataset(image_folder=image_folder,
                                              scene_index=train_scene_index,
                                              first_dim='image',
                                              transform=transform)
trainloader = torch.utils.data.DataLoader(
    unlabeled_trainset, batch_size=batch_size, shuffle=True, num_workers=0)

unlabeled_testset = UnlabeledRotationDataset(image_folder=image_folder,
                                             scene_index=test_scene_index,
                                             first_dim='image',
                                             transform=transform)
testloader = torch.utils.data.DataLoader(
    unlabeled_testset, batch_size=batch_size, shuffle=True, num_workers=0)

model_cnn = CNNVAE_ROT()
model_cnn = model_cnn.to(device)

learning_rate = 1e-3
optimizer = torch.optim.Adam(
    model_cnn.parameters(),
    lr=learning_rate,
    betas=(0.5, 0.999)
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.1,
                                                       patience=5,
                                                       verbose=True)
print('Number of parameters: {}'.format(get_n_params(model_cnn)))

train_and_test_RotNet(model_cnn,
                       learning_rate,
                       optimizer,
                       scheduler,
                       trainloader,
                       testloader,
                       name='rotation_learning_model',
                       epochs=25)
