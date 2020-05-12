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

from models import CNN_VAE, CNN_VAE_transfer, UNetTransfer, UNet2
from loss_functions import loss_function_iou
from trainers import train_validate_segmentation_model_Roadmap, train_validate_segmentation_model_BB
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
    labeled_scene_index, int(np.ceil(0.9*len(labeled_scene_index))))

test_scene_index = labeled_scene_index[np.isin(
    labeled_scene_index, train_scene_index, invert=True)]

val_scene_index, test_scene_index = split_list(test_scene_index)


# transform=torchvision.transforms.Compose([torchvision.transforms.RandomCrop((200,200)),
#                                          torchvision.transforms.Resize((100,100)),
#                                          torchvision.transforms.ToTensor(),
#                              torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                             ])

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((200, 200)),
                                            torchvision.transforms.ToTensor(),
                                            #                              torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ])

batch_size = 4

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
                                          num_workers=2,
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
                                         num_workers=2,
                                         collate_fn=collate_fn)

# Training and testing the VAE

# Unet model supervised


model = UNet2(1).to(device)
# Setting the optimiser

learning_rate = 3e-4

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.75, 0.999))


#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.1,
                                                       patience=3,
                                                       verbose=True)

train_validate_segmentation_model_Roadmap(model=model,
                                          loss_function=loss_function_iou,
                                          optimizer=optimizer,
                                          scheduler=scheduler,
                                          trainloader=trainloader,
                                          testloader=testloader,
                                          threshold=0.5,
                                          num_epochs=25,
                                          img_w=200,
                                          img_h=200,
                                          name='unet_single_2_200',
                                          device=device)

#del model

# Unet model tansfer w/o finetuning

model = UNetTransfer(1).to(device)
# Setting the optimiser

learning_rate = 3e-4

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.75, 0.999))


#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.1,
                                                       patience=3,
                                                       verbose=True)

train_validate_segmentation_model_Roadmap(model=model,
                                          loss_function=loss_function_iou,
                                          optimizer=optimizer,
                                          scheduler=scheduler,
                                          trainloader=trainloader,
                                          testloader=testloader,
                                          threshold=0.5,
                                          num_epochs=25,
                                          img_w=200,
                                          img_h=200,
                                          name='unet_single_2_200_transfer',
                                          device=device)

del model

# Transfer Learning finetuned

model = UNetTransfer(1).to(device)
# Setting the optimiser

learning_rate = 3e-4

# Observe that all parameters are being optimized
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.75, 0.999))


#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.1,
                                                       patience=3,
                                                       verbose=True)

train_validate_segmentation_model_Roadmap(model=model,
                                          loss_function=loss_function_iou,
                                          optimizer=optimizer,
                                          scheduler=scheduler,
                                          trainloader=trainloader,
                                          testloader=testloader,
                                          threshold=0.5,
                                          num_epochs=25,
                                          img_w=200,
                                          img_h=200,
                                          name='unet_single_2_200_transfer_finetuned',
                                          device=device)

del model
