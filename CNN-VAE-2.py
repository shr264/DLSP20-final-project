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

from models import CNN_VAE
from loss_functions import loss_function_CNNVAE
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
image_folder = '../../DLSP20Dataset/data'
annotation_csv = '../../DLSP20Dataset/data/annotation.csv'

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
hidden_dim = 4096 - 6*650
accuracy_list = []
best_loss = 1000000000000000
threshold = 0.5
epochs = 25
codes = dict(mu=list(), logsigma=list(), y=list())
for epoch in range(0, epochs + 1):
    # Training
    if epoch >= 0:  # test untrained net first
        model.train()
        train_loss = 0
        for i, data in enumerate(trainloader):
            sample, target, road_image, extra = data
            batch_size = len(road_image)
            x = torch.zeros((batch_size, 1, 800, 800))
            x[:, 0, :, :] = torch.zeros((batch_size, 800, 800))
            for i in range(batch_size):
                for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                    x[i, 0, :, :] = 1.0*convert_to_binary_mask(bb)
            x = x.to(device)
            y = torch.stack(sample).reshape(6, -1, 3, 256, 256).to(device)
            # ===================forward=====================
            x_hat, mu, logvar = model(x, y)
            loss = loss_function_CNNVAE(x_hat, x, mu, logvar)
            train_loss += loss.item()
            # ===================backward====================
            if not math.isnan(loss.item()):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            else:
                print('warning, NaN')
        # ===================log========================
        print(
            f'====> Epoch: {epoch} Average loss: {train_loss / len(trainloader.dataset):.4f}')

    means, logvars, labels = list(), list(), list()
    with torch.no_grad():
        model.eval()
        test_loss_post = 0
        test_loss_prior = 0
        road_correct_post = 0
        road_correct_prior = 0
        total_road = 0
        conf_matrix_road = torch.zeros(2, 2).to(device)
        for batch_idx, data in enumerate(testloader):
            sample, target, road_image, extra = data
            batch_size = len(road_image)
            x = torch.zeros((batch_size, 1, 800, 800))
            x[:, 0, :, :] = torch.zeros((batch_size, 800, 800))
            for i in range(batch_size):
                for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                    x[i, 0, :, :] = 1.0*convert_to_binary_mask(bb)
            x = x.to(device)
            y = torch.stack(sample).reshape(6, -1, 3, 256, 256).to(device)

            # ===================forward=====================
            mu = torch.mean(mu, 0).repeat(
                batch_size).view(batch_size, hidden_dim)
            logvar = torch.mean(logvar, 0).repeat(
                batch_size).view(batch_size, hidden_dim)
            x_hat_post = model.inference(y, mu, logvar)
            x_hat_prior = model.inference(y)
            test_loss_post += loss_function_CNNVAE(x_hat_post,
                                                   x, mu, logvar, epoch=10).item()
            test_loss_prior += loss_function_CNNVAE(x_hat_prior,
                                                    x, mu, logvar, epoch=10).item()
            # =====================log=======================
            means.append(mu.detach())
            logvars.append(logvar.detach())

            road_correct_post += (x_hat_post > threshold).eq(
                (x == 1).data.view_as((x_hat_post > threshold))).cpu().sum().item()
            road_correct_prior += (x_hat_prior > threshold).eq(
                (x == 1).data.view_as((x_hat_prior > threshold))).cpu().sum().item()
            total_road += x.nelement()

            if batch_idx % 100 == 0:
                for i in range(0, 60):
                    thld = 0.2+i*0.01
                    print('Confusion Matrix (Post) at threshold: {}'.format(thld))
                    print(create_conf_matrix2(1*(x == 1), 1*(x_hat_post > thld)))
                    print('='*50)
                    print('Confusion Matrix (Prior) at threshold: {}'.format(thld))
                    print(create_conf_matrix2(
                        1*(x == 1), 1*(x_hat_prior > thld)))
                    print('='*50)
                print('='*100)
                print('='*100)
            conf_matrix_road += create_conf_matrix2(
                1*(x == 1), 1*(x_hat_post > threshold))

    road_accuracy_post = 100. * road_correct_post / total_road
    road_accuracy_prior = 100. * road_correct_prior / total_road

    if test_loss_post < best_loss:
        print('Updating best model')
        best_loss = copy.deepcopy(test_loss_post)
        best_model = copy.deepcopy(model)
        torch.save(best_model.state_dict(),
                   'models/CNNVAE_BB_model_uw.pth')

    scheduler.step(test_loss_post)
    accuracy_list.append(road_accuracy_prior)
    print("""\nTest set: Average loss: {:.4f}, 
    Accuracy BB (Post): {}/{} ({:.0f}%) ,
    Accuracy BB (Prior): {}/{} ({:.0f}%) ,
    Road: TP {} , 
    TN {}
    FP {}
    FN {}""".format(
        test_loss_post,
        road_correct_post, total_road, road_accuracy_post,
        road_correct_prior, total_road, road_accuracy_prior,
        *classScores(conf_matrix_road)))

    # labels.append(y.detach())
    # ===================log========================
    codes['mu'].append(torch.cat(means))
    codes['logsigma'].append(torch.cat(logvars))
    # codes['y'].append(torch.cat(labels))
    test_loss_post /= len(testloader.dataset)
    test_loss_prior /= len(testloader.dataset)
    print(f'====> Posterior Test set loss: {test_loss_post:.4f}')
    print(f'====> Prior Test set loss: {test_loss_prior:.4f}')
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.imshow((x[0].squeeze() == 1).detach().cpu().numpy(), cmap='binary')
    plt.subplot(1, 3, 2)
    plt.imshow((x_hat_post[0].squeeze() >
                threshold).detach().cpu().numpy(), cmap='binary')
    plt.subplot(1, 3, 3)
    plt.imshow((x_hat_prior[0].squeeze() >
                threshold).detach().cpu().numpy(), cmap='binary')
    plt.savefig("imgs/CNNVAE_plot_epoch_"+str(epoch)+".png", dpi=150)
    plt.close(fig)

pd.DataFrame(accuracy_list).to_csv('CNNVAE_accuracy_list_uw.csv')

torch.save(codes, "CNNVAE_codes_uw.pth")
