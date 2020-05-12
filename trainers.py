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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_and_test_CNNVAE_BB(model,
                             learning_rate,
                             optimizer,
                             scheduler,
                             trainloader,
                             testloader,
                             name='CNNVAE_BB',
                             epochs=25,
                             threshold=0.5,
                             device=device):
    hidden_dim = 4096 - 6*650
    accuracy_list = []
    best_loss = 1000000000000000

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
                loss = loss_function_CNNVAE(x_hat, x, mu, logvar, device)
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
                        print(create_conf_matrix2(
                            1*(x == 1), 1*(x_hat_post > thld)))
                        print('='*50)
                        print(
                            'Confusion Matrix (Prior) at threshold: {}'.format(thld))
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
                       'models/'+name+'.pth')

        scheduler.step(test_loss_post)
        accuracy_list.append(road_accuracy_prior)
        print("""\nTest set: Average loss: {:.4f}, 
        Accuracy (Post): {}/{} ({:.0f}%) ,
        Accuracy (Prior): {}/{} ({:.0f}%) ,
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
        plt.savefig("imgs/"+name+"_"+str(epoch)+".png", dpi=150)
        plt.close(fig)

    pd.DataFrame(accuracy_list).to_csv(name+'.csv')

    torch.save(codes, name+"_codes.pth")


def train_and_test_CNNVAE_Road(model,
                               learning_rate,
                               optimizer,
                               scheduler,
                               trainloader,
                               testloader,
                               name='CNNVAE_BB',
                               epochs=25,
                               threshold=0.5,
                               device=device):
    hidden_dim = 4096 - 6*650
    accuracy_list = []
    best_loss = 1000000000000000

    codes = dict(mu=list(), logsigma=list(), y=list())
    for epoch in range(0, epochs + 1):
        # Training
        if epoch >= 0:  # test untrained net first
            model.train()
            train_loss = 0
            for i, data in enumerate(trainloader):
                sample, target, road_image, extra = data
                batch_size = len(road_image)
                x = 1.0*torch.stack(road_image).reshape(-1,
                                                        1, 800, 800).to(device)
                #x = torch.zeros((batch_size, 1, 800, 800))
                #x[:, 0, :, :] = torch.zeros((batch_size, 800, 800))
                # for i in range(batch_size):
                #    for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                #        x[i, 0, :, :] = 1.0*convert_to_binary_mask(bb)
                x = x.to(device)
                y = torch.stack(sample).reshape(6, -1, 3, 256, 256).to(device)
                # ===================forward=====================
                x_hat, mu, logvar = model(x, y)
                loss = loss_function_CNNVAE(x_hat, x, mu, logvar, device)
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
                x = 1.0*torch.stack(road_image).reshape(-1,
                                                        1, 800, 800).to(device)
                #x = torch.zeros((batch_size, 1, 800, 800))
                #x[:, 0, :, :] = torch.zeros((batch_size, 800, 800))
                # for i in range(batch_size):
                #    for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                #        x[i, 0, :, :] = 1.0*convert_to_binary_mask(bb)
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
                        print(create_conf_matrix2(
                            1*(x == 1), 1*(x_hat_post > thld)))
                        print('='*50)
                        print(
                            'Confusion Matrix (Prior) at threshold: {}'.format(thld))
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
                       'models/'+name+'.pth')

        scheduler.step(test_loss_post)
        accuracy_list.append(road_accuracy_prior)
        print("""\nTest set: Average loss: {:.4f}, 
        Accuracy (Post): {}/{} ({:.0f}%) ,
        Accuracy (Prior): {}/{} ({:.0f}%) ,
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
        plt.savefig("imgs/"+name+"_"+str(epoch)+".png", dpi=150)
        plt.close(fig)

    pd.DataFrame(accuracy_list).to_csv(name+'.csv')

    torch.save(codes, name + "_codes.pth")


def train_and_test_RotNet(model_cnn,
                          learning_rate,
                          optimizer,
                          scheduler,
                          trainloader,
                          testloader,
                          name,
                          epochs=25):

    accuracy_list = []
    best_loss = 1000000
    best_model = copy.deepcopy(model_cnn)

    def train(epoch, model):
        model.train()
        for batch_idx, (data, target, data_idx) in enumerate(trainloader):
            # send to device
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))

    def test(model):
        global best_loss, best_model
        model.eval()
        test_loss = 0
        correct = 0
        for data, target, data_idx in testloader:
            # send to device
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        test_loss /= len(testloader.dataset)
        accuracy = 100. * correct / len(testloader.dataset)
        accuracy_list.append(accuracy)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(testloader.dataset),
            accuracy))
        if test_loss < best_loss:
            print('Updating best model')
            best_loss = copy.deepcopy(test_loss)
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(),
                       'models/'+name+'.pth')

        return test_loss

    for epoch in range(0, epochs):
        train(epoch, model_cnn)
        test_loss = test(model_cnn)
        scheduler.step(test_loss)


def train_validate_segmentation_model_Roadmap(model,
                                              loss_function,
                                              optimizer,
                                              scheduler,
                                              trainloader,
                                              testloader,
                                              threshold=0.5,
                                              num_epochs=25,
                                              img_w=200,
                                              img_h=200,
                                              name='unet_single_2_200z',
                                              device=device):

    bb_accuracy_list = []
    threat_score_list = []
    m_dict = Counter()
    best_loss = 1000000000000000
    epochs = num_epochs
    for epoch in range(0, epochs + 1):
        # Training
        if epoch >= 0:  # test untrained net first
            model.train()
            train_loss = 0
            for i, data in enumerate(trainloader):
                sample, target, road_image, extra = data
                batch_size = len(road_image)
                x = torch.zeros((batch_size, 1, 800, 800))
                x[:, 0, :, :] = 1.0 * \
                    torch.stack(road_image).reshape(-1, 800, 800)
                #x[:,0,:,:] = torch.zeros((batch_size,800,800))
                # for i in range(batch_size):
                #    for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                #        x[i,0,:,:] = 1.0*convert_to_binary_mask(bb)
                x = x.to(device)
                y = torch.stack(sample).reshape(
                    6, -1, 3, img_w, img_h).to(device)
                # ===================forward=====================
                x_hat = model(y)
                loss = loss_function(x_hat, x, device=device)
                train_loss += loss.item()
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print(
                f'====> Epoch: {epoch} Average loss: {train_loss / len(trainloader.dataset):.9f}')

        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = [0, 0]
            total = [0, 0]
            conf_matrices = [torch.zeros(2, 2).to(
                device), torch.zeros(2, 2).to(device)]
            for batch_idx, data in enumerate(testloader):
                sample, target, road_image, extra = data
                batch_size = len(road_image)
                x = torch.zeros((batch_size, 1, 800, 800))
                x[:, 0, :, :] = 1.0 * \
                    torch.stack(road_image).reshape(-1, 800, 800)
                #x[:,0,:,:] = torch.zeros((batch_size,800,800))
                # for i in range(batch_size):
                #    for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                #        x[i,0,:,:] = 1.0*convert_to_binary_mask(bb)
                x = x.to(device)
                y = torch.stack(sample).reshape(
                    6, -1, 3, img_w, img_h).to(device)

                # ===================forward=====================
                x_hat = model(y)
                test_loss += loss_function(x_hat, x, device=device).item()
                # =====================log=======================

                i = 0
                print('Channel:{}'.format(i))
                correct[i] += (x_hat[:, i, :, :] > threshold).eq(
                    (x[:, i, :, :] == 1).data.view_as((
                        x_hat[:, i, :, :] > threshold))).cpu().sum().item()
                total[i] += x[:, i, :, :].nelement()

                # if batch_idx % 100 == 0:
                #    cur_threat_score = []
                #    for k in range(0,49):
                #        thld = 0.01+k*0.02
                #        TP, TN, FP, FN = classScores(create_conf_matrix2(1*(x[:,i,:,:]==1), 1*(x_hat[:,i,:,:]>thld)))
                #        cur_threat_score.append(TP[1]*1.0/(TP[1]+FP[1]+FN[1]))
                #    m = max(cur_threat_score)
                #    m_idx = [i for i, j in enumerate(cur_threat_score) if j == m]
                #    print('Max threat score: {} at threshold: {}'.format(m,0.01+m_idx*0.02))
                #    m_dict[0.01+m_idx*0.02] += 1
                #    threshold = max(m_dict, key=m_dict.get)
                conf_matrices[i] += create_conf_matrix2(
                    1*(x[:, i, :, :] == 1), 1*(x_hat[:, i, :, :] > threshold))
                # print('='*100)

        bb_accuracy = 100. * correct[0] / total[0]

        if test_loss < best_loss:
            print('Updating best model')
            best_loss = copy.deepcopy(test_loss)
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(),
                       'models/'+name+'_Roadmap_model.pth')

        TP, TN, FP, FN = classScores(conf_matrices[0])

        threat_score = TP[1]*1.0/(TP[1]+FP[1]+FN[1])

        scheduler.step(test_loss)
        bb_accuracy_list.append(bb_accuracy)
        threat_score_list.append(threat_score)
        print("""\nTest set: Average loss: {:.9f}, 
        Accuracy BB: {}/{} ({:.9f}%) ,
        BB: 
        TP {} 
        TN {}
        FP {}
        FN {}
        Threat Score {}""".format(
            test_loss,
            correct[0], total[0], bb_accuracy,
            TP[1], TN[1], FP[1], FN[1], threat_score))

        # labels.append(y.detach())
        # ===================log========================
        # codes['y'].append(torch.cat(labels))
        test_loss /= len(testloader.dataset)
        print(f'====> Test set loss: {test_loss:.9f}')
        fig = plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow((x[0, 0, :, :].squeeze() ==
                    1).detach().cpu().numpy(), cmap='binary')
        plt.subplot(1, 2, 2)
        plt.imshow((x_hat[0, 0, :, :].squeeze() >
                    threshold).detach().cpu().numpy(), cmap='binary')
        plt.savefig('imgs/'+name+'_Roadmap_plot_epoch_' +
                    str(epoch)+'.png', dpi=150)
        plt.close(fig)
    # dictionary of lists
    dict = {'bb_accuracy': bb_accuracy_list, 'threat_score': threat_score_list}
    pd.DataFrame(dict).to_csv(name+'_Roadmap_accuracy_ts_list.csv')

    torch.save(model.state_dict(), 'models/'+name+'_Roadmap_model_final.pth')


def train_validate_segmentation_model_BB(model,
                                         loss_function,
                                         optimizer,
                                         scheduler,
                                         trainloader,
                                         testloader,
                                         threshold=0.5,
                                         num_epochs=25,
                                         img_w=200,
                                         img_h=200,
                                         name='unet_single_2',
                                         device=device):

    bb_accuracy_list = []
    threat_score_list = []
    best_loss = 1000000000000000
    epochs = num_epochs
    for epoch in range(0, epochs + 1):
        # Training
        if epoch >= 0:  # test untrained net first
            model.train()
            train_loss = 0
            for i, data in enumerate(trainloader):
                sample, target, road_image, extra = data
                batch_size = len(road_image)
                x = torch.zeros((batch_size, 1, 800, 800))
                #x[:,0,:,:] = 1.0*torch.stack(road_image).reshape(-1, 800, 800)
                x[:, 0, :, :] = torch.zeros((batch_size, 800, 800))
                for i in range(batch_size):
                    for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                        x[i, 0, :, :] = 1.0*convert_to_binary_mask(bb)
                x = x.to(device)
                y = torch.stack(sample).reshape(
                    6, -1, 3, img_w, img_h).to(device)
                # ===================forward=====================
                x_hat = model(y)
                loss = loss_function(x_hat, x, device=device)
                train_loss += loss.item()
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print(
                f'====> Epoch: {epoch} Average loss: {train_loss / len(trainloader.dataset):.9f}')

        means, logvars, labels = list(), list(), list()
        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = [0, 0]
            total = [0, 0]
            conf_matrices = [torch.zeros(2, 2).to(
                device), torch.zeros(2, 2).to(device)]
            for batch_idx, data in enumerate(testloader):
                sample, target, road_image, extra = data
                batch_size = len(road_image)
                x = torch.zeros((batch_size, 1, 800, 800))
                #x[:,0,:,:] = 1.0*torch.stack(road_image).reshape(-1, 800, 800)
                x[:, 0, :, :] = torch.zeros((batch_size, 800, 800))
                for i in range(batch_size):
                    for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                        x[i, 0, :, :] = 1.0*convert_to_binary_mask(bb)
                x = x.to(device)
                y = torch.stack(sample).reshape(
                    6, -1, 3, img_w, img_h).to(device)

                # ===================forward=====================
                x_hat = model(y)
                test_loss += loss_function(x_hat, x, device=device).item()
                # =====================log=======================

                i = 0
                # print('='*100)
                print('Channel:{}'.format(i))
                correct[i] += (x_hat[:, i, :, :] > threshold).eq(
                    (x[:, i, :, :] == 1).data.view_as((
                        x_hat[:, i, :, :] > threshold))).cpu().sum().item()
                total[i] += x[:, i, :, :].nelement()

                # if batch_idx % 100 == 0:
                #    for k in range(0,49):
                #        thld = 0.01+k*0.02
                #        print('Confusion Matrix at threshold: {} for Channel {}'.format(thld, i))
                #        print(create_conf_matrix2(1*(x[:,i,:,:]==1), 1*(x_hat[:,i,:,:]>thld)))
                #        print('='*50)
                #    print('='*75)
                conf_matrices[i] += create_conf_matrix2(
                    1*(x[:, i, :, :] == 1), 1*(x_hat[:, i, :, :] > threshold))
                # print('='*100)

        bb_accuracy = 100. * correct[0] / total[0]

        if test_loss < best_loss:
            print('Updating best model')
            best_loss = copy.deepcopy(test_loss)
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(),
                       'models/'+name+'_model.pth')

        TP, TN, FP, FN = classScores(conf_matrices[0])

        threat_score = TP[1]*1.0/(TP[1]+FP[1]+FN[1])

        scheduler.step(test_loss)
        bb_accuracy_list.append(bb_accuracy)
        threat_score_list.append(threat_score)
        print("""\nTest set: Average loss: {:.9f}, 
        Accuracy BB: {}/{} ({:.9f}%) ,
        BB: 
        TP {} 
        TN {}
        FP {}
        FN {}
        Threat Score {}""".format(
            test_loss,
            correct[0], total[0], bb_accuracy,
            TP[1], TN[1], FP[1], FN[1], threat_score))

        # labels.append(y.detach())
        # ===================log========================
        # codes['y'].append(torch.cat(labels))
        test_loss /= len(testloader.dataset)
        print(f'====> Test set loss: {test_loss:.9f}')
        fig = plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow((x[0, 0, :, :].squeeze() ==
                    1).detach().cpu().numpy(), cmap='binary')
        plt.subplot(1, 2, 2)
        plt.imshow((x_hat[0, 0, :, :].squeeze() >
                    threshold).detach().cpu().numpy(), cmap='binary')
        plt.savefig('imgs/'+name+'_plot_epoch_'+str(epoch)+'.png', dpi=150)
        plt.close(fig)
    # dictionary of lists
    dict = {'bb_accuracy': bb_accuracy_list, 'threat_score': threat_score_list}
    pd.DataFrame(dict).to_csv(name+'_accuracy_ts_list.csv')

    torch.save(model.state_dict(), 'models/'+name+'_model_final.pth')
