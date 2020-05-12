import os
import time
import sys
import random
import psutil

import json
import copy
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
torch.manual_seed(0);

# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file
#image_folder = '../../DLSP20Dataset/data'
#annotation_csv = '../../DLSP20Dataset/data/annotation.csv'

#azure
image_folder = '../../data'
annotation_csv = '../../data/annotation.csv'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.is_available()


# function to count number of parameters
def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

def order_points(pts):
    from scipy.spatial import distance as dist
    import numpy as np
    
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")

def arrange_box(x1,y1):
    box=np.array(list(zip(x1,y1)))
    box=order_points(box)
    return box

def iou(box1, box2):
    from shapely.geometry import Polygon
    a = Polygon(torch.t(box1)).convex_hull
    b = Polygon(torch.t(box2)).convex_hull
    
    return a.intersection(b).area / a.union(b).area

#def iou(xy1,xy2):
#    
#    from shapely.geometry import Polygon
#    
#    boxA = Polygon(arrange_box(xy1[0],xy1[1])).buffer(1e-9)
#    boxB = Polygon(arrange_box(xy2[0],xy2[1])).buffer(1e-9)
#    
#    try:
#        return boxA.intersection(boxB).area / boxA.union(boxB).area
#    except:
#        print('Box 1:',xy1[0],xy1[1])
#        print('Box 2:',xy2[0],xy2[1])
#        sys.exit(1)

def map_to_ground_truth(overlaps, print_it=False):
    prior_overlap, prior_idx = overlaps.max(1)
    if print_it: print(prior_overlap)
#     pdb.set_trace()
    gt_overlap, gt_idx = overlaps.max(0)
    gt_overlap[prior_idx] = 1.99
    for i,o in enumerate(prior_idx): gt_idx[o] = i
    return gt_overlap,gt_idx

def calculate_overlap(target_bb, predicted_bb):
    overlaps = torch.zeros(target_bb.size(0),predicted_bb.size(0))

    for j in range(overlaps.shape[0]):
        for k in range(overlaps.shape[1]):
            overlaps[j][k] = iou(target_bb[j],predicted_bb[k])
            
    return overlaps

def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]

from skimage import draw
import numpy as np

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = torch.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def convert_to_binary_mask(corners, shape=(800,800)):
    point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])
    x,y = point_squence.T[0].detach() * 10 + 400, -point_squence.T[1].detach() * 10 + 400
    new_im = poly2mask(y, x, shape)
    return new_im

def create_conf_matrix(target, pred, debug=True):
    import sys
    
    target = target.reshape(-1)
    pred = pred.reshape(-1)
    
    if debug:
        print('Target values:', target.unique())
        print('Predicted values:', pred.unique())
        print('Target shape:', target.shape)
        print('Predicted shape:', pred.shape)
    
    nb_classes = max(target.unique())
    if len(pred.unique()) > (nb_classes+1) :
        print('More predicted classes than true classes')
        sys.exit(1)
        
    conf_matrix = torch.zeros(nb_classes+1, nb_classes+1)
    for t, p in zip(target, pred):
        conf_matrix[t, p] += 1
    
    return conf_matrix

def create_conf_matrix2(target, pred, debug=True):
    import sys
    
    target = target.reshape(-1).cpu().numpy()
    pred = pred.reshape(-1).cpu().numpy()
    
        
    conf_matrix = torch.from_numpy(confusion_matrix(target, pred)).to(device)
    threat_score = (1.0*conf_matrix[1,1])/(conf_matrix[1,1]+conf_matrix[1,0]+conf_matrix[0,1])
    
    print('Threat Score: {}'.format(threat_score))
    
    return conf_matrix

def classScores(conf_matrix):
    print('Confusion matrix\n', conf_matrix)
    TP = conf_matrix.diag()
    TN = torch.zeros_like(TP)
    FP = torch.zeros_like(TP)
    FN = torch.zeros_like(TP)
    for c in range(conf_matrix.size(0)):
        idx = torch.ones(conf_matrix.size(0)).byte()
        idx[c] = 0
        # all non-class samples classified as non-class
        TN[c] = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
        # all non-class samples classified as class
        FP[c] = conf_matrix[idx, c].sum()
        # all class samples not classified as class
        FN[c] = conf_matrix[c, idx].sum()

        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
            c, TP[c], TN[c], FP[c], FN[c]))
        
    return (TP.detach().cpu().numpy(), 
            TN.detach().cpu().numpy(), 
            FP.detach().cpu().numpy(), 
            FN.detach().cpu().numpy())

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]


# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(106)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(106, 134)

train_scene_index = np.random.choice(labeled_scene_index, int(np.ceil(0.8*len(labeled_scene_index))))

test_scene_index = labeled_scene_index[np.isin(labeled_scene_index, train_scene_index,invert=True)]


transform=torchvision.transforms.Compose([torchvision.transforms.Resize((200,200)),
                                          torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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




def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 48)
        self.dconv_down4 = double_conv(48, 64)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(48 + 64, 48)
        self.dconv_up2 = double_conv(32 + 48, 32)
        self.dconv_up1 = double_conv(32 + 16, 16)
        
        
        self.dconv_up0 = double_conv(6*16, 3*16)
        self.dconv_up00 = double_conv(3*16,16)
        
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
    
    def forward(self,x):
        x = [self.forward_once(y) for y in x]
        x = torch.cat(x,axis=1)
        
        x = self.upsample(x)
        x = self.dconv_up0(x)
        
        x = self.upsample(x)
        x = self.dconv_up00(x)
        
        out = self.conv_last(x)
        
        return torch.sigmoid(out)
    
    
   
    
model = UNet(2).to(device)
# Setting the optimiser

learning_rate = 1e-3

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

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

# Reconstruction + KL divergence losses summed over all elements and batch

def loss_function(x_hat, x, epoch=None):
    #only weighted bCE for first 8 epochs
    if epoch!=None and epoch < 15:
        #weighted
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


# Training and testing the VAE

road_accuracy_list = []
bb_accuracy_list = []
best_loss = 100000000000
threshold = 0.5
epochs = 25
for epoch in range(0, epochs + 1):
    # Training
    if epoch >= 0:  # test untrained net first
        model.train()
        train_loss = 0
        for i, data in enumerate(trainloader):
            sample, target, road_image, extra = data
            batch_size = len(road_image)
            x = torch.zeros((batch_size,2,800,800))
            x[:,0,:,:] = 1.0*torch.stack(road_image).reshape(-1, 800, 800)
            x[:,1,:,:] = torch.zeros((batch_size,800,800))
            for i in range(batch_size):
                for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                    x[i,1,:,:] = 1.0*convert_to_binary_mask(bb)
            x = x.to(device)
            y = torch.stack(sample).reshape(6,-1,3,200,200).to(device)
            # ===================forward=====================
            x_hat = model(y)
            loss = loss_function(x_hat, x)
            train_loss += loss.item()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(trainloader.dataset):.4f}')

    means, logvars, labels = list(), list(), list()
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = [0,0]
        total = [0,0]
        conf_matrices = [torch.zeros(2,2).to(device),torch.zeros(2,2).to(device)]
        for batch_idx, data in enumerate(testloader):
            sample, target, road_image, extra = data
            batch_size = len(road_image)
            x = torch.zeros((batch_size,2,800,800))
            x[:,0,:,:] = 1.0*torch.stack(road_image).reshape(-1, 800, 800)
            x[:,1,:,:] = torch.zeros((batch_size,800,800))
            for i in range(batch_size):
                for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                    x[i,1,:,:] = 1.0*convert_to_binary_mask(bb)
            x = x.to(device) 
            y = torch.stack(sample).reshape(6,-1,3,200,200).to(device)
            
            # ===================forward=====================
            x_hat = model(y)
            test_loss += loss_function(x_hat, x).item()
            # =====================log=======================
            
            for i in [0,1]:
                print('='*100)
                print('Channel:{}'.format(i))
                correct[i] += (x_hat[:,i,:,:]>threshold).eq(
                    (x[:,i,:,:]==1).data.view_as((
                        x_hat[:,i,:,:]>threshold))).cpu().sum().item()
                total[i] += x[:,i,:,:].nelement()

                if batch_idx % 100 == 0:
                    for k in range(0,60):
                        thld = 0.2+k*0.01
                        print('Confusion Matrix (Post) at threshold: {} for Channel {}'.format(thld, i))
                        print(create_conf_matrix2(1*(x[:,i,:,:]==1), 1*(x_hat[:,i,:,:]>thld)))
                        print('='*50)
                    print('='*75)
                conf_matrices[i] += create_conf_matrix2(1*(x[:,i,:,:]==1), 1*(x_hat[:,i,:,:]>threshold))
                print('='*100)
            
                       
    road_accuracy = 100. * correct[0] / total[0]
    bb_accuracy = 100. * correct[1] / total[1]
    
    if test_loss < best_loss:
        print('Updating best model')
        best_loss = copy.deepcopy(test_loss)
        best_model = copy.deepcopy(model)
        torch.save(best_model.state_dict(), 
                   'models/unet_dual_model.pth')

        
    scheduler.step(test_loss)
    road_accuracy_list.append(road_accuracy)
    bb_accuracy_list.append(bb_accuracy)
    print("""\nTest set: Average loss: {:.4f}, 
    Accuracy Road: {}/{} ({:.0f}%) ,
    Accuracy BB: {}/{} ({:.0f}%) ,
    Road: 
    TP {} 
    TN {}
    FP {}
    FN {};
    BB: 
    TP {}
    TN {}
    FP {}
    FN {}""".format(
        test_loss, 
        correct[0], total[0], road_accuracy,
        correct[1], total[1], bb_accuracy,
        *classScores(conf_matrices[0]),
        *classScores(conf_matrices[1]) ))

            #labels.append(y.detach())
    # ===================log========================
    #codes['y'].append(torch.cat(labels))
    test_loss /= len(testloader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(1,2,1)
    plt.imshow((x[0,0,:,:].squeeze()==1).detach().cpu().numpy(), cmap='binary')
    plt.imshow((x[0,1,:,:].squeeze()==1).detach().cpu().numpy(), cmap='binary')
    plt.subplot(1,2,2)
    plt.imshow((x_hat[0,0,:,:].squeeze()>threshold).detach().cpu().numpy(), cmap='binary')
    plt.imshow((x_hat[0,1,:,:].squeeze()>threshold).detach().cpu().numpy(), cmap='binary')
    plt.savefig("imgs/unet_dual_plot_epoch_"+str(epoch)+".png", dpi=150)
    plt.close(fig)
    
pd.DataFrame([road_accuracy_list,bb_accuracy_list], 
             columns=['road_accuracy','bb_accuracy']).to_csv('unet_dual_accuracy_list.csv')
