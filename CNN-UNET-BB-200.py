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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.metrics import confusion_matrix


from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
from dice_loss import IoULoss

random.seed(0)
np.random.seed(0)
torch.manual_seed(0);

# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file
image_folder = '../../DLSP20Dataset/data'
annotation_csv = '../../DLSP20Dataset/data/annotation.csv'

#azure
#image_folder = '../../data'
#annotation_csv = '../../data/annotation.csv'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

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
    
    #print('Threat Score: {}'.format(threat_score))
    
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

train_scene_index = np.random.choice(labeled_scene_index, int(np.ceil(0.9*len(labeled_scene_index))))

test_scene_index = labeled_scene_index[np.isin(labeled_scene_index, train_scene_index, invert=True)]

val_scene_index, test_scene_index = split_list(test_scene_index)


#transform=torchvision.transforms.Compose([torchvision.transforms.RandomCrop((200,200)),
#                                          torchvision.transforms.Resize((100,100)),
#                                          torchvision.transforms.ToTensor(),
#                              torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                             ])

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((200,200)),
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

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.BatchNorm2d(out_channels),
        #nn.Dropout(0.5),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.BatchNorm2d(out_channels),
        #nn.Dropout(0.5)
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

    
def single_conv(in_channels, out_channels, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=padding),
        nn.LeakyReLU(negative_slope=0.1),
        nn.BatchNorm2d(out_channels),
        #nn.Dropout(0.5)
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
            #nn.Dropout(0.5)
                            )
    else:
        return nn.Sequential(nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                output_padding=output_padding),
            #nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(out_channels),
            #nn.Dropout(0.5)
                            )
        

    
    
class UNet2(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 48)
        self.dconv_down4 = double_conv(48, 64)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)     
         
        
        self.dconv_up4 = double_conv2(64, 64, 0, non_lin=True)
        self.dconv_up3 = double_conv2(48 + 64, 48, non_lin=True)
        self.dconv_up2 = double_conv2(32 + 48, 32, non_lin=True)
        self.dconv_up1 = double_conv2(32 + 16, 16, non_lin=True)
        
        
        self.dconv_up0 = double_conv2(6*16, 3*16, non_lin=True)
        self.dconv_up00 = double_conv2(3*16, 16, non_lin=True)
        self.dconv_up000 = double_conv2(2*16,16)
        
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

        #print(x.shape)
        #x = self.upsample(x) 
        x = self.dconv_up4(x)
        #print(x.shape)
        #print(conv3.shape)
        x = torch.cat([x, conv3], dim=1)

        
        x = self.dconv_up3(x)        
        x = torch.cat([x, conv2], dim=1)   

        x = self.dconv_up2(x)       
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        return x
    
    def forward(self,x):
        x = [self.forward_once(y) for y in x]
        x = torch.cat(x,axis=1)
        
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
    

    
def unet_weight_map(y, wc=None, w0 = 10, sigma = 5):

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
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w


def unet_weight_map(y, wc=None, w0 = 100, sigma = 3):

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
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def dice_loss_weighted(pred, target, smooth = 1.):
    
    weight = torch.tensor([1/np.sqrt(5110000), 1/np.sqrt(10000)])
    weight_ = weight[target.data.view(-1).long()].view_as(target).to(device)
    
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (weight_ * pred * target).sum(dim=2).sum(dim=2)
    
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

def loss_function_unet(x_hat, x, epoch=None):
    #only weighted bCE for first 8 epochs
    batch_size = x.size(0)
    weight_ = torch.zeros_like(x)
    for i in range(batch_size):
        weight_[i,0,:,:] = torch.from_numpy(unet_weight_map(x[i,0,:,:])).to(device)
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x, reduction='none'
    )
    BCE = (BCE*weight_).mean()

    return BCE 

def loss_function_weighted(x_hat, x, gamma=0.75, epoch=None):
    #only weighted bCE for first 8 epochs
    
    #weighted
    weight = torch.tensor([1/np.sqrt(5110000), 1/np.sqrt(10000)])
    #weight = torch.tensor([1/(5110000), 1/(10000)])
    weight_ = weight[x.data.view(-1).long()].view_as(x).to(device)
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x, reduction='none'
    )
    BCE = (BCE*weight_).mean()

    DICE = dice_loss_weighted(x_hat, x)

    return (1-gamma)*BCE + gamma*DICE

# Training and testing the VAE



def train_validate_segmentation_model_BB(model,
                                         loss_function,
                                         optimizer,
                                         scheduler,
                                         threshold = 0.5,
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
                x = torch.zeros((batch_size,1,800,800))
                #x[:,0,:,:] = 1.0*torch.stack(road_image).reshape(-1, 800, 800)
                x[:,0,:,:] = torch.zeros((batch_size,800,800))
                for i in range(batch_size):
                    for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                        x[i,0,:,:] = 1.0*convert_to_binary_mask(bb)
                x = x.to(device)
                y = torch.stack(sample).reshape(6,-1,3,img_w,img_h).to(device)
                # ===================forward=====================
                x_hat = model(y)
                loss = loss_function(x_hat, x)
                train_loss += loss.item()
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print(f'====> Epoch: {epoch} Average loss: {train_loss / len(trainloader.dataset):.9f}')

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
                x = torch.zeros((batch_size,1,800,800))
                #x[:,0,:,:] = 1.0*torch.stack(road_image).reshape(-1, 800, 800)
                x[:,0,:,:] = torch.zeros((batch_size,800,800))
                for i in range(batch_size):
                    for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                        x[i,0,:,:] = 1.0*convert_to_binary_mask(bb)
                x = x.to(device) 
                y = torch.stack(sample).reshape(6,-1,3,img_w,img_h).to(device)

                # ===================forward=====================
                x_hat = model(y)
                test_loss += loss_function(x_hat, x).item()
                # =====================log=======================

                i = 0
                #print('='*100)
                print('Channel:{}'.format(i))
                correct[i] += (x_hat[:,i,:,:]>threshold).eq(
                    (x[:,i,:,:]==1).data.view_as((
                        x_hat[:,i,:,:]>threshold))).cpu().sum().item()
                total[i] += x[:,i,:,:].nelement()

                #if batch_idx % 100 == 0:
                #    for k in range(0,49):
                #        thld = 0.01+k*0.02
                #        print('Confusion Matrix at threshold: {} for Channel {}'.format(thld, i))
                #        print(create_conf_matrix2(1*(x[:,i,:,:]==1), 1*(x_hat[:,i,:,:]>thld)))
                #        print('='*50)
                #    print('='*75)
                conf_matrices[i] += create_conf_matrix2(1*(x[:,i,:,:]==1), 1*(x_hat[:,i,:,:]>threshold))
                #print('='*100)


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

                #labels.append(y.detach())
        # ===================log========================
        #codes['y'].append(torch.cat(labels))
        test_loss /= len(testloader.dataset)
        print(f'====> Test set loss: {test_loss:.9f}')
        fig = plt.figure(figsize=(10, 6))
        plt.subplot(1,2,1)
        plt.imshow((x[0,0,:,:].squeeze()==1).detach().cpu().numpy(), cmap='binary')
        plt.subplot(1,2,2)
        plt.imshow((x_hat[0,0,:,:].squeeze()>threshold).detach().cpu().numpy(), cmap='binary')
        plt.savefig('imgs/'+name+'_plot_epoch_'+str(epoch)+'.png', dpi=150)
        plt.close(fig)
    # dictionary of lists  
    dict = {'bb_accuracy': bb_accuracy_list, 'threat_score': threat_score_list} 
    pd.DataFrame(dict).to_csv(name+'_accuracy_ts_list.csv')

    torch.save(model.state_dict(), 'models/'+name+'_model_final.pth')

# Training and testing the VAE

def train_validate_segmentation_model_Roadmap(model,
                                         loss_function,
                                         optimizer,
                                         scheduler,
                                         threshold = 0.5,
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
                x = torch.zeros((batch_size,1,800,800))
                x[:,0,:,:] = 1.0*torch.stack(road_image).reshape(-1, 800, 800)
                #x[:,0,:,:] = torch.zeros((batch_size,800,800))
                #for i in range(batch_size):
                #    for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                #        x[i,0,:,:] = 1.0*convert_to_binary_mask(bb)
                x = x.to(device)
                y = torch.stack(sample).reshape(6,-1,3,img_w,img_h).to(device)
                # ===================forward=====================
                x_hat = model(y)
                loss = loss_function(x_hat, x)
                train_loss += loss.item()
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print(f'====> Epoch: {epoch} Average loss: {train_loss / len(trainloader.dataset):.9f}')


        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = [0,0]
            total = [0,0]
            conf_matrices = [torch.zeros(2,2).to(device),torch.zeros(2,2).to(device)]
            for batch_idx, data in enumerate(testloader):
                sample, target, road_image, extra = data
                batch_size = len(road_image)
                x = torch.zeros((batch_size,1,800,800))
                x[:,0,:,:] = 1.0*torch.stack(road_image).reshape(-1, 800, 800)
                #x[:,0,:,:] = torch.zeros((batch_size,800,800))
                #for i in range(batch_size):
                #    for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                #        x[i,0,:,:] = 1.0*convert_to_binary_mask(bb)
                x = x.to(device) 
                y = torch.stack(sample).reshape(6,-1,3,img_w,img_h).to(device)

                # ===================forward=====================
                x_hat = model(y)
                test_loss += loss_function(x_hat, x).item()
                # =====================log=======================

                i = 0
                print('Channel:{}'.format(i))
                correct[i] += (x_hat[:,i,:,:]>threshold).eq(
                    (x[:,i,:,:]==1).data.view_as((
                        x_hat[:,i,:,:]>threshold))).cpu().sum().item()
                total[i] += x[:,i,:,:].nelement()

                #if batch_idx % 100 == 0:
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
                conf_matrices[i] += create_conf_matrix2(1*(x[:,i,:,:]==1), 1*(x_hat[:,i,:,:]>threshold))
                #print('='*100)


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

                #labels.append(y.detach())
        # ===================log========================
        #codes['y'].append(torch.cat(labels))
        test_loss /= len(testloader.dataset)
        print(f'====> Test set loss: {test_loss:.9f}')
        fig = plt.figure(figsize=(10, 6))
        plt.subplot(1,2,1)
        plt.imshow((x[0,0,:,:].squeeze()==1).detach().cpu().numpy(), cmap='binary')
        plt.subplot(1,2,2)
        plt.imshow((x_hat[0,0,:,:].squeeze()>threshold).detach().cpu().numpy(), cmap='binary')
        plt.savefig('imgs/'+name+'_Roadmap_plot_epoch_'+str(epoch)+'.png', dpi=150)
        plt.close(fig)
    # dictionary of lists  
    dict = {'bb_accuracy': bb_accuracy_list, 'threat_score': threat_score_list} 
    pd.DataFrame(dict).to_csv(name+'_Roadmap_accuracy_ts_list.csv')

    torch.save(model.state_dict(), 'models/'+name+'_Roadmap_model_final.pth')

iou_loss = IoULoss()

def loss_function_iou(x_hat, x, gamma = 0.95):
    
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

## Unet model supervised

model = UNet2(1).to(device)
# Setting the optimiser

learning_rate = 3e-4

optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.75, 0.999))


#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                       mode='min', 
                                                       factor=0.1, 
                                                       patience=3,
                                                       verbose=True)

#train_validate_segmentation_model_BB(model=model,
#                                    loss_function=loss_function_iou,
#                                    optimizer=optimizer,
#                                    scheduler=scheduler,
#                                    threshold = 0.5,
#                                    num_epochs=25,
#                                    img_w=200,
#                                    img_h=200,
#                                    name='unet_single_2_200',
#                                    device=device)

#del model

# Unet model tansfer w/o finetuning

unetencoder = UNetEncoder().to(device)

pretrained_dict = torch.load('models/rotation_learning_model_unet.pth', map_location=device)
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
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)     
         
        
        self.dconv_up4 = double_conv2(64, 64, 0, non_lin=True)
        self.dconv_up3 = double_conv2(48 + 64, 48, non_lin=True)
        self.dconv_up2 = double_conv2(32 + 48, 32, non_lin=True)
        self.dconv_up1 = double_conv2(32 + 16, 16, non_lin=True)
        
        
        self.dconv_up0 = double_conv2(6*16, 3*16, non_lin=True)
        self.dconv_up00 = double_conv2(3*16,16, non_lin=True)
        #self.dconv_up000 = double_conv2(2*16,16)
        
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

        #print(x.shape)
        #x = self.upsample(x) 
        x = self.dconv_up4(x)
        #print(x.shape)
        #print(conv3.shape)
        x = torch.cat([x, conv3], dim=1)

        
        x = self.dconv_up3(x)        
        x = torch.cat([x, conv2], dim=1)   

        x = self.dconv_up2(x)       
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        return x
    
    def forward(self,x):
        x = [self.forward_once(y) for y in x]
        x = torch.cat(x,axis=1)
        
        x = self.dconv_up0(x)
        
        x = self.dconv_up00(x)
        
        #x = self.dconv_up000(x)
        
        out = self.conv_last(x)
        
        return torch.sigmoid(out)

model = UNetTransfer(1).to(device)
# Setting the optimiser

learning_rate = 3e-4

optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.75, 0.999))


#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                       mode='min', 
                                                       factor=0.1, 
                                                       patience=3,
                                                       verbose=True)

train_validate_segmentation_model_BB(model=model,
                                    loss_function=loss_function_iou,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    threshold = 0.5,
                                    num_epochs=25,
                                    img_w=200,
                                    img_h=200,
                                    name='unet_single_2_200_transfer',
                                    device=device)

del model

## Transfer Learning finetuned

unetencoder = UNetEncoder().to(device)

pretrained_dict = torch.load('models/rotation_learning_model_unet.pth', map_location=device)
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
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)     
         
        
        self.dconv_up4 = double_conv2(64, 64, 0, non_lin=True)
        self.dconv_up3 = double_conv2(48 + 64, 48, non_lin=True)
        self.dconv_up2 = double_conv2(32 + 48, 32, non_lin=True)
        self.dconv_up1 = double_conv2(32 + 16, 16, non_lin=True)
        
        
        self.dconv_up0 = double_conv2(6*16, 3*16, non_lin=True)
        self.dconv_up00 = double_conv2(3*16,16, non_lin=True)
        self.dconv_up000 = double_conv2(2*16,16)
        
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

        #print(x.shape)
        #x = self.upsample(x) 
        x = self.dconv_up4(x)
        #print(x.shape)
        #print(conv3.shape)
        x = torch.cat([x, conv3], dim=1)

        
        x = self.dconv_up3(x)        
        x = torch.cat([x, conv2], dim=1)   

        x = self.dconv_up2(x)       
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        return x
    
    def forward(self,x):
        x = [self.forward_once(y) for y in x]
        x = torch.cat(x,axis=1)
        
        x = self.dconv_up0(x)
        
        x = self.dconv_up00(x)
        
        #x = self.dconv_up000(x)
        
        out = self.conv_last(x)
        
        return torch.sigmoid(out)

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

train_validate_segmentation_model_BB(model=model,
                                    loss_function=loss_function_iou,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    threshold = 0.5,
                                    num_epochs=25,
                                    img_w=200,
                                    img_h=200,
                                    name='unet_single_2_200_transfer_finetuned',
                                    device=device)

del model