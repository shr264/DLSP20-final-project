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
image_folder = '../../DLSP20Dataset/data'
annotation_csv = '../../DLSP20Dataset/data/annotation.csv'

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
    
    print('Threat Score: {}'.format((1.0*conf_matrix[1,1])/(conf_matrix[1,1]+conf_matrix[1,0]+conf_matrix[0,1])))
    
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


transform=torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),
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

class Encoder1(nn.Module):
    def __init__(self, d):
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
        x = self.lin1(x.reshape(-1,2048*3*3))
        return x

class CNN(nn.Module):
    def __init__(self, d=650, output_size=4):
        super(CNN, self).__init__()
        self.encoder = Encoder1(d=d)
        self.linear = nn.Linear(d,4)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
    
class Encoder(nn.Module):
    def __init__(self, d=650):
        super(Encoder, self).__init__()
        self.encoder = Encoder1(d=d)
        
    def forward(self, x):
        x = self.encoder(x)
        return x


class EncoderY(nn.Module):
    def __init__(self,  d):
        super(EncoderY, self).__init__()
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

class EncoderX(nn.Module):
    def __init__(self, d):
        super(EncoderX, self).__init__()
        self.conv1 = ConvLayer(1,16, stride=2)
        self.conv2 = ConvLayer(16,32, stride=2)
        self.conv3 = ConvLayer(32,48, stride=2)
        self.conv4 = ConvLayer(48,64, stride=2)
        self.conv5 = ConvLayer(64,96, stride=2)
        self.conv6 = ConvLayer(96,128, stride=2)
        self.conv7 = ConvLayer(128,256, stride=2)
        self.conv8 = ConvLayer(256,512, stride=2)
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
        #print(x.shape)
        x = self.lin1(x.reshape(-1,512*2*2))
        return x

class DecoderX(nn.Module):
    def __init__(self):
        super(DecoderX, self).__init__()
        self.convt1 = ConvTLayer(4096, 2048, kernel_size=3, stride=2)
        self.convt2 = ConvTLayer(2048, 1024, kernel_size=3, stride=3, output_padding=(0,0))
        self.convt3 = ConvTLayer(1024, 512, kernel_size=3, stride=2, padding=(1,1), output_padding=(0,0))
        self.convt4 = ConvTLayer(512, 256, kernel_size=3, stride=3, padding=(1,1), output_padding=(0,0))
        self.convt5 = ConvTLayer(256, 128, kernel_size=3, stride=2, output_padding=(0,0))
        self.convt6 = ConvTLayer(128, 96, kernel_size=3, stride=2, output_padding=(0,0))
        self.convt7 = ConvTLayer(96, 64, kernel_size=3, stride=2, output_padding=(0,0))
        self.convt8 = ConvTLayer(64, 1, kernel_size=3, stride=2, output_padding=(1,1))
        
    def forward(self,z):
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
    def __init__(self, hidden_d=196, image_d=650): #hidden_d=196, image_d=650 or hidden_d=286, image_d=625
        super().__init__()
        
        self.d = hidden_d
        self.id = image_d
        
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
        #print(mu_logvar.shape)
        img_enc = [self.y_encoder(img.squeeze()) for img in y] 
        mu = mu_logvar[:, 0, :]
        #print(mu.shape)
        logvar = mu_logvar[:, 1, :]
        #print(logvar.shape)
        z = self.reparameterise(mu, logvar)
        img_enc.append(z)
        out = torch.cat(img_enc,axis=1).reshape(-1,4096,1,1)
        return self.x_decoder(out), mu, logvar
    
    def inference(self, y, mu=None, logvar=None):
        N = y.size(1)
        z = torch.randn((N, self.d)).to(device)
        #print('Prior:',z.shape)
        if mu is not None and logvar is not None:
            #print(mu.shape)
            #print(logvar.shape)
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            z = eps.mul(std).add_(mu)
            #print('Post:',z.shape)
        z = z.reshape(-1,self.d)
        img_enc = [self.y_encoder(img.squeeze()) for img in y] 
        img_enc.append(z)
        out = torch.cat(img_enc,axis=1).reshape(-1,4096,1,1)
        return self.x_decoder(out)
    

class DecoderC(nn.Module):
    def __init__(self):
        super(DecoderC, self).__init__()
        self.convt1 = ConvTLayer(3900, 2048, kernel_size=3, stride=2)
        self.convt2 = ConvTLayer(2048, 1024, kernel_size=3, stride=3, output_padding=(0,0))
        self.convt3 = ConvTLayer(1024, 512, kernel_size=3, stride=2, padding=(1,1), output_padding=(0,0))
        self.convt4 = ConvTLayer(512, 256, kernel_size=3, stride=3, padding=(1,1), output_padding=(0,0))
        self.convt5 = ConvTLayer(256, 128, kernel_size=3, stride=2, output_padding=(0,0))
        self.convt6 = ConvTLayer(128, 96, kernel_size=3, stride=2, output_padding=(0,0))
        self.convt7 = ConvTLayer(96, 64, kernel_size=3, stride=2, output_padding=(0,0))
        self.convt8 = ConvTLayer(64, 1, kernel_size=3, stride=2, output_padding=(1,1))
        
    def forward(self,z):
        z = self.convt1(z)
        z = self.convt2(z)
        z = self.convt3(z)
        z = self.convt4(z)
        z = self.convt5(z)
        z = self.convt6(z)
        z = self.convt7(z)
        z = self.convt8(z)
        return torch.sigmoid(z)
    

    
class CNN(nn.Module):
    def __init__(self, image_d=650): #hidden_d=196, image_d=650 or hidden_d=286, image_d=625
        super().__init__()
        self.id = image_d
        self.y_encoder = EncoderY(d=self.id)
        self.x_decoder = DecoderC()

    def forward(self, y):
        img_enc = [self.y_encoder(img.squeeze()) for img in y] 
        out = torch.cat(img_enc,axis=1).reshape(-1,3900,1,1)
        return self.x_decoder(out)
    
    def inference(self, y, mu=None, logvar=None):
        img_enc = [self.y_encoder(img.squeeze()) for img in y] 
        out = torch.cat(img_enc,axis=1).reshape(-1,3900,1,1)
        return self.x_decoder(out)
    
    
    
model = CNN().to(device)
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

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.sum()

# Reconstruction + KL divergence losses summed over all elements and batch

def loss_function(x_hat, x, epoch=None):
    #only weighted bCE for first 8 epochs
    if epoch!=None and epoch < 15:
        #weighted
        weight = torch.tensor([1, 1000])
        weight_ = weight[x.view(-1, 800*800).data.view(-1).long()].view_as(x.view(-1, 800*800)).to(device)
        BCE = nn.functional.binary_cross_entropy(
            x_hat.view(-1, 800*800), x.view(-1, 800*800), reduction='none'
        )
        BCE = (BCE*weight_).sum()
    else:
        BCE = nn.functional.binary_cross_entropy(
            x_hat.view(-1, 800*800), x.view(-1, 800*800), reduction='sum'
        )
    
    DICE = dice_loss(x_hat, x)

    return BCE + DICE


# Training and testing the VAE

accuracy_list = []
best_loss = 1000000000000000
threshold = 0.5
epochs = 25
codes = dict(μ=list(), logσ2=list(), y=list())
for epoch in range(0, epochs + 1):
    # Training
    if epoch >= 0:  # test untrained net first
        model.train()
        train_loss = 0
        for i, data in enumerate(trainloader):
            sample, target, road_image, extra = data
            batch_size = len(road_image)
            x = torch.zeros((batch_size,1,800,800))
            x[:,0,:,:] = torch.zeros((batch_size,800,800))
            for i in range(batch_size):
                for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                    x[i,0,:,:] = 1.0*convert_to_binary_mask(bb)
            x = x.to(device) 
            y = torch.stack(sample).reshape(6,-1,3,256,256).to(device)
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
        road_correct = 0
        total_road = 0
        conf_matrix_road = torch.zeros(2,2).to(device)
        for batch_idx, data in enumerate(testloader):
            sample, target, road_image, extra = data
            batch_size = len(road_image)
            x = torch.zeros((batch_size,1,800,800))
            x[:,0,:,:] = torch.zeros((batch_size,800,800))
            for i in range(batch_size):
                for cat, bb in zip(target[i]['category'], target[i]['bounding_box']):
                    x[i,0,:,:] = 1.0*convert_to_binary_mask(bb)
            x = x.to(device) 
            y = torch.stack(sample).reshape(6,-1,3,256,256).to(device)
            
            # ===================forward=====================
            x_hat = model.inference(y)
            test_loss += loss_function(x_hat, x).item()
            # =====================log=======================
            
            road_correct += (x_hat>threshold).eq((x==1).data.view_as((x_hat>threshold))).cpu().sum().item()
            total_road += x.nelement()
            
            if batch_idx % 100 == 0:
                for i in range(0,60):
                    thld = 0.2+i*0.01
                    print('Confusion Matrix (Post) at threshold: {}'.format(thld))
                    print(create_conf_matrix2(1*(x==1), 1*(x_hat>thld)))
                    print('='*50)
                print('='*100)
                print('='*100)
            conf_matrix_road += create_conf_matrix2(1*(x==1), 1*(x_hat>threshold))
                       
    road_accuracy = 100. * road_correct / total_road
    
    if test_loss < best_loss:
        print('Updating best model')
        best_loss = copy.deepcopy(test_loss)
        best_model = copy.deepcopy(model)
        torch.save(best_model.state_dict(), 
                   'models/CNN_BB_model.pth')

        
    scheduler.step(test_loss)
    accuracy_list.append(road_accuracy)
    print("""\nTest set: Average loss: {:.4f}, 
    Accuracy Road (Post): {}/{} ({:.0f}%) ,
    Road: TP {} , 
    TN {}
    FP {}
    FN {}""".format(
        test_loss, 
        road_correct, total_road, road_accuracy, 
        *classScores(conf_matrix_road)))

            #labels.append(y.detach())
    # ===================log========================
    #codes['y'].append(torch.cat(labels))
    test_loss /= len(testloader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(1,2,1)
    plt.imshow((x[0].squeeze()==1).detach().cpu().numpy(), cmap='binary')
    plt.subplot(1,2,2)
    plt.imshow((x_hat[0].squeeze()>threshold).detach().cpu().numpy(), cmap='binary')
    plt.savefig("imgs/CNN_plot_epoch_"+str(epoch)+".png", dpi=150)
    plt.close(fig)
    
pd.DataFrame(accuracy_list).to_csv('CNN_accuracy_list.csv')
