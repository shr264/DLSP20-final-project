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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
from sklearn.metrics import confusion_matrix
from random import sample

random.seed(0)
np.random.seed(0)
torch.manual_seed(0);

# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file

##### ON PRINCE
image_folder = '../../DLSP20Dataset/data'
annotation_csv = '../../DLSP20Dataset/data/annotation.csv'


##### ON WORK LAPTOP
#image_folder = '/Users/rasy7001/Documents/DeepLearning/competition /data'
#annotation_csv = '/Users/rasy7001/Documents/DeepLearning/competition /data'

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
        
    return TP, TN, FP, FN



# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(106)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(106, 134)

# training for rotation 
train_scene_index = np.random.choice(unlabeled_scene_index, int(np.ceil(0.95*len(unlabeled_scene_index))))

# test for rotation 
test_scene_index = unlabeled_scene_index[np.isin(unlabeled_scene_index, train_scene_index, invert=True)]


#transform = torchvision.transforms.ToTensor()

#transform=torchvision.transforms.Compose([torchvision.transforms.RandomCrop((200,200)),
#                                          torchvision.transforms.Resize((100,100)),
##                                          torchvision.transforms.ToTensor(),
 #                             #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
 #                            ])

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((200,200)),
                                          torchvision.transforms.ToTensor(),
#                              torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])

unlabeled_trainset = UnlabeledDataset(image_folder=image_folder, 
                                      scene_index=train_scene_index, 
                                      first_dim='sample', 
                                      transform=transform)
trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=16, shuffle=True, num_workers=2)

unlabeled_testset = UnlabeledDataset(image_folder=image_folder, 
                                      scene_index=test_scene_index, 
                                      first_dim='sample', 
                                      transform=transform)
testloader = torch.utils.data.DataLoader(unlabeled_testset, batch_size=16, shuffle=True, num_workers=2)

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


class CNN(nn.Module):
    def __init__(self, d=650, output_size=4):
        super().__init__()
        self.dconv_down1 = double_conv(3, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 48)
        self.dconv_down4 = double_conv(48, 64)        
        self.maxpool = nn.MaxPool2d(2)
        self.linear = nn.Linear(64*25*25,4)
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)


        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)  

        
        x = self.dconv_down4(x)
        #print(x.shape)
        x = self.linear(x.reshape(-1,64*25*25))
        return F.log_softmax(x, dim=1)

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]

# The dataset class for unlabeled data.
class UnlabeledRotationDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index, first_dim, transform):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data 
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, NUM_IMAGE_PER_SAMPLE, 3, H, W]
                'image' will return [batch_size, 3, H, W] and the index of the camera [0 - 5]
                    CAM_FRONT_LEFT: 0
                    CAM_FRONT: 1
                    CAM_FRONT_RIGHT: 2
                    CAM_BACK_LEFT: 3
                    CAM_BACK.jpeg: 4
                    CAM_BACK_RIGHT: 5
            transform (Transform): The function to process the image
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.transform = transform

        self.first_dim = 'image'

    def __len__(self):
        if self.first_dim == 'sample':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE
        elif self.first_dim == 'image':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE
    
    def __getitem__(self, index):
        if self.first_dim == 'sample':
            scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
            sample_id = index % NUM_SAMPLE_PER_SCENE
            sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}') 

            images = []
            labels = []
            for image_name in image_names:
                
                rot_class = np.random.randint(4)
                rot_angle = rot_class * 90
                
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                rot_image = image.rotate(rot_angle)
                labels.append(torch.from_numpy(np.asarray(rot_class)))
                images.append(self.transform(image))
            
            image_tensor = torch.stack(images)
            label_tensor = torch.stack(labels)
            
            return image_tensor, label_tensor
            
        elif self.first_dim == 'image':
            scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
            sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
            image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

            image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name) 

            rot_class = np.random.randint(4)
            rot_angle = rot_class * 90
            image = Image.open(image_path)
            rot_image = image.rotate(rot_angle)

            return self.transform(rot_image), torch.from_numpy(np.asarray(rot_class)), index % NUM_IMAGE_PER_SAMPLE
            

## Get individual image

unlabeled_trainset = UnlabeledDataset(image_folder=image_folder, 
                                      scene_index=unlabeled_scene_index, 
                                      first_dim='image', 
                                      transform=transform)
trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=2, shuffle=True, num_workers=2)

# Rotation Representation Learning

batch_size=16

unlabeled_trainset = UnlabeledRotationDataset(image_folder=image_folder, 
                                      scene_index=train_scene_index, 
                                      first_dim='image', 
                                      transform=transform)
trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=batch_size, shuffle=True, num_workers=0)

unlabeled_testset = UnlabeledRotationDataset(image_folder=image_folder, 
                                      scene_index=test_scene_index, 
                                      first_dim='image', 
                                      transform=transform)
testloader = torch.utils.data.DataLoader(unlabeled_testset, batch_size=batch_size, shuffle=True, num_workers=0)

sample, label, img_index = iter(trainloader).next()

model_cnn = CNN()
model_cnn = model_cnn.to(device)

accuracy_list = []
best_loss = 1000000

model_cnn = CNN()
model_cnn.to(device)
best_model = copy.deepcopy(model_cnn)

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
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
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
        torch.save(best_model.state_dict(), 'models/rotation_learning_model_unet.pth')
        
    return test_loss
        
for epoch in range(0, 25):
    train(epoch, model_cnn)
    test_loss = test(model_cnn)
    scheduler.step(test_loss)

