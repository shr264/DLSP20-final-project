import os
from PIL import Image
import time
import sys
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
from scipy.optimize import linear_sum_assignment

random.seed(0)
np.random.seed(0)
torch.manual_seed(0);

# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file
#image_folder = '../../DLSP20Dataset/data'
#annotation_csv = '../../DLSP20Dataset/data/annotation.csv'


# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file
image_folder = '../../data'
annotation_csv = '../../data/annotation.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#device = 'cpu'

#device = "cpu"

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

    return np.array([tl, tr, br, bl]).astype(float)

def arrange_box(x1,y1):
    box=np.array(list(zip(x1,y1)))
    box=order_points(box)
    return box

def iou(box1, box2):
    from shapely.geometry import Polygon
    try: 
        a = Polygon(torch.t(box1)).convex_hull
        b = Polygon(torch.t(box2)).convex_hull

        return a.intersection(b).area / a.union(b).area
    except:
        return 0

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

def IOU(bbox1, bbox2):
    '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]  # TODO: Check if its more performant if tensor elements are accessed directly below.
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    w_I = max(w_I, 0)  # set w_I and h_I zero if there is no intersection
    h_I = max(h_I, 0)
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U

def dist(bbox1, bbox2):
    return torch.sqrt(torch.sum(torch.square(bbox1[:2] - bbox2[:2])))

# Dataset

# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(106)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(106, 134)

train_scene_index = np.random.choice(labeled_scene_index, int(np.ceil(0.9*len(labeled_scene_index))))

test_scene_index = labeled_scene_index[np.isin(labeled_scene_index, train_scene_index,invert=True)]
#test_scene_index = train_scene_index


transform=torchvision.transforms.Compose([torchvision.transforms.Resize((100,100)),
                                          torchvision.transforms.ToTensor(),
                              #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])


# Labeled dataset

batch_size = 1

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

def mapX(x):
    return (x+40)/80

def mapY(x):
    return (x+40)/80

def boxToHWC(boxes1):
    boxes1_max_x = mapX(boxes1[:, 0].max(dim=1)[0])
    boxes1_min_x = mapX(boxes1[:, 0].min(dim=1)[0])
    boxes1_max_y = mapY(boxes1[:, 1].max(dim=1)[0])
    boxes1_min_y = mapY(boxes1[:, 1].min(dim=1)[0])
    
    x = (boxes1_max_x + boxes1_min_x)/2
    y = (boxes1_max_y + boxes1_min_y)/2
    h = (boxes1_max_y - boxes1_min_y)
    w = (boxes1_max_x - boxes1_min_x)
    
    return torch.stack((x,y,h,w)).T


def imgToHWC(img):
    height, width = img.shape[:2]     
    x1 = int(row['XMin'] * width)     
    x2 = int(row['XMax'] * width)     
    y1 = int(row['YMin'] * height)     
    y2 = int(row['YMax'] * height)
    return (x1+x2)/2, (y1+y2)/2, height, width


def transformTargetstoHWC(targets):
    bounding_box = targets["bounding_box"]
    max, ind = torch.max(bounding_box, dim=3)
    min, ind = torch.min(bounding_box, dim=3)
    c = (400 + ((max + min)/2) * 10) / 1.92
    w = (max - min) * 10 / 1.92
    target = torch.cat((c, w), dim=2)
    category = targets["category"]
    targets = torch.cat((target, category.unsqueeze(2).double()), dim=2).float()
    target_lengths = torch.tensor(targets.numpy().shape[:2])
    return targets, target_lengths


# The dataset class for labeled data.
class LabeledSSDDataset(torch.utils.data.Dataset):    
    def __init__(self, image_folder, annotation_file, scene_index, transform, extra_info=True):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data 
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """
        
        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info
    
    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}') 

        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            images.append(self.transform(image))
        image_tensor = torch.stack(images)

        data_entries = self.annotation_dataframe[(self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']].to_numpy()
        categories = data_entries.category_id.to_numpy()
        
        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        #road_image = convert_map_to_road_map(ego_image)
        
        target = {}
        target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 4)
        target['category'] = torch.as_tensor(categories).view(1,-1)
        
        target['chw'] = boxToHWC(target['bounding_box']).unsqueeze(0).double()


        return image_tensor, target['chw'], target['bounding_box'], target['category']
        

# The labeled dataset can only be retrieved by sample.
# And all the returned data are tuple of tensors, since bounding boxes may have different size
# You can choose whether the loader returns the extra_info. It is optional. You don't have to use it.
labeled_trainset = LabeledSSDDataset(image_folder=image_folder,
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

labeled_testset = LabeledSSDDataset(image_folder=image_folder,
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
        self.linear1 = nn.Linear(6*64*12*12,4*88) ## 88 boxes
        self.linear2 = nn.Linear(6*64*12*12,88) ## 88 boxes
        
    def forward_once(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)


        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)  

        
        x = self.dconv_down4(x)
        return x
    
    def forward(self,x):
        x = [self.forward_once(y) for y in x]
        x = torch.cat(x,axis=1)
        x1 = self.linear1(x.reshape(-1,6*64*12*12))
        x2 = self.linear2(x.reshape(-1,6*64*12*12))
        return (torch.sigmoid(x1),torch.sigmoid(x2))

model = CNN()

mse = nn.MSELoss()

learning_rate = 1e-2

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                       mode='min', 
                                                       factor=0.1, 
                                                       patience=3,
                                                       verbose=True)

best_loss = 1000000000000000
num_epochs = 50
threshold = 0.5
num_objects = 88
best_iou = 0

for epoch in range(0, num_epochs):
    model.train()
    train_loss = {}
    train_loss['boxes'] = 0
    train_loss['probs'] = 0
    train_loss['total'] = 0
    for (x, bb_true, bb_orig, class_true) in trainloader:
        exp = torch.stack(bb_true)
        exp = exp.reshape(1, -1, 4)
        pred, probs = model(torch.stack(x).reshape(6,-1,3,100,100))
        pred = pred.reshape(1, -1, 4)
        probs = probs.reshape(1, -1, 88)
        num_true = exp.size(1)
        pred = pred.reshape(num_objects, -1)
        exp = exp.reshape(num_true, -1)      
        pred_bboxes = pred[:, :4]
        exp_bboxes = exp[:, :4]
        # TODO: Try flipping array and see if results differ.
        ious = np.zeros((num_true, num_objects))
        for i, exp_bbox in enumerate(exp_bboxes):
            for j, pred_bbox in enumerate(pred_bboxes):
                ious[i, j] = IOU(exp_bbox, pred_bbox)

        exp_idx, pred_idx = linear_sum_assignment(-ious)
    

        loss1 = mse(exp[exp_idx],pred[pred_idx])
        train_loss['boxes'] += loss1.item()
            
        label = torch.zeros(88)
        label[pred_idx] = 1
        loss2 = mse(probs.reshape(-1), label.reshape(-1))
        train_loss['probs'] = loss2.item()
        
        loss = loss1 + loss2
        
        train_loss['total'] = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('====> Epoch: {} Average loss (boxes): {} / {}'.format(
    epoch,train_loss['boxes'],len(trainloader.dataset)))
    print('====> Epoch: {} Average loss (probs): {} / {}'.format(
    epoch,train_loss['probs'],len(trainloader.dataset)))
    print('====> Epoch: {} Average loss (total): {} / {}'.format(
    epoch,train_loss['total'],len(trainloader.dataset)))
    
    with torch.no_grad():
        model.eval()
        test_loss = {}
        test_loss['boxes'] = 0
        test_loss['probs'] = 0
        test_loss['total'] = 0
        iou_list = []
        for (x, bb_true, bb_orig, class_true) in testloader:
            exp = torch.stack(bb_true)
            exp = exp.reshape(1, -1, 4)
            pred, probs = model(torch.stack(x).reshape(6,-1,3,100,100))
            pred = pred.reshape(1, -1, 4)
            probs = probs.reshape(1, -1, 88)
            num_true = exp.size(1)
            pred = pred.reshape(num_objects, -1)
            exp = exp.reshape(num_true, -1)      
            pred_bboxes = pred[:, :4]
            exp_bboxes = exp[:, :4]

            pred_idx = (probs>threshold).squeeze()
            predicted_boxes = pred[pred_idx]
            predicted_num_objects = torch.sum(probs>threshold)
            ious = np.zeros((num_true, predicted_num_objects))
            for i, exp_bbox in enumerate(exp_bboxes):
                for j, pred_bbox in enumerate(predicted_boxes):
                    ious[i, j] = IOU(exp_bbox, pred_bbox)
                    
            iou_list.append(np.mean(ious))
        mean_iou = sum(iou_list) / len(iou_list)
        print('====> Epoch: {} Average IOU on Test Set: {}'.format(epoch, mean_iou))
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), 'models/best_simple_object_detector.pth')
        scheduler.step(mean_iou)
            
torch.save(model.state_dict(), 'models/final_simple_object_detector.pth')