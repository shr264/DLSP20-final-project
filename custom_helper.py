from skimage import draw
from sklearn.metrics import confusion_matrix
from helper import collate_fn, draw_box
from data_helper import UnlabeledDataset, LabeledDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import time
import sys
import random
import psutil

import math
import json
import copy
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
image_folder = '../../DLSP20Dataset/data'
annotation_csv = '../../DLSP20Dataset/data/annotation.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.is_available()


# function to count number of parameters
def get_n_params(model):
    np = 0
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


def arrange_box(x1, y1):
    box = np.array(list(zip(x1, y1)))
    box = order_points(box)
    return box


def iou(box1, box2):
    from shapely.geometry import Polygon
    a = Polygon(torch.t(box1)).convex_hull
    b = Polygon(torch.t(box2)).convex_hull

    return a.intersection(b).area / a.union(b).area

# def iou(xy1,xy2):
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
    if print_it:
        print(prior_overlap)
#     pdb.set_trace()
    gt_overlap, gt_idx = overlaps.max(0)
    gt_overlap[prior_idx] = 1.99
    for i, o in enumerate(prior_idx):
        gt_idx[o] = i
    return gt_overlap, gt_idx


def calculate_overlap(target_bb, predicted_bb):
    overlaps = torch.zeros(target_bb.size(0), predicted_bb.size(0))

    for j in range(overlaps.shape[0]):
        for k in range(overlaps.shape[1]):
            overlaps[j][k] = iou(target_bb[j], predicted_bb[k])

    return overlaps


def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, shape)
    mask = torch.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def convert_to_binary_mask(corners, shape=(800, 800)):
    point_squence = torch.stack(
        [corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])
    x, y = point_squence.T[0].detach() * 10 + 400, - \
        point_squence.T[1].detach() * 10 + 400
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
    if len(pred.unique()) > (nb_classes+1):
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

    print('Threat Score: {}'.format(
        (1.0*conf_matrix[1, 1])/(conf_matrix[1, 1]+conf_matrix[1, 0]+conf_matrix[0, 1])))

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
        # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
        TN[c] = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
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
