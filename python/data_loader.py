#!/usr/bin/python

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
# from util.image_pool import ImagePool
from collections import OrderedDict
import time
# from options.train_options import TrainOptions
from collections import defaultdict
import h5py
import scipy.io
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torchvision
import os
from easydict import EasyDict as edict
import random
import matplotlib.pyplot as plt
import sys
import ntpath
import time
from scipy.misc import imresize
import json

def load_data(filename):
# def load_data(filename,neglect_2=False):
    print('Loading '+filename)
    arrays = {}
    f = h5py.File(filename,'r')
    features_wo_offset = torch.tensor(np.transpose(np.array(f.get('features_wo_offset'), dtype=np.float32)), dtype=torch.float32)
    features_w_offset = torch.tensor(np.transpose(np.array(f.get('features_w_offset'), dtype=np.float32)), dtype=torch.float32)
    labels_gaussian_2d = torch.tensor(np.transpose(np.array(f.get('labels_gaussian_2d'), dtype=np.float32)), dtype=torch.float32)
    
    	
    return features_wo_offset,features_w_offset, labels_gaussian_2d


