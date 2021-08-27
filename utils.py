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
# from scipy.misc import imresize
import json

from Generators import *
# from LocationNetworks import *

def write_log(log_values, model_name, log_dir="", log_type='loss', type_write='a'):
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    with open(log_dir+"/"+model_name+"_"+log_type+".txt", type_write) as f:
        f.write(','.join(log_values)+"\n")

def get_model_funct(model_name):
    if model_name == "G":
        return define_G
    # else:
    #     return define_L

# def define_L(opt, gpu_ids):
#     return LocationNetwork()

def define_G(opt, gpu_ids):
    net = None
    input_nc    = opt.input_nc
    output_nc   = opt.output_nc
    ngf         = opt.ngf
    net_type    = opt.base_model
    norm        = opt.norm
    use_dropout = opt.no_dropout 
    init_type   = opt.init_type
    init_gain   = opt.init_gain

    norm_layer = get_norm_layer(norm_type=norm)

    if net_type == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif net_type == 'resnet_nblocks':
        n_blocks    = opt.resnet_blocks
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif net_type == 'resnet2_nblocks':
        n_blocks    = opt.resnet_blocks
        net = ResnetGenerator2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif net_type == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif net_type == 'resnet_encoder':
        n_blocks    = opt.resnet_blocks
        net = ResnetEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif net_type == 'resnet_decoder':
        n_blocks    = opt.resnet_blocks
        net = ResnetDecoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, encoder_blocks=opt.encoder_res_blocks)
    elif net_type == 'resnet_encoder_cat':
        n_blocks    = opt.resnet_blocks
        net = ResnetEncoderCat(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif net_type == 'resnet_decoder_cat':
        n_blocks    = opt.resnet_blocks
        net = ResnetDecoderCat(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, encoder_blocks=opt.encoder_res_blocks)        
    elif net_type == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif net_type == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net_type)
    return init_net(net, init_type, init_gain, gpu_ids)

# In[6]:


def get_scheduler(optimizer, opt):
    if opt.starting_epoch_count=='best' and opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            print("lambda update %s, %s, %s", (epoch, 0))
            lr_l = 1.0 - max(0, epoch + 1 - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.starting_epoch_count!='best' and opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            print("lambda update %s, %s, %s", (epoch, opt.starting_epoch_count))
            lr_l = 1.0 - max(0, epoch + 1 + opt.starting_epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.starting_epoch_count!='best' and opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.9)
    elif opt.starting_epoch_count!='best' and opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.starting_epoch_count!='best' and opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# In[7]:


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# In[8]:


def init_weights(net, init_type='normal', gain=1):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


# In[9]:


def init_net(net, init_type='normal', init_gain=1, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        device_ = torch.device('cuda:{}'.format(gpu_ids[0]))
#         net.to(d)
        gpu_ids_int = list(map(int,gpu_ids))
        net = torch.nn.DataParallel(net, gpu_ids_int)
        net.to(device_)
    init_weights(net, init_type, gain=init_gain)
    return net

def localization_error(output_predictions,input_labels,scale=0.1):
    outputs = np.squeeze(output_predictions)
    inputs = np.squeeze(input_labels)
    image_size = outputs.shape
    error = np.zeros((image_size[0]))
    if(image_size[0]==161):
        label_temp = inputs
        pred_temp = outputs
        label_index = np.asarray(np.unravel_index(np.argmax(label_temp), label_temp.shape))
        prediction_index = np.asarray(np.unravel_index(np.argmax(pred_temp),pred_temp.shape))
        error[0] = np.sqrt( np.sum( np.power(np.multiply( label_index-prediction_index, scale ), 2)) )
        return error
    else:
        for i in range(image_size[0]):

            label_temp = inputs[i,:,:].squeeze()
            pred_temp = outputs[i,:,:].squeeze()
            label_index = np.asarray(np.unravel_index(np.argmax(label_temp), label_temp.shape))
            prediction_index = np.asarray(np.unravel_index(np.argmax(pred_temp),pred_temp.shape))
            error[i] = np.sqrt( np.sum( np.power(np.multiply( label_index-prediction_index, scale ), 2)) )
        return error

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def RGB2Gray(img):
    return 0.2125*img[:,:,0] + 0.7154*img[:,:,1] + 0.0721*img[:,:,2]
