#!/usr/bin/python
'''
Contains the utilities used for
loading, initating and running up the networks
for all training, validation and evaluation.
'''
from cmath import nan
import torch
import torchvision
import torch.nn as nn
from torch.nn import init
import functools
from torchvision.ops import box_iou
from torch.optim import lr_scheduler
import numpy as np
import os
from Generators import *
from params import *

def write_log(log_values, model_name, log_dir="", log_type='loss', type_write='a'):
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    with open(log_dir+"/"+model_name+"_"+log_type+".txt", type_write) as f:
        f.write(','.join(log_values)+"\n")

def get_model_funct(model_name):
    if model_name == "G":
        return define_G
    elif model_name == "D":
        return define_D
    elif model_name == "R":
        return define_R

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

    if net_type == 'resnet_encoder':
        n_blocks    = opt.resnet_blocks
        net = ResnetEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif net_type == 'resnet_decoder':
        n_blocks    = opt.resnet_blocks
        net = ResnetDecoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, encoder_blocks=opt.encoder_res_blocks)
    elif net_type == 'resnet_classifier':
        n_blocks    = opt.resnet_blocks
        net = ResnetClassifier(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, encoder_blocks=opt.encoder_res_blocks)
    elif net_type == 'resnet_bounding':
        n_blocks = opt.resnet_blocks
        net = ResnetBoundingBox(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif net_type == 'aoa_classifier':
        net = AoAClassifier(norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net_type)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_R(opt, gpu_ids):
    net = None
    input_nc    = opt.input_nc
    output_nc   = opt.output_nc
    growth_rate = opt.ngf
    net_type    = opt.base_model
    norm        = opt.norm
    drop_rate   = opt.dropout_rate 
    init_type   = opt.init_type
    init_gain   = opt.init_gain
    depth       = opt.n_blocks
    norm_layer  = get_norm_layer(norm_type=norm)

    if net_type == 'DenseNet3':
        
        net = DenseNet3(depth, input_nc, growth_rate, dropRate=drop_rate)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net_type)
    return init_net(net, init_type, init_gain, gpu_ids)

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


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


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

def localization_error(output_predictions,input_labels,scale=[0.1,0.1], ap_aoas=None, ap_locs=None):
    """
    output_predictions: (N,1,H,W), model prediction 
    input_labels: (N,1,H,W), ground truth target
    scale: (optional): (1,2), scle along x,y axis for index to meter conversion
    ap_aoas: (optional): (N, 1)
    ap_locs: (optional) : (N, 2)
    """
    if opt_exp.label_type == 'Gaussian' :
        image_size = output_predictions.shape
        error = np.zeros(image_size[0])
        scale = np.asarray(scale)
        for i in range(image_size[0]):
            label_temp = input_labels[i,:,:,:].squeeze() # ground truth label
            pred_temp = output_predictions[i,:,:,:].squeeze() # model prediction
            label_index = np.asarray(np.unravel_index(np.argmax(label_temp), label_temp.shape))
            prediction_index = np.asarray(np.unravel_index(np.argmax(pred_temp),pred_temp.shape))
            error[i] = np.sqrt( np.sum( np.power(np.multiply( label_index-prediction_index, scale ), 2)) )

    elif opt_exp.label_type == 'XY':
        temp_input_labels = np.copy(input_labels)
        temp_output_preds = np.copy(output_predictions)

        temp_input_labels[:,0] *= (opt_exp.label_Xstop - opt_exp.label_Xstart)
        temp_input_labels[:,0] += opt_exp.label_Xstart
        temp_output_preds[:,0] *= (opt_exp.label_Xstop - opt_exp.label_Xstart)
        temp_output_preds[:,0] += opt_exp.label_Xstart
        temp_input_labels[:,1] *= (opt_exp.label_Ystop - opt_exp.label_Ystart)
        temp_input_labels[:,1] +=  opt_exp.label_Ystart
        temp_output_preds[:,1] *= (opt_exp.label_Ystop - opt_exp.label_Ystart)
        temp_output_preds[:,1] +=  opt_exp.label_Ystart
        image_size = temp_output_preds.shape
        error = np.zeros(image_size[0])

        for i in range(image_size[0]):
            label_temp = temp_input_labels[i,:] # ground truth label
            pred_temp = temp_output_preds[i,:] # model prediction          
            error[i] = np.sqrt( np.sum( np.power(np.multiply( label_temp-pred_temp, 1 ), 2)) )

    elif opt_exp.label_type == "BB" or opt_exp.label_type == "BB_fixed":
        temp_input_labels = np.copy(input_labels)
        temp_output_preds = np.copy(output_predictions)

        xScale = opt_exp.label_Xstop - opt_exp.label_Xstart
        yScale = opt_exp.label_Ystop - opt_exp.label_Ystart

        if opt_exp.label_type == "BB_fixed":
            box = np.tile([opt_exp.box_width, opt_exp.box_height], (temp_input_labels.shape[0], 1))
            temp_input_labels = np.concatenate((temp_input_labels, box), axis=1)
            temp_output_preds = np.concatenate((temp_output_preds, box), axis=1)

        temp_input_labels = np.add(np.multiply(temp_input_labels, [xScale, yScale, xScale, yScale]), 
                                                                    [opt_exp.label_Xstart + opt_exp.box_width, opt_exp.label_Ystart - opt_exp.box_height, 0, 0])

        temp_output_preds = np.add(np.multiply(temp_output_preds, [xScale, yScale, xScale, yScale]), 
                                                                    [opt_exp.label_Xstart + opt_exp.box_width, opt_exp.label_Ystart - opt_exp.box_height, 0, 0])

        error = np.zeros(temp_output_preds.shape[0]) # error is (N) size

        for i in range(temp_input_labels.shape[0]):
            input_label = np.array([temp_input_labels[i, 0] + temp_input_labels[i, 2], temp_input_labels[i, 1] + temp_input_labels[i, 3]])
            output_pred = np.array([temp_output_preds[i, 0] + temp_output_preds[i, 2], temp_input_labels[i, 1] + temp_input_labels[i, 3]])
            error[i] = np.sqrt(np.sum(np.power(np.multiply(input_label - output_pred, 1 ), 2)))

    elif opt_exp.label_type == "AoA":

        error = np.zeros(input_labels.shape[0])

        for i in range(input_labels.shape[0]):
            outp = output_predictions[i, :]
            inp = input_labels[i, :]
            if opt_decoder.loss_type == "Cross_Entropy":
                error[i] = np.absolute(np.argmax(outp)-inp)
            elif opt_decoder.loss_type == "KL_Div":
                error[i] = np.absolute(np.argmax(outp)-np.argmax(inp))
    #     temp_input_labels = np.copy(input_labels)
    #     temp_output_preds = np.copy(output_predictions)

    #     '''
    # A = [sin(thetas), -cos(thetas)];
    # b = X0.*sin(thetas)-Y0.*cos(thetas);
    # P_intersect = ((A'*W*A)\A'*W*b).';

    #     '''
    #     n_points, n_ap = output_predictions.shape
    #     pred_scale = np.array([np.pi/2]).repeat(n_ap)

    #     output_theta = (temp_output_preds * pred_scale) - np.squeeze(ap_aoas) 

    #     output_xy = np.zeros((n_points, 2)) 

    #     for i in range(n_points):
    #         A = np.transpose(np.array([np.sin(output_theta[i, :]), -np.cos(output_theta[i, :])]))
    #         B = np.array([ap_locs[i, :, 0] * np.sin(output_theta[i, :]) - ap_locs[i, :, 1] * np.cos(output_theta[i, :])]).transpose()
    #         W = np.eye(n_ap)
    #         output_xy[i, :] = np.transpose(np.linalg.lstsq(A.transpose() @ W @ A, A.transpose())[0] @ W @ B)

    #     error = np.zeros(input_labels.shape[0])
    #     for i in range(n_points):
    #         error[i] = np.sqrt(np.sum(np.power(temp_input_labels[i, :] - output_xy[i, :], 2)))
    return error

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def RGB2Gray(img):
    return 0.2125*img[:,:,0] + 0.7154*img[:,:,1] + 0.0721*img[:,:,2]