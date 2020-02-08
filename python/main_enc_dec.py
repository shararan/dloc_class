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

from utils import *
from modelADT import ModelADT
from Generators import *
# from LocationNetworks import *
from data_loader import *
from joint_model import *
from params import *


print('Entered Main')

def train(model, loaded_data, loaded_test_data, input_index=0, output_index=1):
    total_steps = 0
    print('Training called')
    # generated_outputs = []
    stopping_count=0
    for epoch in range(model.opt.starting_epoch_count+1, model.opt.n_epochs+1): # opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        temp_generated_outputs = []
        epoch_loss = 0
        epoch_regloss = 0
        error= []
        for i, data in enumerate(loaded_data):
            iter_start_time = time.time()
            total_steps += model.opt.batch_size

            model.set_data(data[input_index], data[output_index], convert=True)
            model.optimize_parameters()

            gen_outputs = model.output
            error.extend(localization_error(gen_outputs.data.cpu().numpy(),data[output_index].cpu().numpy(),scale=0.1))
            # temp_generated_outputs.extend(gen_outputs.data.cpu().numpy())
            
            write_log([str(model.loss.item())], model.model_name, log_dir=model.opt.log_dir, log_type='loss')
            
            if total_steps % model.opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            epoch_loss += model.loss.item()
            epoch_regloss += model.reg_loss.item()
        median_error_tr = np.median(error)
        epoch_loss /= i
        epoch_regloss /= i
        write_log([str(epoch_loss)], model.model_name, log_dir=model.opt.log_dir, log_type='epoch_loss')
        write_log([str(epoch_regloss)], model.model_name, log_dir=model.opt.log_dir, log_type='epoch_regloss')
        write_log([str(median_error_tr)], model.model_name, log_dir=model.opt.log_dir, log_type='train_median_error')
        if (epoch==1):
            min_eval_loss,median_error = eval(model, loaded_test_data, input_index=input_index, output_index=output_index, save_output=True)
        else:
            new_eval_loss,new_med_error = eval(model, loaded_test_data, input_index=input_index, output_index=output_index, save_output=True)
            if (median_error>=new_med_error):
                stopping_count = stopping_count+1
                median_error = new_med_error



        # generated_outputs = temp_generator_outputs
        if epoch % model.opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
            if (stopping_count==2):
                print('Saving best model at %d epoch' %(epoch))
                model.save_networks('best')
                stopping_count=0

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, model.opt.niter + model.opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

def eval(model, loaded_data, input_index=0, output_index=1, max_data_to_run = -1, save_output=True):
    print('Evaluation Called')
    model.eval()
    generated_outputs = []
    error= []
    total_loss = 0
    for i, data in enumerate(loaded_data):
            model.set_data(data[input_index], data[output_index], convert=True)
            model.test()
            gen_outputs = model.output
            generated_outputs.extend(gen_outputs.data.cpu().numpy())
            error.extend(localization_error(gen_outputs.data.cpu().numpy(),data[output_index].cpu().numpy(),scale=0.1))
            total_loss += model.loss.item()
    print("saving")
    total_loss /= i
    median_error = np.median(error)
    write_log([str(total_loss)], model.model_name, log_dir=model.opt.log_dir, log_type='test_loss')
    write_log([str(median_error)], model.model_name, log_dir=model.opt.log_dir, log_type='test_median_error')
    if not os.path.exists(model.results_save_dir):
        os.makedirs(model.results_save_dir, exist_ok=True)
    if save_output:
        scipy.io.savemat(model.results_save_dir+"/"+model.model_name+".mat", mdict={"outputs":generated_outputs})

    print("done")

    return total_loss, median_error

if "data" in opt_exp and opt_exp.data == "rw_to_rw_atk_noref":
    trainpath = ['/media/user1/easystore/datasets/quantenna/features/dataset_non_fov_train_July18.mat',
                '/media/user1/easystore/datasets/quantenna/features/dataset_fov_train_July18.mat']
    testpath = ['/media/user1/easystore/datasets/quantenna/features/dataset_non_fov_test_July18.mat',
                '/media/user1/easystore/datasets/quantenna/features/dataset_fov_test_July18.mat']
    print('Real World to Real World experiments started')


if opt_exp.phase != "train" or opt_exp.isTrainGen:
    gen_model = ModelADT()
    gen_model.initialize(opt_gen)
    gen_model.setup(opt_gen)
    print('Model has been set up')




#### main training/testing code

if "train" in opt_exp.phase:
    print('Training Phase started')

    B_train,A_train,labels_train = load_data(trainpath[0])
    for i in range(len(trainpath)-1):
        f,f1,l = load_data(trainpath[i+1])
        B_train = torch.cat((B_train, f), 0)
        A_train = torch.cat((A_train, f1), 0)
        labels_train = torch.cat((labels_train, l), 0)

    labels_train = torch.unsqueeze(labels_train, 1)
    if 'db' in opt_exp.phase:
        B_train  = np.log10(np.add(B_train,1))
        A_train  = np.log10(np.add(A_train,1))
    train_data = torch.utils.data.TensorDataset(B_train, A_train, labels_train)

    print('Training Data Loaded')

    B_test,A_test,labels_test = load_data(testpath[0])
    for i in range(len(testpath)-1):
        f,f1,l = load_data(testpath[i+1])
        B_test = torch.cat((B_test, f), 0)
        A_test = torch.cat((A_test, f1), 0)
        labels_test = torch.cat((labels_test, l), 0)

    if 'db' in opt_exp.phase:
        B_test  = np.log10(np.add(B_test,1))
        A_test  = np.log10(np.add(A_test,1))

    labels_test = torch.unsqueeze(labels_test, 1)

    test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)
    test_loader =torch.utils.data.DataLoader(test_data, batch_size=opt_exp.batch_size, shuffle=False)

    print('Test Data Loaded')

    if opt_exp.isTrainGen:
        train_loader =torch.utils.data.DataLoader(train_data, batch_size=opt_gen.batch_size, shuffle=True)
        dataset_size = len(train_loader)
        print('#training images = %d' % dataset_size)
        train(gen_model, train_loader, test_loader, input_index=1, output_index=2)



elif "test" in opt_exp.phase:
    B_test,A_test,labels_test = load_data(testpath[0])
    for i in range(len(testpath)-1):
        f,f1,l = load_data(testpath[i+1])
        B_test = torch.cat((B_test, f), 0)
        A_test = torch.cat((A_test, f1), 0)
        labels_test = torch.cat((labels_test, l), 0)

    labels_test = torch.unsqueeze(labels_test, 1)

    print('Test Data Loaded')

    gen_model.load_networks(gen_model.opt.starting_epoch_count)
    test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)

    test_loader =torch.utils.data.DataLoader(test_data, batch_size=opt_gen.batch_size, shuffle=False)
    dataset_size = len(test_loader)
    print('#training images = %d' % dataset_size)
    eval(gen_model, test_loader, input_index=1, output_index=2)