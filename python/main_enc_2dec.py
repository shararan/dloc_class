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
import hdf5storage

from utils import *
from modelADT import ModelADT
from Generators import *
from LocationNetworks import *
from data_loader import *
from joint_model import *
from params import *

def train(model, loaded_data, loaded_test_data, input_index=1, output_index=2, offset_output_index=0):
    total_steps = 0
    print('Training called')
    # generator_outputs = []
    # location_outputs = []
    stopping_count = 0
    for epoch in range(model.opt.starting_epoch_count+1, model.opt.n_epochs+1): # opt.niter + opt.niter_decay + 1):
        # epoch_loss_enc = 0
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_offset_loss = 0
        # last_weights = list(joint_model.encoder.net.parameters())[6].data.cpu().numpy()
        error =[]
        for i, data in enumerate(train_loader):
            total_steps += model.opt.batch_size

            model.set_input(data[input_index], data[output_index], data[offset_output_index], shuffle_channel=False)
            model.optimize_parameters()
            enc_outputs = model.encoder.output
            dec_outputs = model.decoder.output
            off_dec_outputs = model.offset_decoder.output
            error.extend(localization_error(dec_outputs.data.cpu().numpy(),data[output_index].cpu().numpy(),scale=0.1))
            # generator_outputs.extend(gen_outputs.data.cpu().numpy())
            # location_outputs.extend(loc_outputs.data.cpu().numpy())
            # new_weights = list(joint_model.encoder.net.parameters())[6].data.cpu().numpy()
            # last_weights = new_weights

            write_log([str(model.decoder.loss.item())], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='loss')
            write_log([str(model.offset_decoder.loss.item())], model.offset_decoder.model_name, log_dir=model.offset_decoder.opt.log_dir, log_type='offset_loss')
            if total_steps % model.decoder.opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            epoch_loss += model.decoder.loss.item()
            epoch_offset_loss += model.offset_decoder.loss.item()

        median_error_tr = np.median(error)
        epoch_loss /= i
        epoch_offset_loss /= i
        write_log([str(epoch_loss)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='epoch_decoder_loss')
        write_log([str(epoch_offset_loss)], model.offset_decoder.model_name, log_dir=model.offset_decoder.opt.log_dir, log_type='epoch_offset_decoder_loss')
        write_log([str(median_error_tr)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='train_median_error')
        if (epoch==1):
            min_eval_loss, median_error = test(model, loaded_test_data, input_index=input_index, output_index=output_index, offset_output_index=offset_output_index, save_output=True)
        else:
            new_eval_loss, new_med_error = test(model, loaded_test_data, input_index=input_index, output_index=output_index, offset_output_index=offset_output_index, save_output=True)
            if (median_error>=new_med_error):
                stopping_count = stopping_count+1
                median_error = new_med_error

        # generated_outputs = temp_generator_outputs
        if epoch % model.encoder.opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
            if (stopping_count==2):
                print('Saving best model at %d epoch' %(epoch))
                model.save_networks('best')
                stopping_count=0

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, model.decoder.opt.niter + model.decoder.opt.niter_decay, time.time() - epoch_start_time))
        model.decoder.update_learning_rate()
        model.encoder.update_learning_rate()
        model.offset_decoder.update_learning_rate()


def test(model, loaded_data, input_index=1, output_index=2,  offset_output_index=0, max_data_to_run = -1, save_output=True):
    print('Evaluation Called')
    model.eval()
    generated_outputs = []
    offset_outputs = []
    total_loss = 0
    total_offset_loss = 0
    error =[]
    for i, data in enumerate(loaded_data):
            model.set_input(data[input_index], data[output_index], data[offset_output_index],convert_enc=True, shuffle_channel=False)
            model.test()
            gen_outputs = model.decoder.output
            off_outputs = model.offset_decoder.output
            generated_outputs.extend(gen_outputs.data.cpu().numpy())
            offset_outputs.extend(off_outputs.data.cpu().numpy())
            error.extend(localization_error(gen_outputs.data.cpu().numpy(),data[output_index].cpu().numpy(),scale=0.1))
            total_loss += model.decoder.loss.item()
            total_offset_loss += model.offset_decoder.loss.item()
    print("saving")
    total_loss /= i
    total_offset_loss /= i
    median_error = np.median(error)

    write_log([str(median_error)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_median_error')
    write_log([str(total_loss)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_loss')
    write_log([str(total_offset_loss)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_offset_loss')
    if not os.path.exists(model.decoder.results_save_dir):
        os.makedirs(model.decoder.results_save_dir, exist_ok=True)
    if save_output:
        hdf5storage.savemat(model.decoder.results_save_dir+"/"+model.decoder.model_name+".mat",mdict={"outputs":generated_outputs,"wo_outputs":offset_outputs}, appendmat=True, format='7.3',truncate_existing=True)
    

    print("done")

    return total_loss, median_error

if "data" in opt_exp and opt_exp.data == "rw_to_rw":
    trainpath = ['/media/user1/easystore/datasets/quantenna/features/dataset_edit_jacobs_July28.mat',
                '/media/user1/easystore/datasets/quantenna/features/dataset_non_fov_train_jacobs_July28_2.mat',
                '/media/user1/easystore/datasets/quantenna/features/dataset_fov_train_jacobs_July28_2.mat']
    testpath = ['/media/user1/easystore/datasets/quantenna/features/dataset_fov_test_jacobs_July28_2.mat',
                '/media/user1/easystore/datasets/quantenna/features/dataset_non_fov_test_jacobs_July28_2.mat']
    print('Real World to Real World experiments started')

elif "data" in opt_exp and opt_exp.data == "data_segment":
    trainpath = ['/media/user1/easystore/datasets/quantenna/features/dataset_test_jacobs_July28.mat',
                '/media/user1/easystore/datasets/quantenna/features/dataset_test_jacobs_July28_2.mat']
    testpath = ['/media/user1/easystore/datasets/quantenna/features/dataset_train_jacobs_July28.mat',
                '/media/user1/easystore/datasets/quantenna/features/dataset_train_jacobs_July28_2.mat']
    print('non-FOV to non-FOV experiments started')

elif "data" in opt_exp and opt_exp.data == "1st_environment":
    testpath = ['/media/user1/easystore/datasets/quantenna/features/dataset_test_jacobs_Aug16_1.mat']
    print('Across environments testing for env1 started')

elif "data" in opt_exp and opt_exp.data == "2nd_environment":
    testpath = ['/media/user1/easystore/datasets/quantenna/features/dataset_train_jacobs_Aug16_2.mat']
    print('Across environments testing for env12started')

elif "data" in opt_exp and opt_exp.data == "3rd_environment":
    testpath = ['/media/user1/easystore/datasets/quantenna/features/dataset_train_jacobs_Aug16_3.mat']
    print('Across environments testing for env3 started')

elif "data" in opt_exp and opt_exp.data == "4th_environment":
    trainpath = ['/media/user1/easystore/datasets/quantenna/features/dataset_train_jacobs_Aug16_4_ref.mat']
    testpath = ['/media/user1/easystore/datasets/quantenna/features/dataset_test_jacobs_Aug16_4_ref.mat']
    print('Adding a reflector data for env4 started')

elif "data" in opt_exp and opt_exp.data == "rw_to_rw_env":
    trainpath = ['/media/user1/easystore/datasets/quantenna/features/dataset_edit_jacobs_July28.mat',
                '/media/user1/easystore/datasets/quantenna/features/dataset_non_fov_train_jacobs_July28_2.mat',
                '/media/user1/easystore/datasets/quantenna/features/dataset_fov_train_jacobs_July28_2.mat',
                '/media/user1/easystore/datasets/quantenna/features/dataset_train_jacobs_Aug16_1.mat']
    testpath = ['/media/user1/easystore/datasets/quantenna/features/dataset_train_jacobs_Aug16_3.mat',]
    print('Real World to Real World experiments started')       


# # if opt_exp.phase != "train" or opt_exp.isTrainGen:
# if opt_exp.isTrainGen:
#     enc_model = ModelADT()
#     enc_model.initialize(opt_encoder)
#     enc_model.setup(opt_encoder)

# # if opt_exp.phase != "train" or opt_exp.isTrainLoc:
# if opt_exp.isTrainLoc:
#     dec_model = ModelADT()
#     dec_model.initialize(opt_decoder)
#     dec_model.setup(opt_decoder)

# if opt_exp.phase!="train":
enc_model = ModelADT()
enc_model.initialize(opt_encoder)
enc_model.setup(opt_encoder)

dec_model = ModelADT()
dec_model.initialize(opt_decoder)
dec_model.setup(opt_decoder)

offset_dec_model = ModelADT()
offset_dec_model.initialize(opt_offset_decoder)
offset_dec_model.setup(opt_offset_decoder)
print('Making the joint_model')
joint_model = Enc_2Dec_Network()
joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, frozen_dec = opt_exp.isFrozen, gpu_ids = opt_exp.gpu_ids)



#### main training/testing code



if "rw_train" in opt_exp.phase:

    B_train,A_train,labels_train = load_data(trainpath[0])
    for i in range(len(trainpath)-1):
        f,f1,l = load_data(trainpath[i+1])
        B_train = torch.cat((B_train, f), 0)
        A_train = torch.cat((A_train, f1), 0)
        labels_train = torch.cat((labels_train, l), 0)
    labels_train = torch.unsqueeze(labels_train, 1)
    train_data = torch.utils.data.TensorDataset(B_train, A_train, labels_train)
    train_loader =torch.utils.data.DataLoader(train_data, batch_size=opt_exp.batch_size, shuffle=True)
    print(A_train.shape)
    print(B_train.shape)
    print(labels_train.shape)
    dataset_size = len(train_loader)
    print('#training images = %d' % dataset_size)


    B_test,A_test,labels_test = load_data(testpath[0])
    for i in range(len(testpath)-1):
        f,f1,l = load_data(testpath[i+1])
        B_test = torch.cat((B_test, f), 0)
        A_test = torch.cat((A_test, f1), 0)
        labels_test = torch.cat((labels_test, l), 0)

    labels_test = torch.unsqueeze(labels_test, 1)

    test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)
    test_loader =torch.utils.data.DataLoader(test_data, batch_size=opt_exp.batch_size, shuffle=False)
    print(A_test.shape)
    print(B_test.shape)
    print(labels_test.shape)
    dataset_size = len(test_loader)
    print('#testing images = %d' % dataset_size)

    print('Test Data Loaded')

    if opt_exp.isFrozen:
        enc_model.load_networks(opt_encoder.starting_epoch_count)
        dec_model.load_networks(opt_decoder.starting_epoch_count)
        offset_dec_model.load_networks(opt_offset_decoder.starting_epoch_count)

    if opt_exp.isTrain:
        train(joint_model, train_loader, test_loader, input_index=1, output_index=2, offset_output_index=0)

elif "rw_test" in opt_exp.phase:

    B_test,A_test,labels_test = load_data(testpath[0])
    for i in range(len(testpath)-1):
        f,f1,l = load_data(testpath[i+1])
        B_test = torch.cat((B_test, f), 0)
        A_test = torch.cat((A_test, f1), 0)
        labels_test = torch.cat((labels_test, l), 0)

    labels_test = torch.unsqueeze(labels_test, 1)

    test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)
    test_loader =torch.utils.data.DataLoader(test_data, batch_size=opt_exp.batch_size, shuffle=False)

    dataset_size = len(test_loader)
    print('#training images = %d' % dataset_size)

    enc_model.load_networks(enc_model.opt.starting_epoch_count)
    dec_model.load_networks(dec_model.opt.starting_epoch_count)
    offset_dec_model.load_networks(opt_offset_decoder.starting_epoch_count)
    joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, frozen_dec = opt_exp.isFrozen, gpu_ids = opt_exp.gpu_ids)

    test(joint_model, test_loader, input_index=1, output_index=2, offset_output_index=0, save_output=True)
# elif "cascade_test_abhishek" in opt_exp.phase:
#     B_test,A_test,labels_test = load_data(testpath[0])
#     for i in range(len(testpath)-1):
#         f,f1,l = load_data(testpath[i+1])
#         B_test = torch.cat((B_test, f), 0)
#         A_test = torch.cat((A_test, f1), 0)
#         labels_test = torch.cat((labels_test, l), 0)

#     labels_test = torch.unsqueeze(labels_test, 1)

#     test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)
#     test_loader =torch.utils.data.DataLoader(test_data, batch_size=opt_exp.batch_size, shuffle=False)

#     dataset_size = len(test_loader)
#     print('#training images = %d' % dataset_size)

#     enc_model.load_networks(enc_model.opt.starting_epoch_count)
#     dec_model.load_networks(dec_model.opt.starting_epoch_count)

#     generator_outputs = []
#     location_outputs = []
#     for i, data in enumerate(test_loader):
#             joint_model.set_input(data[0], data[2])
#             joint_model.test()
#             gen_outputs = joint_model.generator.output
#             loc_outputs = joint_model.location.output
#             generator_outputs.extend(gen_outputs.data.cpu().numpy())
#             location_outputs.extend(loc_outputs.data.cpu().numpy())

#     if not os.path.exists(opt_exp.results_dir+"/"+opt_exp.phase):
#         os.makedirs(opt_exp.results_dir+"/"+opt_exp.phase, exist_ok=True)
#     scipy.io.savemat(enc_model.results_save_dir+"/"+enc_model.model_name+".mat", mdict={
#                                                                        "outputs":generator_outputs})
#     scipy.io.savemat(dec_model.results_save_dir+"/"+dec_model.model_name+".mat", mdict={
#                                                                        "outputs":location_outputs})


# elif "train" in opt_exp.phase:

#     B_train,A_train,labels_train = load_data(trainpath[0])
#     for i in range(len(trainpath)-1):
#         f,f1,l = load_data(trainpath[i+1])
#         B_train = torch.cat((B_train, f), 0)
#         A_train = torch.cat((A_train, f1), 0)
#         labels_train = torch.cat((labels_train, l), 0)

#     labels_train = torch.unsqueeze(labels_train, 1)
#     train_data = torch.utils.data.TensorDataset(B_train, A_train, labels_train)

#     B_test,A_test,labels_test = load_data(testpath[0])
#     for i in range(len(testpath)-1):
#         f,f1,l = load_data(testpath[i+1])
#         B_test = torch.cat((B_test, f), 0)
#         A_test = torch.cat((A_test, f1), 0)
#         labels_test = torch.cat((labels_test, l), 0)

#     labels_test = torch.unsqueeze(labels_test, 1)

#     test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)
#     test_loader =torch.utils.data.DataLoader(test_data, batch_size=opt_exp.batch_size, shuffle=False)

#     if opt_exp.isTrainGen:
#         train_loader =torch.utils.data.DataLoader(train_data, batch_size(opt_encoder.batch_size, shuffle=True)
#         dataset_size = len(train_loader)
#         print('#training images = %d' % dataset_size)
#         train(enc_model, train_loader, test_loader, input_index=1, output_index=0)

#     if opt_exp.isTrainLoc:
#         train_loader =torch.utils.data.DataLoader(train_data, batch_size=opt_decoder.batch_size, shuffle=True)
#         dataset_size = len(train_loader)
#         print('#training images = %d' % dataset_size)
#         train(dec_model, train_loader, test_loader, input_index=1, output_index=2)

# elif opt_exp.phase == "eval":
#   B_train,A_train,labels_train = load_data(trainpath[0])
#     for i in range(len(trainpath)-1):
#         f,f1,l = load_data(trainpath[i+1])
#         B_train = torch.cat((B_train, f), 0)
#         A_train = torch.cat((A_train, f1), 0)
#         labels_train = torch.cat((labels_train, l), 0)
    # labels_train = torch.unsqueeze(labels_train, 1)

#     train_data = torch.utils.data.TensorDataset(B_train, A_train, labels_train)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt_exp.batch_size, shuffle=False)

#     dataset_size = len(train_loader)
#     print('#training images = %d' % dataset_size)

#     eval(enc_model, train_loader, input_index=1, output_index=0, max_data_to_run=10000/opt_exp.batch_size)
#     eval(dec_model, train_loader, input_index=1, output_index=2, max_data_to_run=10000/opt_exp.batch_size)

# elif "test_abhishek" in opt_exp.phase:
#     B_test,A_test,labels_test = load_data(testpath[0])
#     for i in range(len(testpath)-1):
#         f,f1,l = load_data(testpath[i+1])
#         B_test = torch.cat((B_test, f), 0)
#         A_test = torch.cat((A_test, f1), 0)
#         labels_test = torch.cat((labels_test, l), 0)

#     labels_test = torch.unsqueeze(labels_test, 1)

#     test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)
#     test_loader =torch.utils.data.DataLoader(test_data, batch_size=opt_exp.batch_size, shuffle=False)

#     dataset_size = len(test_loader)
#     print('#training images = %d' % dataset_size)

#     enc_model.load_networks(enc_model.opt.starting_epoch_count)
#     dec_model.load_networks(dec_model.opt.starting_epoch_count)
#     # add load
#     # eval(enc_model, test_loader, input_index=1, output_index=0)
#     eval(dec_model, test_loader, input_index=0, output_index=2)


# elif "test" in opt_exp.phase:
#     B_test,A_test,labels_test = load_data(testpath[0])
#     for i in range(len(testpath)-1):
#         f,f1,l = load_data(testpath[i+1])
#         B_test = torch.cat((B_test, f), 0)
#         A_test = torch.cat((A_test, f1), 0)
#         labels_test = torch.cat((labels_test, l), 0)

#     labels_test = torch.unsqueeze(labels_test, 1)

#     enc_model.load_networks(enc_model.opt.starting_epoch_count)
#     dec_model.load_networks(dec_model.opt.starting_epoch_count)
#     test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)

#     test_loader =torch.utils.data.DataLoader(test_data, batch_size(opt_encoder.batch_size, shuffle=False)
#     dataset_size = len(test_loader)
#     print('#training images = %d' % dataset_size)
#     eval(enc_model, test_loader, input_index=1, output_index=0)

#     test_loader =torch.utils.data.DataLoader(test_data, batch_size=opt_decoder.batch_size, shuffle=False)
#     dataset_size = len(test_loader)
#     print('#training images = %d' % dataset_size)
#     eval(dec_model, test_loader, input_index=1, output_index=2)

# elif "retrain" in opt_exp.phase:
#     B_train,A_train,labels_train = load_data(trainpath[0])
#     for i in range(len(trainpath)-1):
#         f,f1,l = load_data(trainpath[i+1])
#         B_train = torch.cat((B_train, f), 0)
#         A_train = torch.cat((A_train, f1), 0)
#         labels_train = torch.cat((labels_train, l), 0)

#     labels_train = torch.unsqueeze(labels_train, 1)
#     train_data = torch.utils.data.TensorDataset(B_train, A_train, labels_train)

#     B_test,A_test,labels_test = load_data(testpath[0])
#     for i in range(len(testpath)-1):
#         f,f1,l = load_data(testpath[i+1])
#         B_test = torch.cat((B_test, f), 0)
#         A_test = torch.cat((A_test, f1), 0)
#         labels_test = torch.cat((labels_test, l), 0)

#     labels_test = torch.unsqueeze(labels_test, 1)

#     enc_model.load_networks(enc_model.opt.starting_epoch_count)
#     dec_model.load_networks(dec_model.opt.starting_epoch_count)
#     test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)

#     labels_test = torch.unsqueeze(labels_test, 1)

#     test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)
#     test_loader =torch.utils.data.DataLoader(test_data, batch_size=opt_exp.batch_size, shuffle=False)

#     if opt_exp.isTrainGen:
#         train_loader =torch.utils.data.DataLoader(train_data, batch_size(opt_encoder.batch_size, shuffle=True)
#         dataset_size = len(train_loader)
#         print('#training images = %d' % dataset_size)
#         train(enc_model, train_loader, test_loader, input_index=1, output_index=0)

#     if opt_exp.isTrainLoc:
#         train_loader =torch.utils.data.DataLoader(train_data, batch_size=opt_decoder.batch_size, shuffle=True)
#         dataset_size = len(train_loader)
#         print('#training images = %d' % dataset_size)
#         train(dec_model, train_loader, test_loader, input_index=1, output_index=2)