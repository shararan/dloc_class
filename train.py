#!/usr/bin/python
import torch
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
from utils import *
from modelADT import ModelADT
from Generators import *
from data_loader import load_data
from joint_model import Enc_2Dec_Network
from params import *
import trainer

# data path
# trainpath = ['/media/datadisk/Roshan/datasets/quantenna/features/dataset_edit_jacobs_July28.mat']
# testpath = ['/media/datadisk/Roshan/datasets/quantenna/features/dataset_fov_test_jacobs_July28_2.mat']
trainpath = ['/media/ehdd_8t1/chenfeng/phone_data/dataset_phone_4AP_train.mat']
testpath = ['/media/ehdd_8t1/chenfeng/phone_data/dataset_phone_4AP_test.mat']

# init encoder
enc_model = ModelADT()
enc_model.initialize(opt_encoder)
enc_model.setup(opt_encoder)

# init decoder1
dec_model = ModelADT()
dec_model.initialize(opt_decoder)
dec_model.setup(opt_decoder)

# init decoder2
offset_dec_model = ModelADT()
offset_dec_model.initialize(opt_offset_decoder)
offset_dec_model.setup(opt_offset_decoder)

# join all models
print('Making the joint_model')
joint_model = Enc_2Dec_Network()
joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, frozen_dec=opt_exp.isFrozen, gpu_ids=opt_exp.gpu_ids)

# load traning data
B_train,A_train,labels_train = load_data(trainpath[0], 0, 0, 0, 1)
for i in range(len(trainpath)-1):
    f,f1,l = load_data(trainpath[i+1], 0, 0, 0, 1)
    B_train = torch.cat((B_train, f), 0)
    A_train = torch.cat((A_train, f1), 0)
    labels_train = torch.cat((labels_train, l), 0)
labels_train = torch.unsqueeze(labels_train, 1)
train_data = torch.utils.data.TensorDataset(B_train, A_train, labels_train)
train_loader =torch.utils.data.DataLoader(train_data, batch_size=opt_exp.batch_size, shuffle=True)

print(f"A_train.shape: {A_train.shape}")
print(f"B_train.shape: {B_train.shape}")
print(f"labels_train.shape: {labels_train.shape}")
print('# training mini batch = %d' % len(train_loader))

# load testing data
B_test,A_test,labels_test = load_data(testpath[0], 0, 0, 0, 1)
for i in range(len(testpath)-1):
    f,f1,l = load_data(testpath[i+1], 0, 0, 0, 1)
    B_test = torch.cat((B_test, f), 0)
    A_test = torch.cat((A_test, f1), 0)
    labels_test = torch.cat((labels_test, l), 0)
labels_test = torch.unsqueeze(labels_test, 1)

# create data loader
test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)
test_loader =torch.utils.data.DataLoader(test_data, batch_size=opt_exp.batch_size, shuffle=False)
print(f"A_test.shape: {A_test.shape}")
print(f"B_test.shape: {B_test.shape}")
print(f"labels_test.shape: {labels_test.shape}")
print('# testing mini batch = %d' % len(test_loader))
print('Test Data Loaded')

if opt_exp.isFrozen:
    enc_model.load_networks(opt_encoder.starting_epoch_count)
    dec_model.load_networks(opt_decoder.starting_epoch_count)
    offset_dec_model.load_networks(opt_offset_decoder.starting_epoch_count)

# train the model
trainer.train(joint_model, train_loader, test_loader)