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

# data path and best epoch
testpath = ['/media/datadisk/Roshan/datasets/quantenna/features/dataset_fov_test_jacobs_July28_2.mat']     
best_epoch = 1

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
joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, frozen_dec = opt_exp.isFrozen, gpu_ids = opt_exp.gpu_ids)

# load test data
B_test,A_test,labels_test = load_data(testpath[0], 0, 0, 0, 1)
for i in range(len(testpath)-1):
    f,f1,l = load_data(testpath[i+1], 0, 0, 0, 1)
    B_test = torch.cat((B_test, f), 0)
    A_test = torch.cat((A_test, f1), 0)
    labels_test = torch.cat((labels_test, l), 0)

# create data loader 
labels_test = torch.unsqueeze(labels_test, 1)
test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)
test_loader =torch.utils.data.DataLoader(test_data, batch_size=opt_exp.batch_size, shuffle=False)
print('#training images = %d' % len(test_loader))

# load network
enc_model.load_networks(best_epoch)
dec_model.load_networks(best_epoch)
offset_dec_model.load_networks(best_epoch)
joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, frozen_dec = opt_exp.isFrozen, gpu_ids = opt_exp.gpu_ids)

# pass data through model
total_loss, median_error = trainer.test(joint_model, 
    test_loader, 
    input_index=1, 
    output_index=2, 
    offset_output_index=0, 
    save_output=False, 
    log=False)
print(f"total_loss: {total_loss}, median_error: {median_error}")
