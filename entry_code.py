#!/usr/bin/python
'''
Script for both training and evaluating the DLoc network
Automatically imports the parameters from params.py.
For further details onto which params file to load
read the README in `params_storage` folder.
'''

from comet_ml import Experiment
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
from utils import *
from modelADT import ModelADT
from Generators import *
from data_loader import load_data
from data_loader import loadPhoneFiles
from data_loader import ToTensor
from data_loader import DLocDataset
from joint_model import Enc_2Dec_Network
from joint_model import Enc_Dec_Network
from params import *
import os
import trainer
import hdf5storage

torch.manual_seed(0)
np.random.seed(0)


'''
Defining the paths from where to Load Data.
Assumes that the data is stored in a subfolder called data in the current data folder
'''

# datapath = ['../datasets/4_floor/aoa_features']

datapath = ['/datasets/DLoc_data_split/dataset_jacobs_July28/features_aoa/ind']

trainpath =  ['../datasets/4_floor/aoa_features']
validpath = ['../datasets/dataset_jacobs_July28/features/ind_valid']
testpath =  ['../datasets/dataset_jacobs_July28/features/ind_test']

# trainpath = ['../datasets/kaggle/features/ind/']
# testpath = ['../datasets/kaggle/testing/env4-1/features/ind/', '../datasets/kaggle/testing/env4-2/features/ind/']

# train_data = DLocDataset(trainpath, transform=transforms.Compose([ToTensor()]), opt=opt_exp)
# valid_data = DLocDataset(validpath, transform=transforms.Compose([ToTensor()]), opt=opt_exp)
# test_data = DLocDataset(testpath, transform=transforms.Compose([ToTensor()]), opt = opt_exp)
# train_size = int(0.85 * len(train_data))
# valid_size = len(train_data) - train_size

# train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

# train_loader = torch.utils.data.DataLoader(train_data, num_workers=int(opt_exp.num_threads), batch_size=opt_exp.batch_size)
# valid_loader = torch.utils.data.DataLoader(valid_data, num_workers=int(opt_exp.num_threads), batch_size=opt_exp.batch_size)
# test_loader = torch.utils.data.DataLoader(test_data, num_workers=int(opt_exp.num_threads), batch_size=opt_exp.batch_size)

# if "ind" in datapath[0] and not opt_exp.disjoint:
train_data = DLocDataset(datapath,
                            transform=transforms.Compose([ToTensor()]), 
                            opt = opt_exp
                        )

train_size = int(0.7 * len(train_data))
valid_size = int(0.1 * len(train_data))
test_size = len(train_data) - train_size - valid_size 

train_data, valid_data, test_data = torch.utils.data.random_split(train_data, [train_size, valid_size, test_size])

train_loader = torch.utils.data.DataLoader(train_data,
                            num_workers=int(opt_exp.num_threads),
                            batch_size=opt_exp.batch_size, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_data,
                            num_workers=int(opt_exp.num_threads),
                            batch_size=opt_exp.batch_size, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data,
                            num_workers=int(opt_exp.num_threads),
                               batch_size=opt_exp.batch_size, drop_last=True)

# elif "ind" in datapath[0] and opt_exp.disjoint:

#     train_data = DLocDataset(trainpath, 
#                               transform=transforms.Compose([ToTensor()]), zeroAP=True)

#     test_data = DLocDataset(testpath,
#                               transform=transforms.Compose([ToTensor()]), zeroAP=True)

#     train_size = int(0.85 * len(train_data))
#     valid_size = len(train_data) - train_size

#     train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])
#     test_data = test_data

#     train_loader =torch.utils.data.DataLoader(train_data,
#                               num_workers=int(opt_exp.num_threads),
#                               batch_size=opt_exp.batch_size)
#     valid_loader =torch.utils.data.DataLoader(valid_data,
#                               num_workers=int(opt_exp.num_threads),
#                               batch_size=opt_exp.batch_size)
#     test_loader = torch.utils.data.DataLoader(test_data,
#                               num_workers=int(opt_exp.num_threads),
#                               batch_size=opt_exp.batch_size)
    

# Print Training Data Info 
(samples, _) = train_data[1]

print('Length of training data = %d' % len(train_data))
print(f"Size of features wo offset {samples['features_wo_offset'].size()}")
print(f"Size of features w offset  {samples['features_w_offset'].size()}")
print(f"Size of labels             {samples['labels_cl'].size()}")
print('# training mini batch = %d' % len(train_loader))

# Print Validation Data Info
(sample, _) = valid_data[1]

print('Length of Validation data = %d' % len(valid_data))
print(f"Size of features wo offset {sample['features_wo_offset'].size()}")
print(f"Size of features w offset  {sample['features_w_offset'].size()}")
print(f"Size of labels             {sample['labels_cl'].size()}")
print('# validation mini batch = %d' % len(valid_loader))

# Print Test Data Info
(sample, _) = test_data[1]

print('Length of Testing data = %d' % len(test_data))
print(f"Size of features wo offset {sample['features_wo_offset'].size()}")
print(f"Size of features w offset  {sample['features_w_offset'].size()}")
print(f"Size of labels             {sample['labels_cl'].size()}")
print('# testing mini batch = %d' % len(test_loader))

##########################################################

# Loading Training and Evaluation Data into their respective Dataloaders
# load traning data


# data = loadPhoneFiles(datapath[0], transform = transforms.Compose([ToTensor()]))

# dataset = torch.utils.data.TensorDataset(data['features_wo_offset'], data['features_w_offset'], data['labels_gaussian_2d'])

# # Accounts for the difference when flooring the indicies to ints
# difference = len(dataset) - int(.7 * len(dataset)) - int(.1 * len(dataset))  - int(.2 * len(dataset))
# train_data, valid_data, test_data = torch.utils.data.random_split(dataset, [int(.7 * len(dataset)), int(.1 * len(dataset)), int(.2 * len(dataset)) + difference])

# train_loader = DataLoader(train_data,
#                           num_workers=int(opt_exp.num_threads),
#                           batch_size=opt_exp.batch_size, drop_last=False)

# valid_loader = DataLoader(valid_data,
#                           num_workers=int(opt_exp.num_threads),
#                           batch_size=opt_exp.batch_size, drop_last=False
#                           )

# test_loader = DataLoader(test_data, 
#                           num_workers=int(opt_exp.num_threads), 
#                           batch_size=opt_exp.batch_size, drop_last=False)

# print("Dataset is Split")

# print('Length of training data = %d' % len(train_data))
# print(f"Size of features wo offset {len(train_data[:][0])}")
# print(f"Size of features w offset  {len(train_data[:][1])}")
# print(f"Size of labels             {len(train_data[:][2])}")
# print('# training mini batch = %d' % len(train_loader))


# print('Length of Validation data = %d' % len(valid_data))
# print(f"Size of features wo offset {len(valid_data[:][0])}")
# print(f"Size of features w offset  {len(valid_data[:][1])}")
# print(f"Size of labels             {len(valid_data[:][2])}")
# print('# validation mini batch = %d' % len(valid_loader))


# print('Length of Test data = %d' % len(test_data))
# print(f"Size of features wo offset {len(test_data[:][0])}")
# print(f"Size of features w offset  {len(test_data[:][1])}")
# print(f"Size of labels             {len(test_data[:][2])}")
# print('# test mini batch = %d' % len(test_loader))

############################################################

'''
Initiate the Network and build the graph
'''

print('\n' + '#' * 10)
print(f'Initiating the Network and Building the Graph\n' + '#' * 10 + '\n\n')


if opt_exp.n_decoders >= 1:

    # init encoder
    enc_model = ModelADT()
    enc_model.initialize(opt_encoder)
    enc_model.setup(opt_encoder)

    # init decoder1
    dec_model = ModelADT()
    dec_model.initialize(opt_decoder)
    dec_model.setup(opt_decoder)

    if opt_exp.n_decoders == 2:
        # init decoder2
        offset_dec_model = ModelADT()
        offset_dec_model.initialize(opt_offset_decoder)
        offset_dec_model.setup(opt_offset_decoder)

        # join all models
        print('Making the joint_model')
        joint_model = Enc_2Dec_Network()
        joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, gpu_ids=opt_exp.gpu_ids)

    elif opt_exp.n_decoders == 1:
        # join all models
        print('Making the joint_model')
        joint_model = Enc_Dec_Network()
        joint_model.initialize(opt_exp, enc_model, dec_model, gpu_ids=opt_exp.gpu_ids)

elif opt_exp.n_decoders == 0:
    cons_model = ModelADT()
    cons_model.initialize(opt_network)
    cons_model.setup(opt_network)

else:
    print('Incorrect number of Decoders specified in the parameters')

if opt_exp.continue_train:
    enc_model.load_networks(opt_encoder.starting_epoch_count)
    dec_model.load_networks(opt_decoder.starting_epoch_count)
    if opt_exp.n_decoders == 2:
        offset_dec_model.load_networks(opt_offset_decoder.starting_epoch_count)

# train the model

experiment = Experiment(
    api_key="HCAGGOQqR93FZTETKVAsIhKnW",
    project_name="sharan-classification",
    workspace="asroshan",
)
experiment.set_name(opt_exp.save_name)

# Report multiple hyperparameters using a dictionary:
if opt_exp.n_decoders >= 1:
    hyper_params = {
        "exp_time_name": opt_exp.save_name,
        "input_data": opt_exp.data,
        "learning_rate": opt_exp.lr,
        "num_layers": opt_exp.n_decoders,
        "decoder_loss": opt_decoder.loss_type,
        "decoder_lambda": opt_decoder.lambda_L,
        "decoder_reg": opt_decoder.lambda_reg,
        "batch_size": opt_exp.batch_size,
        "num_epochs": opt_exp.n_epochs,
        "encoder_dropout" : opt_encoder.no_dropout,
        "decoder_dropout" : opt_decoder.no_dropout
        # "offset_decoder_loss": opt_offset_decoder.loss_type,
        # "offset_decoder_lambda": opt_offset_decoder.lambda_L,
        # "offset_decoder_reg": opt_offset_decoder.lambda_reg
    }
else:
    hyper_params = {
        "exp_time_name": opt_exp.save_name,
        "input_data": opt_exp.data,
        "learning_rate": opt_exp.lr,
        "num_layers": opt_network.n_blocks,
        "loss": opt_network.loss_type,
        "lambda": opt_network.lambda_L,
        "reg": opt_network.lambda_reg,
        "cross": opt_network.lambda_cross,
        "batch_size": opt_exp.batch_size,
        "num_epochs": opt_exp.n_epochs,
        "dropout" : opt_network.dropout_rate
    }

experiment.log_parameters(hyper_params)
'''
Training the network
'''
print('Beginning Training of the Model')

trainer.train(joint_model, train_loader, valid_loader, experiment)

'''
Model Evaluation at the best epoch
'''

epoch = "latest"  # int/"best"/"last"
# load network
if opt_exp.n_decoders == 2:
    offset_dec_model.load_networks(epoch)
    joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, gpu_ids = opt_exp.gpu_ids)
elif opt_exp.n_decoders == 1:
    joint_model.initialize(opt_exp, enc_model, dec_model, gpu_ids = opt_exp.gpu_ids)
elif opt_exp.n_decoders == 0:
    cons_model.load_networks(epoch)
    cons_model.initialize(opt_network)
    cons_model.setup(opt_network)

# pass data through model
total_loss, median_error = trainer.test(joint_model,
    test_loader, 
    experiment,
    save_output=True,
    save_name=f"decoder_test_result_epoch_{epoch}",
    log=True)

print(f"total_loss: {total_loss}, median_error: {median_error}")