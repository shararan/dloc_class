import torch
import torchvision
import torch.nn as nn
import numpy as np
from data_loader import *
from params import *
from Generators import *
import matplotlib.pyplot as plt

# f = h5py.File("/Results/RetinaNet_runs/s2r2023_01_31_07h47m47s/decoder_test_result_epoch_latest.h5", 'r')
# print(list(f.keys()))

# error = np.array(f.get('error'))
# temp = error[2055][0]
# print(temp)
# name = h5py.h5r.get_name(temp, f.id)
# data = f[name]
# print(np.array(data))

# x = np.sort(error)
# y = np.arange(len(x))/float(len(x))
# plt.plot(x, y)



# scp -r interns@137.110.118.67:/media/datadisk_2/loc_results/wifi/Intern_Results/RetinaNet_runs/s2r2023_02_01_14h38m52s/decoder_test_result_epoch_latest.h5 ~/fov_test

# model_conv = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
# model = list(model_conv.head.regression_head.children())
# model += [nn.Flatten()]
# linear_shape = int(np.prod((8, 32, 41, 91))/8)
# model += [nn.Linear(linear_shape, 256), nn.ReLU()]
# model += [nn.Linear(256, 32), nn.ReLU()]
# model += [nn.Linear(32, 2)]
# model += [nn.Tanh()]
# print(nn.Sequential(*model))

loadpath = ['/datasets/DLoc_data_split/dataset_jacobs_July28/features_aoa/ind']
train_data = DLocDataset(loadpath, opt_exp, transform=transforms.Compose([ToTensor()]))

sample = train_data[20]
print((sample[0]["labels_cl"]))

# f = h5py.File("/datasets/DLoc_data_split/dataset_jacobs_July28/features_aoa/ind/5.h5", 'r')
# print(list(f.keys()))
# lcl = (np.transpose(np.array(f.get('labels_cl'), dtype=np.float32)))
# lab = lcl[:, 2]
# print(torch.tensor(lab, dtype=torch.long))

# y = torch.rand(360)
# x = torch.tensor(lab, dtype=torch.long)

# example = train_data[1]
# print(torch.Tensor.size(example["labels"]))
# model = []
# model += [ResnetEncoder(4, 1)]
# model += [AoAClassifier()]

# mod = nn.Sequential(*model)

# x = torch.rand(4,315,401)
# print(torch.Tensor.size(mod(x)))