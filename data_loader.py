#!/usr/bin/python
'''
Data Loading Code for the DLoc dataset
Has a simpler dataloading pipeline for loading
Legacy mat file chunks, and
Custom DataLoader class DLocDataset that loads instances
from the folder as required into the memory.
'''
import os
from termios import FF1
from tracemalloc import start
import torch
import h5py
import scipy.io
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler
from torchvision import transforms, utils
import time
from params import opt_encoder, opt_exp, opt_decoder
import hdf5storage

#######################Legacy DataLoader########################
# A simple data loader that imports the train and test mat files
# from the `filename` and converts them to torch.tensors()
# to be loaded for training and testing DLoc network
# `features_wo_offset`: targets for the consistency decoder
# `features_w_offset` : inputs for the network/encoder
# `labels_gaussian_2d`: targets for the location decoder
# Demands multiple of 10s of Gigabytes of system memory
def load_data(filename):
    print('Loading '+filename + " in regular")
    arrays = {}
    start_time = time.time()
    f = h5py.File(filename,'r')
    features_wo_offset = torch.tensor(np.transpose(np.array(f.get('features_wo_offset'), dtype=np.float32)), dtype=torch.float32)
    features_w_offset = torch.tensor(np.transpose(np.array(f.get('features_w_offset'), dtype=np.float32)), dtype=torch.float32)
    labels_gaussian_2d = torch.tensor(np.transpose(np.array(f.get('labels_gaussian_2d'), dtype=np.float32)), dtype=torch.float32)
    
    return features_wo_offset,features_w_offset, labels_gaussian_2d
    
def loadPhoneFiles(root_dir, transform=None):
    data = [f for f in os.listdir(root_dir) if "np" not in f] 

    labels_gaussian_2d = np.array(np.load(os.path.join(root_dir, "guassian_labels.npz"))['labels_guassian'], dtype=np.float32)
    labels_gaussian_2d = np.expand_dims(labels_gaussian_2d,axis=1)
    print("Loading Data...")
    for file_num, ap in enumerate(data):
        f_wo_offset = np.load(os.path.join(root_dir, ap, "features_wo_offset.npz"))['arr_0'] # pkt * x * y
        f_w_offset = np.load(os.path.join(root_dir, ap, "features_w_offset.npz"))['arr_0']

        for i in range(f_wo_offset.shape[0]):
            f_wo_offset[i, :, :] = ((f_wo_offset[i, :, :] - np.min(f_wo_offset[i, :, :]))/ (np.max(f_wo_offset[i, :, :]) - np.min(f_wo_offset[i, :, :])))
        for j in range(f_w_offset.shape[0]): 
            f_w_offset[j, :, :] = ((f_w_offset[j, :, :] - np.min(f_w_offset[j, :, :]))/ (np.max(f_w_offset[j, :, :]) - np.min(f_w_offset[j, :, :])))

        if file_num == 0:
            size = f_w_offset.shape
            features_wo_offset = np.zeros((size[0], len(data), size[1], size[2]), dtype=np.float32) # pkt * ap * X * Y
            features_w_offset = np.zeros((size[0], len(data), size[1], size[2]), dtype=np.float32)
        
        for i in range(f_w_offset.shape[0]):
            features_wo_offset[i, file_num, :, :] = f_wo_offset[i]
            features_w_offset[i, file_num, :, :] = f_w_offset[i]
    
    # Random AP Testing
    # for pkt in range(features_wo_offset.shape[0]):
    #     np.random.shuffle(features_w_offset[pkt])
    #     # np.random.shuffle(features_wo_offset[pkt])

    # # Zero out AP Testing
    for pkt in range(features_w_offset.shape[0]):
        if (np.random.rand() <= 0.5):
            features_w_offset[pkt, np.random.randint(features_w_offset.shape[1]), :, :] = np.zeros((features_w_offset.shape[2], features_w_offset.shape[3]))

    # Farthest AP Zero Testing
    # zero_labels = np.load(os.path.join(root_dir, "zero_path_labels.npy"))
    # for i in range(zero_labels.shape[0]):
    #     features_w_offset[i, int(zero_labels[i]), :, :] = np.zeros((features_w_offset.shape[2], features_w_offset.shape[3]))


    features_wo_offset = np.squeeze(features_wo_offset)
    features_w_offset = np.squeeze(features_w_offset)

    sample = {'features_wo_offset': features_wo_offset, 'features_w_offset': features_w_offset, 'labels_gaussian_2d': labels_gaussian_2d}
    if transform:
        sample = transform(sample)

    return sample          
    
#######################Custom DataLoader########################
# Demands only a few Gigabytes of memory
# `features_wo_offset`: targets for the consistency decoder
# `features_w_offset` : inputs for the network/encoder
# `labels_gaussian_2d`: targets for the location decoder
class DLocDataset(Dataset):
    # DLOC Dataset

    def __init__(self, root_dir, opt, transform=None, zeroAP=False):
        super(DLocDataset, self).__init__()
        """
        Args:
            root_dirs (list of strings): List of Directories with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        # print(self.root_dir)
        self.transform = transform
        self.zeroAP = zeroAP
        self.zeroAPIndices = []
        self.n_files = []
        self.opt = opt
        self.file_names = []
        for dir_names in self.root_dir:
            onlyfiles = next(os.walk(dir_names))[2]
            for files_temp in onlyfiles:
                self.file_names.append(os.path.join(dir_names, files_temp))
            self.n_files.append(len(onlyfiles))

        
        if self.zeroAP:
            self.zeroAPIndices = np.zeros((self.n_files[0]), dtype=np.int32)
            for i in range(self.n_files[0]):
                if np.random.rand() >= .5:
                    self.zeroAPIndices[i] = np.random.randint(1, opt_encoder.input_nc+1)
                    
            if not os.path.exists(opt_encoder.results_dir):
                os.makedirs(opt_encoder.results_dir, exist_ok=True)

            save_path = f"{opt_encoder.results_dir}/zero_indices_{self.n_files[0]}.h5"
            h5_options = hdf5storage.Options(store_python_metadata=True, matlab_compatible=True)
            hdf5storage.writes(
                mdict={"zero_ap_indices" : self.zeroAPIndices}, 
                filename=save_path,
                options=h5_options
            )
            print(f"Zero AP Indices saved in {save_path}")
        

    def __len__(self):
        return np.sum(self.n_files)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx_list = idx.tolist()
        elif isinstance(idx, int):
            idx_list = []
            idx_list.append(idx) 
        else:
            idx_list = idx
        # file_num = 0
        count = 0

        ## RSSI Check Code
        # rssi_file = h5py.File("../datasets/4_floor/extras/rssi.h5", 'r')
        # rssi = np.transpose(np.array(rssi_file.get('rssi')))
        ##

        for file_num, idx_num in enumerate(idx_list):
            filename = self.file_names[idx_num]
            
            f_wo_offset, f_w_offset, labels_2d, labels_xy, l_cl = self.get_sample_from_filename(filename)

            # rssi_list = rssi[idx_num]
            # for ap in range(f_wo_offset.shape[1]):
            #     if rssi_list[ap] <= -40:
            #         f_wo_offset[:, ap, :, :] *= np.exp(rssi_list[ap] / 40)
            #         f_w_offset[:, ap, :, :] *= np.exp(rssi_list[ap] / 40)
            count += 1
            # f_wo_offset = np.transpose(np.array(f.get('features_wo_offset'), dtype=np.float32))
            # f_w_offset = np.transpose(np.array(f.get('features_w_offset'), dtype=np.float32))
            # labels_2d = np.transpose(np.array(f.get('labels_gaussian_2d'), dtype=np.float32))


            if (file_num == 0) :
                features_wo_offset = f_wo_offset
                features_w_offset = f_w_offset
                labels_gaussian_2d = labels_2d
                labels_gaussian_xy = labels_xy
                labels_cl = l_cl
            else:
                features_wo_offset = np.concatenate((features_wo_offset, f_wo_offset),axis=0)
                features_w_offset = np.concatenate((features_w_offset, f_w_offset),axis=0)
                labels_gaussian_2d = np.concatenate((labels_gaussian_2d, labels_2d),axis=0)
                labels_gaussian_xy = np.concatenate((labels_gaussian_xy, labels_xy), axis=0)
                labels_cl = np.concatenate((labels_cl, l_cl), axis=0)
           
            if(count%1000 == 0):
                print('Loaded %d datapoints' % (count))
                
        features_wo_offset = np.squeeze(features_wo_offset)
        features_w_offset = np.squeeze(features_w_offset)

        if(len(np.shape(labels_gaussian_2d))==3):
            if(np.shape(labels_gaussian_2d)[0]!=1):
                labels_gaussian_2d = np.expand_dims(labels_gaussian_2d, axis=1)
        

        if self.opt.label_type == "AoA":
            # import pdb 
            # pdb.set_trace()
            f = h5py.File("/datasets/DLoc_data_split/dataset_jacobs_July28/features_aoa/ap.h5", 'r')
            ap_aoas = np.array(f.get('/ap_aoas'), dtype=np.float32)
            ap_locs = np.transpose(np.array(f.get('/aps'), dtype=np.float32))
            sample = {
                        'features_wo_offset': features_wo_offset, 
                        'features_w_offset': features_w_offset, 
                        'labels_gaussian_2d': labels_gaussian_2d, 
                        'labels_xy' : labels_gaussian_xy,
                        'labels_cl' : labels_cl, 
                        'ap_aoas' : ap_aoas,
                        'ap_locs' : ap_locs,
                    } 
        else:
            sample = {'features_wo_offset': features_wo_offset, 'features_w_offset': features_w_offset, 'labels_gaussian_2d': labels_gaussian_2d} 
        
        
        if self.transform:
            sample = self.transform(sample)

        return (sample, idx)

    ## helper loading function from filename

    def get_sample_from_filename(self, filename):
        x_vals = np.arange(self.opt.label_Xstart,self.opt.label_Xstop,self.opt.label_Xstep)
        y_vals = np.arange(self.opt.label_Ystart,self.opt.label_Ystop,self.opt.label_Ystep)
        xScale = self.opt.label_Xstop - self.opt.label_Xstart
        yScale = self.opt.label_Ystop - self.opt.label_Ystart
        f = h5py.File(filename,'r')
        # f_wo_offset = np.swapaxes(np.transpose(np.array(f.get('features_wo_offset'), dtype=np.float32)), 2, 3)
        # f_w_offset = np.swapaxes(np.transpose(np.array(f.get('features_w_offset'), dtype=np.float32)), 2, 3)

        if self.opt.label_type == "Gaussian":
            f_wo_offset = np.transpose(np.array(f.get('features_wo_offset'), dtype=np.float32))
            f_w_offset = np.transpose(np.array(f.get('features_w_offset'), dtype=np.float32))
            if not self.opt.label_new:
                labels_2d = np.transpose(np.array(f.get('labels_gaussian_2d'), dtype=np.float32))
            else:
                label = np.transpose(np.array(f.get('labels'), dtype=np.float32))
                labels_2d = get_gaussian_labels(labels, x_vals, y_vals, sigma=self.opt.label_sigma)
            return f_wo_offset, f_w_offset, labels_2d

        elif self.opt.label_type == "XY":
            label = np.squeeze(np.divide(np.subtract(np.transpose(np.array(f.get('labels'), dtype=np.float32)),[self.opt.label_Xstart, self.opt.label_Ystart]), [xScale, yScale]))
         
        elif self.opt.label_type == "BB":
            label = np.squeeze(np.divide(np.subtract(np.transpose(np.array(f.get('BB_labels'), dtype=np.float32)),
                                        [self.opt.label_Xstart - self.opt.box_width, self.opt.label_Ystart + self.opt.box_height, 0, 0]), 
                                        [xScale, yScale, xScale, yScale]))
        elif self.opt.label_type == "BB_fixed": 
            label = np.squeeze(np.divide(np.subtract(np.transpose(np.array(f.get('BB_labels'), dtype=np.float32)),
                                        [self.opt.label_Xstart - self.opt.box_width, self.opt.label_Ystart + self.opt.box_height, 0, 0]), 
                                        [xScale, yScale, xScale, yScale]))[:2]

        elif self.opt.label_type == "AoA":
            # import pdb
            # pdb.set_trace()
            f_wo_offset = np.transpose(np.array(f.get("features_2d"), dtype=np.float32))
            f_w_offset = np.transpose(np.array(f.get("features_2d"), dtype=np.float32))
            label = np.squeeze(np.transpose(np.array(f.get('aoa_gnd'), dtype=np.float32)))
            label_2d = np.squeeze(np.transpose(np.array(f.get('labels'), dtype=np.float32)))
            lab = (np.transpose(np.array(f.get('labels_cl'), dtype=np.float32)))*180/(315)
            if opt_decoder.loss_type == "Cross_Entropy":
                labels_cl = lab[:, opt_exp.AP_no]
            elif opt_decoder.loss_type == "KL_Div":
                labs_cl = int(lab[:, opt_exp.AP_no])
                labels_cl = np.zeros(180)
                labels_cl[labs_cl] = 1

            return f_wo_offset, f_w_offset, label.astype(np.float32), label_2d.astype(np.float32), labels_cl


        return f_wo_offset, f_w_offset, label.astype(np.float32)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features_wo_offset, features_w_offset, labels_gaussian_2d, labels_xy, ap_aoas, ap_locs, labels_cl = sample['features_wo_offset'], sample['features_w_offset'], sample['labels_gaussian_2d'], sample['labels_xy'], sample['ap_aoas'], sample['ap_locs'], sample['labels_cl']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        return {'labels_gaussian_2d': torch.tensor(labels_gaussian_2d, dtype=torch.float32),
                'labels_xy' : torch.tensor(labels_xy, dtype=torch.float32),
                'labels_cl' : torch.tensor(labels_cl, dtype=torch.long), # use dtype=torch.long for CrossEntropyLoss
                'features_wo_offset': torch.tensor(features_wo_offset, dtype=torch.float32),
                'features_w_offset': torch.tensor(features_w_offset, dtype=torch.float32),
                'ap_aoas' : torch.tensor(ap_aoas, dtype=torch.float32),
                'ap_locs' : torch.tensor(ap_locs, dtype=torch.float32)
                }


# class CustomDatasetDLocLoader():
#     """Wrapper class of Dataset class that performs multi-threaded data loading"""

#     def __init__(self, path, opt, shuffle=True):
#         """Initialize this class
#         Step 1: create a dataset instance given the name [dataset_mode]
#         Step 2: create a multi-threaded data loader.
#         """
#         self.opt = opt
#         self.dataset = DLocDataset(root_dir=path,
#                           transform=transforms.Compose([ToTensor()]))

#         if not shuffle:
#             self.dataloader = torch.utils.data.DataLoader(
#                 self.dataset,
#                 num_workers=int(opt.num_threads),
#                 sampler=BatchSampler(
#                 SequentialSampler(self.dataset), batch_size=opt.batch_size, drop_last=False
#                 ))
#         else:
#             self.dataloader = torch.utils.data.DataLoader(
#                 self.dataset,
#                 num_workers=int(opt.num_threads),
#                 sampler=BatchSampler(
#                 RandomSampler(self.dataset), batch_size=opt.batch_size, drop_last=False
#                 ))

#     def load_data(self):
#         return self

#     def __len__(self):
#         """Return the number of data in the dataset"""
#         return min(len(self.dataset), self.opt.max_dataset_size)

#     def __iter__(self):
        """Return a batch of data"""
        # for i, data in enumerate(self.dataloader):
        #     if i * self.opt.batch_size >= self.opt.max_dataset_size:
        #         break
        #     yield data


 # def __getitem__(self, idx):

    #     if torch.is_tensor(idx):
    #         idx_list = idx.tolist()
    #     elif isinstance(idx, int):
    #         idx_list = []
    #         idx_list.append(idx) 
    #     else:
	#        	idx_list = idx
    #     for file_num, idx_num in enumerate(idx_list):
    #         filename = self.file_names[idx_num]
    #         f = h5py.File(filename,'r')
    #         f_w_offset_201 = np.array(f.get('features_w_offset_201'), dtype=np.float32)
    #         f_w_offset_203 = np.array(f.get('features_w_offset_203'), dtype=np.float32)
    #         f_w_offset_204 = np.array(f.get('features_w_offset_204'), dtype=np.float32)
    #         f_w_offset_207 = np.array(f.get('features_w_offset_207'), dtype=np.float32)

    #         f_wo_offset_201 = np.array(f.get('features_w_offset_201'), dtype=np.float32)
    #         f_wo_offset_203 = np.array(f.get('features_w_offset_203'), dtype=np.float32)
    #         f_wo_offset_204 = np.array(f.get('features_w_offset_204'), dtype=np.float32)
    #         f_wo_offset_207 = np.array(f.get('features_w_offset_207'), dtype=np.float32)
            
    #         [x_len,y_len] = f_w_offset_201.shape

    #         f_w_offset = np.zeros((4,x_len,y_len))
    #         f_w_offset[0,:,:] = f_w_offset_201
    #         f_w_offset[1,:,:] = f_w_offset_203
    #         f_w_offset[2,:,:] = f_w_offset_204
    #         f_w_offset[3,:,:] = f_w_offset_207

    #         f_wo_offset = np.zeros((4,x_len,y_len))
    #         f_wo_offset[0,:,:] = f_wo_offset_201
    #         f_wo_offset[1,:,:] = f_wo_offset_203
    #         f_wo_offset[2,:,:] = f_wo_offset_204
    #         f_wo_offset[3,:,:] = f_wo_offset_207

    #         for i in range(f_wo_offset.shape[0]):
    #             f_wo_offset[i, :, :] = ((f_wo_offset[i, :, :] - np.min(f_wo_offset[i, :, :]))/ (np.max(f_wo_offset[i, :, :]) - np.min(f_wo_offset[i, :, :])))
    #         for j in range(f_w_offset.shape[0]): 
    #             f_w_offset[j, :, :] = ((f_w_offset[j, :, :] - np.min(f_w_offset[j, :, :]))/ (np.max(f_w_offset[j, :, :]) - np.min(f_w_offset[j, :, :])))
    #         labels_2d = np.array(f.get('gaussian_label'), dtype=np.float32)
    #         if (file_num == 0) :
    #             features_wo_offset = f_wo_offset
    #             features_w_offset = f_w_offset
    #             labels_gaussian_2d = labels_2d
    #         else:
    #             features_wo_offset = np.concatenate((features_wo_offset, f_wo_offset),axis=0)
    #             features_w_offset = np.concatenate((features_w_offset, f_w_offset),axis=0)
    #             labels_gaussian_2d = np.concatenate((labels_gaussian_2d, labels_2d),axis=0)
    
    #     if self.zeroAP:
    #         if self.zeroAPIndices[idx]:
    #             features_w_offset[self.zeroAPIndices[idx]-1, :, :] = np.zeros((features_w_offset.shape[1], features_w_offset.shape[2]))
    #             features_wo_offset[self.zeroAPIndices[idx]-1, :, :] = np.zeros((features_wo_offset.shape[1], features_wo_offset.shape[2]))
                    
    #     features_wo_offset = np.squeeze(features_wo_offset)
    #     features_w_offset = np.squeeze(features_w_offset)
    #     if(len(np.shape(labels_gaussian_2d)) == 2):
    #         if(np.shape(labels_gaussian_2d)[0]!=1):
    #             labels_gaussian_2d = np.expand_dims(labels_gaussian_2d, axis=0)
    #     sample = {'features_wo_offset': features_wo_offset, 'features_w_offset': features_w_offset, 'labels_gaussian_2d': labels_gaussian_2d}
    #     if self.transform:
    #         sample = self.transform(sample)

    #     return sample