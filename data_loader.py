#!/usr/bin/python
import torch
import h5py
import numpy as np
from params import *
from utils import RGB2Gray
from PIL import Image
from torch.utils.data import Dataset


def pre_load_data(root_dir, map_dir, ds_step):
    
    file_name = root_dir[0]
    print('Loading ',file_name)
    f = h5py.File(file_name,'r')            

    read_labels_gaussian_2d = np.array(f.get('labels_gaussian_2d'), dtype=np.float32)
    read_wo_offset = np.array(f.get('features_wo_offset'), dtype=np.float32)
    read_w_offset = np.array(f.get('features_w_offset'), dtype=np.float32)

    ds_wo_offset = read_wo_offset[:,:,:,0:read_wo_offset.shape[3]:ds_step]
    ds_w_offset = read_w_offset[:,:,:,0:read_w_offset.shape[3]:ds_step]
    ds_labels_gaussian_2d = read_labels_gaussian_2d[:,:,0:read_labels_gaussian_2d.shape[2]:ds_step]

    features_wo_offset = np.transpose(ds_wo_offset)
    features_w_offset = np.transpose(ds_w_offset)
    labels_gaussian_2d = np.transpose(ds_labels_gaussian_2d)
    
    map_image = RGB2Gray(np.array(Image.open(map_dir[0]))/255)
    map_reshape = np.repeat(map_image[np.newaxis,np.newaxis,:],features_w_offset.shape[0],axis=0)

    for i in range(len(root_dir)-1):
        file_name = root_dir[i+1]
        print('Loading '+file_name)
        f = h5py.File(file_name,'r')            
    
        read_labels_gaussian_2d = np.array(f.get('labels_gaussian_2d'), dtype=np.float32)
        read_wo_offset = np.array(f.get('features_wo_offset'), dtype=np.float32)
        read_w_offset = np.array(f.get('features_w_offset'), dtype=np.float32)
    
        ds_wo_offset = read_wo_offset[:,:,:,0:read_wo_offset.shape[3]:ds_step]
        ds_w_offset = read_w_offset[:,:,:,0:read_w_offset.shape[3]:ds_step]
        ds_labels_gaussian_2d = read_labels_gaussian_2d[:,:,0:read_labels_gaussian_2d.shape[2]:ds_step]
    
        features_wo_offset_get = np.transpose(ds_wo_offset)
        features_w_offset_get = np.transpose(ds_w_offset)
        labels_gaussian_2d_get = np.transpose(ds_labels_gaussian_2d)
        
        features_wo_offset = np.concatenate((features_wo_offset, features_wo_offset_get), axis=0)
        features_w_offset = np.concatenate((features_w_offset, features_w_offset_get), axis=0)
        labels_gaussian_2d = np.concatenate((labels_gaussian_2d, labels_gaussian_2d_get), axis=0)
        
        map_image = RGB2Gray(np.array(Image.open(map_dir[i+1]))/255)
        map_reshape_get = np.repeat(map_image[np.newaxis,np.newaxis,:],features_w_offset_get.shape[0],axis=0)
        map_reshape = np.concatenate((map_reshape, map_reshape_get), axis=0)
        
    return (features_wo_offset,
            features_w_offset,
            labels_gaussian_2d,
            map_reshape
            )
    
class rw_to_rw_Dataset(Dataset):
    def __init__(self, features_wo_offset, features_w_offset, labels_gaussian_2d, map_reshape):

        self.features_wo_offset = features_wo_offset
        self.features_w_offset = features_w_offset
        self.labels_gaussian_2d = labels_gaussian_2d
        self.map_reshape = map_reshape

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()     

        features_wo_offset_item = self.features_wo_offset[idx];
        features_w_offset_item = self.features_w_offset[idx];
        labels_gaussian_2d_item = self.labels_gaussian_2d[idx];
        map_reshape_item = self.map_reshape[idx];
        
        return (
            torch.from_numpy(features_wo_offset_item).to(dtype=torch.float32),
            torch.from_numpy(features_w_offset_item).to(dtype=torch.float32),
            torch.from_numpy(labels_gaussian_2d_item).to(dtype=torch.float32),
            torch.from_numpy(map_reshape_item).to(dtype=torch.float32)       
        )        

    def __len__(self):
        length = self.labels_gaussian_2d.shape[0]
        return length    

class rw_to_rw_Dataset_trn_(Dataset):
    def __init__(self, root_dir, map_dir, ds_step):

        self.root_dir = root_dir
        self.map_dir = map_dir
        self.ds_step = ds_step

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()     
                        
        file_name = self.root_dir[0]
        print('Loading ',file_name)
        f = h5py.File(file_name,'r')            

        read_labels_gaussian_2d = np.array(f.get('labels_gaussian_2d'), dtype=np.float32)
        read_wo_offset = np.array(f.get('features_wo_offset'), dtype=np.float32)
        read_w_offset = np.array(f.get('features_w_offset'), dtype=np.float32)
    
        ds_wo_offset = read_wo_offset[:,:,:,0:read_wo_offset.shape[3]:self.ds_step]
        ds_w_offset = read_w_offset[:,:,:,0:read_w_offset.shape[3]:self.ds_step]
        ds_labels_gaussian_2d = read_labels_gaussian_2d[:,:,0:read_labels_gaussian_2d.shape[2]:self.ds_step]
    
        features_wo_offset = np.transpose(ds_wo_offset)
        features_w_offset = np.transpose(ds_w_offset)
        labels_gaussian_2d = np.transpose(ds_labels_gaussian_2d)
        
        map_image = RGB2Gray(np.array(Image.open(self.map_dir[0]))/255)
        map_reshape = np.repeat(map_image[np.newaxis,np.newaxis,:],features_w_offset.shape[0],axis=0)
          

        for i in range(len(self.root_dir)-1):
            file_name = self.root_dir[i+1]
            print('Loading '+file_name)
            f = h5py.File(file_name,'r')            
        
            read_labels_gaussian_2d = np.array(f.get('labels_gaussian_2d'), dtype=np.float32)
            read_wo_offset = np.array(f.get('features_wo_offset'), dtype=np.float32)
            read_w_offset = np.array(f.get('features_w_offset'), dtype=np.float32)
        
            ds_wo_offset = read_wo_offset[:,:,:,0:read_wo_offset.shape[3]:self.ds_step]
            ds_w_offset = read_w_offset[:,:,:,0:read_w_offset.shape[3]:self.ds_step]
            ds_labels_gaussian_2d = read_labels_gaussian_2d[:,:,0:read_labels_gaussian_2d.shape[2]:self.ds_step]
        
            features_wo_offset_get = np.transpose(ds_wo_offset)
            features_w_offset_get = np.transpose(ds_w_offset)
            labels_gaussian_2d_get = np.transpose(ds_labels_gaussian_2d)
            
            features_wo_offset = np.concatenate((features_wo_offset, features_wo_offset_get), axis=0)
            features_w_offset = np.concatenate((features_w_offset, features_w_offset_get), axis=0)
            labels_gaussian_2d = np.concatenate((labels_gaussian_2d, labels_gaussian_2d_get), axis=0)
            
            map_image = RGB2Gray(np.array(Image.open(self.map_dir[i+1]))/255)
            map_reshape_get = np.repeat(map_image[np.newaxis,np.newaxis,:],features_w_offset_get.shape[0],axis=0)
            map_reshape = np.concatenate((map_reshape, map_reshape_get), axis=0)
            
        #print("features_wo_offset = ", features_wo_offset.shape)
        #print("features_w_offset = ", features_w_offset.shape)
        #print("labels_gaussian_2d_get = ", labels_gaussian_2d_get.shape)
        #print("map_reshape = ", map_reshape.shape)     
            
        features_wo_offset_item = features_wo_offset[idx];
        features_w_offset_item = features_w_offset[idx];
        labels_gaussian_2d_item = labels_gaussian_2d[idx];
        map_reshape_item = map_reshape[idx];
        
        return (
            torch.from_numpy(features_wo_offset_item).to(dtype=torch.float32),
            torch.from_numpy(features_w_offset_item).to(dtype=torch.float32),
            torch.from_numpy(labels_gaussian_2d_item).to(dtype=torch.float32),
            torch.from_numpy(map_reshape_item).to(dtype=torch.float32)       
        )        

    def __len__(self):
        length = 20846
        return length
    
class rw_to_rw_Dataset_tst_(Dataset):
    def __init__(self, root_dir, map_dir, ds_step):

        self.root_dir = root_dir
        self.map_dir = map_dir
        self.ds_step = ds_step

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()     
                        
        file_name = self.root_dir[0]
        print('Loading ',file_name)
        f = h5py.File(file_name,'r')            

        read_labels_gaussian_2d = np.array(f.get('labels_gaussian_2d'), dtype=np.float32)
        read_wo_offset = np.array(f.get('features_wo_offset'), dtype=np.float32)
        read_w_offset = np.array(f.get('features_w_offset'), dtype=np.float32)
    
        ds_wo_offset = read_wo_offset[:,:,:,0:read_wo_offset.shape[3]:self.ds_step]
        ds_w_offset = read_w_offset[:,:,:,0:read_w_offset.shape[3]:self.ds_step]
        ds_labels_gaussian_2d = read_labels_gaussian_2d[:,:,0:read_labels_gaussian_2d.shape[2]:self.ds_step]
    
        features_wo_offset = np.transpose(ds_wo_offset)
        features_w_offset = np.transpose(ds_w_offset)
        labels_gaussian_2d = np.transpose(ds_labels_gaussian_2d)
        
        map_image = RGB2Gray(np.array(Image.open(self.map_dir[0]))/255)
        map_reshape = np.repeat(map_image[np.newaxis,np.newaxis,:],features_w_offset.shape[0],axis=0)
          

        for i in range(len(self.root_dir)-1):
            file_name = self.root_dir[i+1]
            print('Loading '+file_name)
            f = h5py.File(file_name,'r')            
        
            read_labels_gaussian_2d = np.array(f.get('labels_gaussian_2d'), dtype=np.float32)
            read_wo_offset = np.array(f.get('features_wo_offset'), dtype=np.float32)
            read_w_offset = np.array(f.get('features_w_offset'), dtype=np.float32)
        
            ds_wo_offset = read_wo_offset[:,:,:,0:read_wo_offset.shape[3]:self.ds_step]
            ds_w_offset = read_w_offset[:,:,:,0:read_w_offset.shape[3]:self.ds_step]
            ds_labels_gaussian_2d = read_labels_gaussian_2d[:,:,0:read_labels_gaussian_2d.shape[2]:self.ds_step]
        
            features_wo_offset_get = np.transpose(ds_wo_offset)
            features_w_offset_get = np.transpose(ds_w_offset)
            labels_gaussian_2d_get = np.transpose(ds_labels_gaussian_2d)
            
            features_wo_offset = np.concatenate((features_wo_offset, features_wo_offset_get), axis=0)
            features_w_offset = np.concatenate((features_w_offset, features_w_offset_get), axis=0)
            labels_gaussian_2d = np.concatenate((labels_gaussian_2d, labels_gaussian_2d_get), axis=0)
            
            map_image = RGB2Gray(np.array(Image.open(self.map_dir[i+1]))/255)
            map_reshape_get = np.repeat(map_image[np.newaxis,np.newaxis,:],features_w_offset_get.shape[0],axis=0)
            map_reshape = np.concatenate((map_reshape, map_reshape_get), axis=0)
            
        print("features_wo_offset = ", features_wo_offset.shape)
        print("features_w_offset = ", features_w_offset.shape)
        print("labels_gaussian_2d_get = ", labels_gaussian_2d_get.shape)
        print("map_reshape = ", map_reshape.shape)     
            
        features_wo_offset_item = features_wo_offset[idx];
        features_w_offset_item = features_w_offset[idx];
        labels_gaussian_2d_item = labels_gaussian_2d[idx];
        map_reshape_item = map_reshape[idx];
        
        return (
            torch.from_numpy(features_wo_offset_item).to(dtype=torch.float32),
            torch.from_numpy(features_w_offset_item).to(dtype=torch.float32),
            torch.from_numpy(labels_gaussian_2d_item).to(dtype=torch.float32),
            torch.from_numpy(map_reshape_item).to(dtype=torch.float32)       
        )        

    def __len__(self):
        length = 4260
        return length


def load_data(filename,start,stop,step,ds_step):

    if opt_exp.location == "machine":
        print('Loading ',filename)
        f = h5py.File(filename,'r')
        
    if opt_exp.location == "server":    
        file_name = filename + '_split' + str(start) + '.h5'
        print('Loading ',file_name)
        f = h5py.File(file_name,'r')
    
    read_labels_gaussian_2d = np.array(f.get('labels_gaussian_2d'), dtype=np.float32)
    read_wo_offset = np.array(f.get('features_wo_offset'), dtype=np.float32)
    read_w_offset = np.array(f.get('features_w_offset'), dtype=np.float32)

    ds_wo_offset = read_wo_offset[:,:,:,0:read_wo_offset.shape[3]:ds_step]
    ds_w_offset = read_w_offset[:,:,:,0:read_w_offset.shape[3]:ds_step]
    ds_labels_gaussian_2d = read_labels_gaussian_2d[:,:,0:read_labels_gaussian_2d.shape[2]:ds_step]

    features_wo_offset = torch.tensor(np.transpose(ds_wo_offset), dtype=torch.float32)
    features_w_offset = torch.tensor(np.transpose(ds_w_offset), dtype=torch.float32)
    labels_gaussian_2d = torch.tensor(np.transpose(ds_labels_gaussian_2d), dtype=torch.float32)    
   
    if opt_exp.location == "server":    
        for i in range(start+1,stop+1,step):
            file_name = filename + '_split' + str(i) + '.h5'
            print('Loading '+file_name)
            f = h5py.File(file_name,'r')
            
            read_labels_gaussian_2d = np.array(f.get('labels_gaussian_2d'), dtype=np.float32)
            read_wo_offset = np.array(f.get('features_wo_offset'), dtype=np.float32)
            read_w_offset = np.array(f.get('features_w_offset'), dtype=np.float32)
    
            ds_wo_offset = read_wo_offset[:,:,:,0:read_wo_offset.shape[3]:ds_step]
            ds_w_offset = read_w_offset[:,:,:,0:read_w_offset.shape[3]:ds_step]
            ds_labels_gaussian_2d = read_labels_gaussian_2d[:,:,0:read_labels_gaussian_2d.shape[2]:ds_step]
    
            features_wo = torch.tensor(np.transpose(ds_wo_offset), dtype=torch.float32)
            features_w = torch.tensor(np.transpose(ds_w_offset), dtype=torch.float32)
            labels_gauss = torch.tensor(np.transpose(ds_labels_gaussian_2d), dtype=torch.float32)        
            
            features_wo_offset = torch.cat((features_wo_offset,features_wo),0)
            features_w_offset = torch.cat((features_w_offset,features_w),0)
            labels_gaussian_2d = torch.cat((labels_gaussian_2d,labels_gauss),0)

    	
    return features_wo_offset,features_w_offset,labels_gaussian_2d

def load_data_np(path):
    
    features = ["features_wo_offset.npy","features_w_offset.npy","labels_gaussian_2d.npy"]
    load_data = []
    
    for i in range(len(path)):
        for j in range(len(features)):
            print("Loading: ", features[j], " from ", path[i])
            file = path[i]+features[j]
            data = np.load(file)
            if i == 0:
                load_data.append(data)
            else:
                load_data[j] = np.concatenate((load_data[j],data),axis=0)
    
    features_wo_offset = torch.tensor(load_data[0], dtype=torch.float32)
    features_w_offset = torch.tensor(load_data[1], dtype=torch.float32)
    labels_gaussian_2d = torch.tensor(load_data[2], dtype=torch.float32)
    #features_wo_offset = load_data[0]
    #features_w_offset = load_data[1]
    #labels_gaussian_2d = load_data[2]
    
    return features_wo_offset,features_w_offset,labels_gaussian_2d

class features_wo_map_Dataset(Dataset):
    def __init__(self, features_wo_offset,features_w_offset,labels_gaussian_2d):

        self.features_wo_offset = features_wo_offset
        self.features_w_offset = features_w_offset
        self.labels_gaussian_2d = labels_gaussian_2d

    def __getitem__(self, i):
        features_wo_offset = self.features_wo_offset[i]
        features_w_offset = self.features_w_offset[i]
        labels_gaussian_2d = self.labels_gaussian_2d[i]

        return (
            torch.from_numpy(features_wo_offset).to(dtype=torch.float32),
            torch.from_numpy(features_w_offset).to(dtype=torch.float32),
            torch.from_numpy(labels_gaussian_2d).to(dtype=torch.float32)
        )

    def __len__(self):
        return self.labels_gaussian_2d.shape[0]

class features_with_map_Dataset(Dataset):
    def __init__(self, features_wo_offset,features_w_offset,labels_gaussian_2d,map_data):

        self.features_wo_offset = features_wo_offset
        self.features_w_offset = features_w_offset
        self.labels_gaussian_2d = labels_gaussian_2d
        self.map_data = map_data

    def __getitem__(self, i):
        features_wo_offset = self.features_wo_offset[i]
        features_w_offset = self.features_w_offset[i]
        labels_gaussian_2d = self.labels_gaussian_2d[i]
        map_data = self.map_data[i]

        return (
            torch.from_numpy(features_wo_offset).to(dtype=torch.float32),
            torch.from_numpy(features_w_offset).to(dtype=torch.float32),
            torch.from_numpy(labels_gaussian_2d).to(dtype=torch.float32),
            torch.from_numpy(map_data).to(dtype=torch.float32)
        )

    def __len__(self):
        return self.labels_gaussian_2d.shape[0]
    
	
def load_map_data(filename,start,stop,step,ds_step):
   
    file_name = filename + '_split' + str(start) + '.h5'
    print('Loading ',file_name)
    f = h5py.File(file_name,'r')
    
    read_confidence_maps = np.array(f.get('final_maps'), dtype=np.float32)
    read_ap = np.array(f.get('ap'), dtype=np.float32)
    read_map = np.array(f.get('map_image'), dtype=np.float32)

    confidence_maps = torch.tensor(np.transpose(read_confidence_maps), dtype=torch.float32)
    ap = torch.tensor(np.transpose(read_ap), dtype=torch.float32)
    original_map = torch.tensor(np.transpose(read_map), dtype=torch.float32)    
   
    if opt_exp.location == "server":    
        for i in range(start+1,stop+1,step):
            file_name = filename + '_split' + str(i) + '.h5'
            print('Loading '+file_name)
            f = h5py.File(file_name,'r')
            
            read_confidence_maps = np.array(f.get('labels_gaussian_2d'), dtype=np.float32)
            read_ap = np.array(f.get('features_wo_offset'), dtype=np.float32)
            read_map = np.array(f.get('features_w_offset'), dtype=np.float32)
    
            read_confidence_maps = torch.tensor(np.transpose(read_confidence_maps), dtype=torch.float32)
            read_ap = torch.tensor(np.transpose(read_ap), dtype=torch.float32)
            original_map = torch.tensor(np.transpose(read_map), dtype=torch.float32)        
            
            confidence_maps = torch.cat((confidence_maps,read_confidence_maps),0)
            ap = torch.cat((ap,read_ap),0)

    	
    return confidence_maps,ap,original_map


