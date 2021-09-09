#!/usr/bin/python
import torch
from utils import *
from Generators import *
from data_loader import *
from params import *


class Joint_Gen_Loc_Network():

    def initialize(self, generator, location, frozen_loc=True, frozen_gen=False):
        self.generator = generator
        self.location = location
        self.frozen_loc = frozen_loc
        self.frozen_gen = frozen_gen

    def set_input(self, input, loc_target, convert_gen=True):
        self.input = input
        self.loc_target = loc_target
        self.generator.set_data(self.input, self.input, convert=convert_gen)
    
    def save_networks(self, epoch):
        self.generator.save_networks(epoch)
        self.location.save_networks(epoch)

    def save_outputs(self):
        self.generator.save_outputs()
        self.location.save_outputs()
    
    def update_learning_rate(self):
        self.generator.update_learning_rate()
        
    def forward(self):
        self.generator.forward()
        self.location.set_data(self.generator.output, self.loc_target)
        self.location.forward()
    
    def test(self):
        self.generator.test()
        self.location.set_data(self.generator.output, self.loc_target)
        self.location.test()

    def backward(self):
        self.location.backward()
        self.generator.backward()
        
    def optimize_parameters(self):
        self.forward()
        self.backward()
        if not self.frozen_gen:
            self.generator.optimizer.step()
        if not self.frozen_loc:
            self.location.optimizer.step()
        self.generator.optimizer.zero_grad()
        self.location.optimizer.zero_grad()

    def eval(self):
        self.generator.forward()
        self.location.set_data(self.generator.output, self.loc_target)
        self.location.forward()

class Enc_Dec_Network():

    def initialize(self, opt, encoder, decoder, frozen_dec=False, frozen_enc=False, gpu_ids='1'):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.encoder = encoder
        self.decoder = decoder
        self.frozen_dec = frozen_dec
        self.frozen_enc = frozen_enc
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) # if self.gpu_ids else torch.device('cpu')
        # self.encoder.net = encoder.net.to(self.device)
        # self.decoder.net = decoder.net.to(self.device)

    def set_input(self, input, target, convert_enc=True, shuffle_channel=True):
        self.input = input.to(self.device)
        self.target = target.to(self.device)
        self.encoder.set_data(self.input, self.input, convert=convert_enc, shuffle_channel=shuffle_channel)
    
    def save_networks(self, epoch):
        self.encoder.save_networks(epoch)
        self.decoder.save_networks(epoch)

    def save_outputs(self):
        self.encoder.save_outputs()
        self.decoder.save_outputs()
    
    def update_learning_rate(self):
        self.encoder.update_learning_rate()
        
    def forward(self):
        self.encoder.forward()
        self.decoder.set_data(self.encoder.output, self.target)
        self.decoder.forward()
    
    def test(self):
        self.encoder.test()
        self.decoder.set_data(self.encoder.output, self.target)
        self.decoder.test()

    def backward(self):
        self.decoder.backward()
        # self.encoder.backward()
        
    def optimize_parameters(self):
        self.forward()
        self.backward()
        if not self.frozen_enc:
            self.encoder.optimizer.step()
        if not self.frozen_dec:
            self.decoder.optimizer.step()
        self.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()

    def eval(self):
        self.encoder.forward()
        self.decoder.set_data(self.encoder.output, self.target)
        self.decoder.forward()



class Enc_2Dec_Network():

    def initialize(self, opt , encoder, decoder, offset_decoder, frozen_dec=False, frozen_enc=False, gpu_ids='1'):
        print('initializing Encoder and 2 Decoders Model')
        self.opt = opt
        self.isTrain = opt.isTrain      
        self.encoder = encoder
        self.decoder = decoder
        self.offset_decoder = offset_decoder
        self.frozen_dec = frozen_dec
        self.frozen_enc = frozen_enc
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) # if self.gpu_ids else torch.device('cpu')
        # self.encoder.net = encoder.net.to(self.device)
        # self.decoder.net = decoder.net.to(self.device)
        self.results_save_dir = opt.results_dir

    def set_input(self, input, target ,offset_target ,convert_enc=True, shuffle_channel=True):
        # features_w_offset, labels_gaussian_2d, features_wo_offset
        # input,             target,             offset_target
        self.input = input.to(self.device)
        self.target = target.to(self.device)
        self.offset_target = offset_target.to(self.device)      
        self.encoder.set_data(self.input, self.input, convert=convert_enc, shuffle_channel=shuffle_channel)
    
    def save_networks(self, epoch):
        self.encoder.save_networks(epoch)
        self.decoder.save_networks(epoch)
        self.offset_decoder.save_networks(epoch)

    def save_outputs(self):
        self.encoder.save_outputs()
        self.decoder.save_outputs()
        self.offset_decoder.save_outputs()
    
    def update_learning_rate(self):
        self.encoder.update_learning_rate()
        self.decoder.update_learning_rate()
        self.offset_decoder.update_learning_rate()
        
    def forward(self):
        self.encoder.forward()
        self.decoder.set_data(self.encoder.output, self.target)
        self.offset_decoder.set_data(self.encoder.output, self.offset_target)
        self.decoder.forward()
        self.offset_decoder.forward()
    
    # Test the network once set into Evaluation mode!
    def test(self):      
        self.encoder.test()
        self.decoder.set_data(self.encoder.output, self.target)
        self.offset_decoder.set_data(self.encoder.output, self.offset_target)
        self.decoder.test()
        self.offset_decoder.test()

    def backward(self):
        self.decoder.backward()
        self.offset_decoder.backward()
        # self.encoder.backward()
        
    def optimize_parameters(self):
        self.forward()
        self.backward()
        if not self.frozen_enc:
            self.encoder.optimizer.step()
        if not self.frozen_dec:
            self.decoder.optimizer.step()
            self.offset_decoder.optimizer.step()
        self.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()
        self.offset_decoder.optimizer.zero_grad()

    # set the models to evaluation mode
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.offset_decoder.eval()


class Enc2_2Dec_Network():

    def initialize(self, opt , map_encoder, encoder, decoder, offset_decoder, frozen_dec=False, frozen_enc=False, gpu_ids='1'):
        print('initializing 2 Encoder and 2 Decoders Model')
        self.opt = opt
        self.isTrain = opt.isTrain
        self.map_encoder = map_encoder        
        self.encoder = encoder
        self.decoder = decoder
        self.offset_decoder = offset_decoder
        self.frozen_dec = frozen_dec
        self.frozen_enc = frozen_enc
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) # if self.gpu_ids else torch.device('cpu')
        # self.encoder.net = encoder.net.to(self.device)
        # self.decoder.net = decoder.net.to(self.device)
        self.results_save_dir = opt.results_dir            


    def set_input(self, maps, input, target ,offset_target ,convert_enc=True, shuffle_channel=True):
        self.maps = maps.to(self.device)
        self.input = input.to(self.device)
        self.target = target.to(self.device)
        self.offset_target = offset_target.to(self.device)
        #print("maps.shape = ", maps.shape)
        #print("input.shape = ", input.shape)
        #print("target.shape = ", target.shape)
        #print("offset_target.shape = ", offset_target.shape)
        self.map_encoder.set_data(self.maps, self.maps, convert=convert_enc, shuffle_channel=shuffle_channel)        
        self.encoder.set_data(self.input, self.input, convert=convert_enc, shuffle_channel=shuffle_channel)
    
    def save_networks(self, epoch):
        self.map_encoder.save_networks(epoch)  
        self.encoder.save_networks(epoch)
        self.decoder.save_networks(epoch)
        self.offset_decoder.save_networks(epoch)

    def save_outputs(self):
        self.map_encoder.save_outputs()
        self.encoder.save_outputs()
        self.decoder.save_outputs()
        self.offset_decoder.save_outputs()
    
    def update_learning_rate(self):
        self.map_encoder.update_learning_rate()
        self.encoder.update_learning_rate()
        self.decoder.update_learning_rate()
        self.offset_decoder.update_learning_rate()
        
    def forward(self):
        self.map_encoder.forward()
        self.encoder.forward()
        #print("self.map_encoder.output.shape = ", self.map_encoder.output.shape)
        #print("self.encoder.output.shape",self.encoder.output.shape)
        #x = torch.mul(self.map_encoder.output, self.encoder.output)
        #x = torch.add(self.map_encoder.output, self.encoder.output)
        #print("a0 = ", self.map_encoder.output[0,0,0,0])
        #print("b0 = ", self.encoder.output[0,0,0,0])
        #print("b1 = ", self.encoder.output[0,1,0,0])
        #print("x0 = ", x[0,0,0,0])
        #print("x1 = ", x[0,1,0,0])
        x = torch.cat((self.map_encoder.output, self.encoder.output), 1)
        #print("x.shape=", x.shape)
        self.decoder.set_data(x, self.target)
        self.offset_decoder.set_data(x, self.offset_target)
        #self.decoder.set_data(self.encoder.output, self.target)
        #self.offset_decoder.set_data(self.encoder.output, self.offset_target)
        self.decoder.forward()
        self.offset_decoder.forward()
    
    # Test the network once set into Evaluation mode!
    def test(self):
        self.map_encoder.test()        
        self.encoder.test()
        #x = torch.mul(self.map_encoder.output, self.encoder.output)
        #x = torch.add(self.map_encoder.output, self.encoder.output)
        x = torch.cat((self.map_encoder.output, self.encoder.output), 1)
        self.decoder.set_data(x, self.target)
        self.offset_decoder.set_data(x, self.offset_target)
        #self.decoder.set_data(self.encoder.output, self.target)
        #self.offset_decoder.set_data(self.encoder.output, self.offset_target)
        self.decoder.test()
        self.offset_decoder.test()

    def backward(self):
        self.decoder.backward()
        self.offset_decoder.backward()
        # self.encoder.backward()
        
    def optimize_parameters(self):
        self.forward()
        self.backward()
        if not self.frozen_enc:
            self.map_encoder.optimizer.step()            
            self.encoder.optimizer.step()
        if not self.frozen_dec:
            self.decoder.optimizer.step()
            self.offset_decoder.optimizer.step()
        self.map_encoder.optimizer.zero_grad()
        self.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()
        self.offset_decoder.optimizer.zero_grad()
        

    # set the models to evaluation mode
    def eval(self):
        self.map_encoder.eval()        
        self.encoder.eval()
        self.decoder.eval()
        self.offset_decoder.eval()
        
class Enc_2Dec_fuse_Network():

    def initialize(self, opt , encoder, decoder, offset_decoder, frozen_dec=False, frozen_enc=False, gpu_ids='1'):
        print('initializing Encoder and 2 Decoders Model')
        self.opt = opt
        self.isTrain = opt.isTrain      
        self.encoder = encoder
        self.decoder = decoder
        self.offset_decoder = offset_decoder
        self.frozen_dec = frozen_dec
        self.frozen_enc = frozen_enc
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) # if self.gpu_ids else torch.device('cpu')
        # self.encoder.net = encoder.net.to(self.device)
        # self.decoder.net = decoder.net.to(self.device)
        self.results_save_dir = opt.results_dir   

    def set_input(self, map_input, input, target ,offset_target ,convert_enc=True, shuffle_channel=True):
        # features_w_offset, labels_gaussian_2d, features_wo_offset
        # input,             target,             offset_target
        self.input = input.to(self.device)
        self.map_input = map_input.to(self.device)
        self.target = target.to(self.device)
        self.offset_target = offset_target.to(self.device)      
        self.encoder.set_data(self.input, self.map_input, self.input, self.map_input, convert=convert_enc, shuffle_channel=shuffle_channel)
    
    def save_networks(self, epoch):
        self.encoder.save_networks(epoch)
        self.decoder.save_networks(epoch)
        self.offset_decoder.save_networks(epoch)

    def save_outputs(self):
        self.encoder.save_outputs()
        self.decoder.save_outputs()
        self.offset_decoder.save_outputs()
    
    def update_learning_rate(self):
        self.encoder.update_learning_rate()
        self.decoder.update_learning_rate()
        self.offset_decoder.update_learning_rate()
        
    def forward(self):
        self.encoder.forward()
        self.decoder.set_data(self.encoder.output, self.target)
        self.offset_decoder.set_data(self.encoder.output, self.offset_target)
        self.decoder.forward()
        self.offset_decoder.forward()
    
    # Test the network once set into Evaluation mode!
    def test(self):      
        self.encoder.test()
        self.decoder.set_data(self.encoder.output, self.target)
        self.offset_decoder.set_data(self.encoder.output, self.offset_target)
        self.decoder.test()
        self.offset_decoder.test()

    def backward(self):
        self.decoder.backward()
        self.offset_decoder.backward()
        # self.encoder.backward()
        
    def optimize_parameters(self):
        self.forward()
        self.backward()
        if not self.frozen_enc:
            self.encoder.optimizer.step()
        if not self.frozen_dec:
            self.decoder.optimizer.step()
            self.offset_decoder.optimizer.step()
        self.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()
        self.offset_decoder.optimizer.zero_grad()

    # set the models to evaluation mode
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.offset_decoder.eval()        
