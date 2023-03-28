#!/usr/bin/python
'''
Defines the Encoders and Decoder that make up the
Enc+Dec/Enc+2Dec model defined in DLoc
Idea and code originally from fstin Johnson's architecture.
https://github.com/jcjohnson/fast-neural-style/
Code base expanded from pix2pix Phillip Isola's implementation
https://github.com/phillipi/pix2pix

You can add your costum network building blocks here to test various other architectures.
'''
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import functools
import numpy as np
import math
from params import *


# Tha base Encoder function defined for the DLoc's Encoder
class ResnetEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='zero'):
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(input_nc, input_nc, kernel_size=7, padding=3,
                           bias=use_bias),
                 norm_layer(input_nc),
                 nn.Tanh()]

        model += [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# The base Decoder function defined for the DLoc's Decoders.
# Depending upon the ModelADT wraper around the decoder,
# the decoder would either be a Location Decoder or a Consistency decoder.
# For more details refer to ModelADT.py wrapper implementation and params.py
class ResnetDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='zero', encoder_blocks=6):
        assert(n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        n_downsampling = 2

        for i in range(n_downsampling):
            mult = 2**i

        mult = 2**n_downsampling
        for i in range(n_blocks):
            if i < encoder_blocks:
                continue
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        '''
        Experiemental adaptive network creation:
        Even numbers of the image inputs make the upconversion after downsampling be less in an axis by one pixel
        The following piece of code ensure to add these additional single pixels in the convTranspose2d
        as output_padding parameters
        '''
        # pad_vals = np.zeros([n_downsampling,2])
        # x_len = len(np.arange(opt_exp.label_Xstart,opt_exp.label_Xstop,opt_exp.label_Xstep))
        # y_len = len(np.arange(opt_exp.label_Ystart,opt_exp.label_Ystop,opt_exp.label_Ystep))
        # pad_vals = []
        # for i in range(n_downsampling):
        #     pad_vals.append([0, 0])
        #     if(x_len%2 == 0):
        #         pad_vals[i][0] = 1
        #     if(y_len%2 == 0):
        #         pad_vals[i][1] = 1
        #     x_len = int(np.ceil(x_len/2))
        #     y_len = int(np.ceil(y_len/2))

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=[3,3])]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        #print("decoder.input = ", input.shape)
        return self.model(input)
# In[3]:


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.25)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

#Define classification CNN model
class ResnetClassifier(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='zero', encoder_blocks=6):
        assert(n_blocks >= 0)
        super(ResnetClassifier, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i

        mult = 2**n_downsampling
        for i in range(n_blocks):
            if i <= encoder_blocks:
                continue
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model += [nn.Conv2d(ngf * mult, 32, kernel_size=3,
                            stride=1, padding=1, bias=use_bias),
                  nn.ReLU()]
        model += [nn.Flatten()]
        linear_shape = int(np.prod((opt_exp.batch_size, 32, 41, 91))/opt_exp.batch_size)
        model += [nn.Linear(linear_shape, 256), nn.ReLU()]
        model += [nn.Linear(256, 32), nn.ReLU()]
        model += [nn.Linear(32, 1)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        #print("decoder.input = ", input.shape)
        return self.model(input)

########################################################################################################################
#Define Classification head
class LocationClassifier(nn.Module):
    def __init__(self, input_nc, ouput_nc, n_hidden = 1, n_perc = [1024], use_droput=False, padding_type='zero'):
        assert(n_hidden>0)
        assert(len(n_perc)>=1)
        super(LocationClassifier, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Linear(input_nc, n_perc[0]),
                 norm_layer(n_perc[0]),
                 nn.ReLU(True),
                 nn.Dropout(p=0.5)]
        if(len(n_perc)>1):
            for i, np in enumerate(n_perc[1:]):
                model += [nn.Linear(n_perc[i], n_perc[i+1]),
                          norm_layer(n_perc[i+1]),
                          nn.ReLU(True)]
                if use_dropout:
                    model += [nn.Dropout(p=0.2)]

        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

######################################################################################################################
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

#####################################DenseNet3#################################################################
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2, stride=2, padding=1)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        x_, y_ = 315, 401
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(num_classes, in_planes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        x_, y_ = int(math.floor((x_-1)/2+1)), int(math.floor((y_-1)/2+1))
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        x_, y_ = int(math.floor((x_-1)/2+1)), int(math.floor((y_-1)/2+1))
        
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        x_, y_ = int(math.floor(x_/2+1)), int(math.floor(y_/2+1))
        # 2nd block
        self.block2 = DenseBlock(n*2, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+2*n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        x_, y_ = int(math.floor((x_)/2+1)), int(math.floor((y_)/2+1))
        # 3rd block
        self.block3 = DenseBlock(n*4, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+4*n*growth_rate)
        self.trans3 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        x_, y_ = int(math.floor((x_)/2+1)), int(math.floor((y_)/2+1))

        # 4th block
        self.block4 = DenseBlock(n*3, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+3*n*growth_rate)
        # global average pooling and classifier
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=7, stride=1, padding = 3)
        # print(x_,y_)
        self.n_up = int(x_*y_)
        self.fc1 = nn.Linear(self.n_up*in_planes, self.n_up)
        self.bn3 = nn.BatchNorm1d(self.n_up)
        self.fc_r = nn.Linear(self.n_up, num_classes)
        self.fc_th = nn.Linear(self.n_up, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        
        out = self.pool1(self.bn1(self.relu(self.conv1(x))))
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.trans3(self.block3(out))
        out = self.block4(out)
        out = self.relu(self.bn2(out))
        out = self.pool2(out)
        # print("Before Reshaping", out.shape)
        out = out.view(-1, self.n_up*self.in_planes)
        # print("N_up is", self.n_up, "IN_planes is", self.in_planes)
        # print("After reshaping", out.shape)
        out = self.relu(self.bn3(self.fc1(out)))
        return self.tanh(self.fc_th(out)), self.relu(self.fc_r(out))

def RetinaNetPreTrained():
    model_conv = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model = list(model_conv.head.regression_head.children())
    model.pop()
    model.append(nn.Conv2d(256, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    model += [nn.Flatten()]
    linear_shape = np.prod((4, 79, 101))
    model += [nn.Linear(linear_shape, 256), nn.ReLU()]
    model += [nn.Linear(256, 32), nn.ReLU()]
    model += [nn.Linear(32, 1)]
    model += [nn.Sigmoid()]
    return nn.Sequential(*model)

class AoAClassifier(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(AoAClassifier, self).__init__()

        model = []
        model += [nn.Conv2d(256, 32, kernel_size=3,
                            stride=1, padding=1, bias=False), nn.ReLU()]
        model += [nn.Flatten()]
        linear_dim = np.prod((32, 79, 101))
        model += [nn.Linear(linear_dim, 512), nn.ReLU()]
        model += [nn.Linear(512, 512), nn.ReLU()]
        model += [nn.Linear(512, 180)]
        if opt_decoder.loss_type == "Cross_Entropy":
            model += [nn.Sigmoid()]
        elif opt_decoder.loss_type == "KL_Div":
            model += [nn.Sigmoid()]
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return self.model(input)

        


