#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Disciminator network :  4 Conv2d layers with ReLu activation and 
    max pooling then 2 fully connected layers
    """
    def __init__(self, 
                 input_shape:tuple, 
                 c:int = 16, 
                 n_conv:int = 3, 
                 n_features:int = 10,
                 max_pool_k_size:int = 4, 
                 conv_k_size:int = 3, 
                 margin:int = 0):
        """
        

        Parameters
        ----------
        input_shape : TYPE
            DESCRIPTION.
        c : TYPE, optional
            DESCRIPTION. The default is 16.
        n_conv : TYPE, optional
            DESCRIPTION. The default is 3.
        n_features : TYPE, optional
            DESCRIPTION. The default is 10.
        """
        
        super().__init__()
        
        self.margin = margin
        
        self.convs = nn.ModuleDict()
        
        #add conv layers
       
        self.convs['conv0'] = nn.Sequential(
                            nn.Conv2d(1,c,kernel_size=conv_k_size),
                            nn.MaxPool2d(kernel_size=max_pool_k_size),
                            nn.LeakyReLU())

        for i in range(n_conv-1) :
            self.convs[f'conv{i+1}'] = nn.Sequential(
                            nn.Conv2d(c*2**i,c*2*(i+1),kernel_size=conv_k_size),
                            nn.MaxPool2d(kernel_size=max_pool_k_size),
                            nn.LeakyReLU())
        
        # try conv layers on fake data to get output size
        xf = torch.ones((1,1,*input_shape))
        xf = xf[:,:,margin:-margin, margin:-margin]
        xf = self.conv(xf)
        output_shape = torch.tensor(xf.shape[1:]).prod()
                        
        #add fully connected and linear layer
        self.fully = nn.Sequential(
                            nn.Linear(output_shape,n_features),
                            nn.ReLU())
        
        self.linear = nn.Linear(n_features,1)
        self.sigmoid = nn.Sigmoid()
        
        
        # initializing the weights with random values and biaises with zeros
        for m in self.modules() :
            if 'Conv' in m.__class__.__name__ or 'Linear' in m.__class__.__name__:
                m.weight.data.normal_(0., 0.01)
                m.bias.data.fill_(0.01)
        
    
    def conv(self, x):
        """
        Help function for forward method : applies only conv layers
        """
        for k,v in self.convs.items() : 
            x = v(x)
        return x
    
    def forward(self, x):
        #remove margin : 
        x = x[:,:,self.margin:-self.margin,self.margin:-self.margin]
        #conv layers
        x = self.conv(x)
        #reshape then fully connected layer
        x = x.flatten(start_dim=1)
        x = self.fully(x)
        #output layer
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
