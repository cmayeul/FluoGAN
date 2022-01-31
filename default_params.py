#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
default configuration for adam algorithm or ISTA / FISTA 
"""

default_adam_params = {
    #model for the generator
    'alpha'          : 10,       #average number of photons per emitter
    'kwidth'         : 51,       #width of the convolution kernel
    'ksigma'         : 8,        #width of the gaussian in conv kernel
    'undersampling'  : 4,        #downsampling factor
    'esigma'         : 0.,       #standard deviation of the gaussian noise
    #training discriminator
    'd_gp'           : 1,        #regularization param for grad penalty (WGAN-GP)
    #training generator
    'g_D'            : 1,        #weight for discriminator distance
    'g_l2'           : 1e-4,     #weight for L2 distance
    'g_x_l1'         : 1e-2,     #param for L1 reg on x
    'g_x_nneg'       : 1,        #param for non negativity reg on x
    'g_b_grad'       : 1,        #param for smooth reg on b
    'g_b_l2'         : 1e-4,     #param for L2 reg on b
    'g_b_nneg'       : 1,        #param for non negativity reg on b
    #training 
    'n_discriminator': 1,        #number of discriminator update during 1 epoch
    'n_generator'    : 10,       #number of generator update during 1 epoch
    'n_epochs'       : 250,      #number of epochs
    'batch_size'     : 32,       #batch size for D and G training
    'fista'          : False,    #type of optimizer
    #learning rates
    'd_lr'           : 1e-4,     #learning rate for the discriminator update
    'g_x_lr'         : .1,       #learning rate for x
    'g_b_lr'         : .01,      #learning rate for b
    #visualization parameters
    'save_losses'    : False,    #save G loss (requires extra computation)
    'save_models'    : True,     #save G and D pytorch modules after training
    'save_video'     : False,    #save a video to show x convergence
    'plot_results'   : True,     #plot and save x and b as plt with their scale
    } 
    
default_fista_params = {
    #model for the generator
    'alpha'          : 10,       #
    'kwidth'         : 51,       #
    'ksigma'         : 8,        #
    'undersampling'  : 4,        #
    'esigma'         : 0.,       #
    #training discriminator
    'd_gp'           : 1,        #
    #training generator
    'g_D'            : 10,       #10 times adam param
    'g_l2'           : 1e-4,     #
    'g_x_l1'         : 1e-2,     #regularization is included in prox operator
    'g_x_nneg'       : 0,        #nneg is already included in prox op
    'g_b_grad'       : 1,        #
    'g_b_l2'         : 1e-4,     #
    'g_b_nneg'       : 0,        #nneg is already included in prox op
    #training 
    'n_discriminator': 1,        #
    'n_generator'    : 10,       #
    'n_epochs'       : 250,      #
    'batch_size'     : 32,       #
    'fista'          : True,     #
    #learning rates
    'd_lr'           : 1e-4,     #
    'g_x_lr'         : 10,       #100 times higher than adam lr
    'g_b_lr'         : .01,      #
    #visualization parameters
    'save_losses'    : False,    #
    'save_models'    : True,     #
    'save_video'     : False,    #
    'plot_results'   : True,     #
    } 
