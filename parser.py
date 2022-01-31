#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json

parser = argparse.ArgumentParser()

#input file : video or lif or tif image
parser.add_argument("input", help="input file : video or lif file or tiff image \
                    the image serie should be well aligned to yield good results")

parser.add_argument("-s", "--serie", type=int, default=0,
                    help="in case of lif input give the serie number")
parser.add_argument("-j", "--json", type=str, default="parameters.json",
                    help="load parameters from json file, \
                    these may be overwritten by directly transmitted parameters")

#model for the generator
parser.add_argument("-a", "--alpha", type=int, default=20, 
                    help="average number of photons per emitter")
parser.add_argument("-w", "--kwidth", type=int, default=81, 
                    help="width of the convolution kernel")
parser.add_argument("-s", "--ksigma", type=int, default=8,
                    help="width of the gaussian in conv kernel")
parser.add_argument("-u", "--undersampling", type=int, default=4,
                    help="downsampling factor")
parser.add_argument("-e", "--esigma", type=float, default=0.,
                    help="standard deviation of the gaussian noise")

#training discriminator
parser.add_argument("-d_gp", type=float, default=10.,
                    help="regularization param for grad penalty (WGAN-GP)")

#training generator
parser.add_argument("-g_D", type=float, default=1e-2,
                    help="weight for discriminator distance")
parser.add_argument("-g_l2", type=float, default=1e-4,
                    help="weight for L2 distance")
parser.add_argument("-g_x_l1", type=float, default=1e-2,
                    help="param for L1 reg on x")
parser.add_argument("-g_b_grad", type=float, default=1.,
                    help="param for smooth reg on b")


#training
parser.add_argument("-n_d", "--n_discriminator", type=int, default=1,
                    help="number of discriminator update during 1 epoch")
parser.add_argument("-n_g", "--n_generator", type=int, default=1,
                    help="number of generator update during 1 epoch")
parser.add_argument("-n_e", "--n_epochs", type=int, default=1000,
                    help="number of epochs")
parser.add_argument("-B", "--batch_size", type=int, default=32,
                    help="batch size for D and G training")
parser.add_argument("-c", "--cpu", action="store_true",
                    help="use CPU instead of GPU")

#learning rates
parser.add_argument("-d_lr", type=float, default=1e-5,
                    help="learning rate for the discriminator update")
parser.add_argument("-g_x_lr", type=float, default=.25,
                    help="learning rate for x (inverse of lipschitz constant L)")
parser.add_argument("-g_b_lr", type=int, default=.001,
                    help="learning rate for b")
parser.add_argument("-g_x_eta", type=float, default=2.,
                    help="eta param to increase Lipschitz constant during FISTA step")

#visualization parameters
parser.add_argument("-l", "--save_losses", action="store_true",
                    help="save G loss (requires extra computation)")
parser.add_argument("-m", "--save_models", action="store_true",
                    help="save G and D pytorch modules after training")
parser.add_argument("-v", "--save_video", action="store_true",
                    help="ave a video to show x convergence")
parser.add_argument("-r", "--plot_results", action="store_true",
                    help="plot and save x and b as plt with their scale")

args = parser.parse_args()

try :
    #load json parameter file
    args_file = json.load(args['json'])
    
    #merge parameters from file and from cli
    for k,v in args_file : 
        if k not in args : 
            args[k] = v
            
    print(f"parameters from {args['json']} loaded")
    
except : 
    print("no valid json parameters file")

    

