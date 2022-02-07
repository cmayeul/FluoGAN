#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import read_lif
import json
import torch
import numpy as np
from tifffile import imread
from torchvision.io import read_video
from pathlib import Path

def load_parameters(args_list, parser) : 
    #try if a parameter file is provided as a short or a long option
    try : 
        i = args_list.index("-p") + 1
        params_file = args_list[i]
    except IndexError : 
        print("no path provided for parameter file")
    except ValueError : 
        try : 
            i = args_list.index("--params_in") + 1
            params_file = args_list[i]
        except IndexError : 
            print("no path provided for parameter file")
        except ValueError : 
            #default value for parameters file path
            params_file = "parameters.json"
      
    #try to load parameters from file (provided path or default)
    try :
        #load json parameter file
        with open(params_file,"r") as f :
            args = argparse.Namespace(**json.load(f))
                
        print(f"parameters loaded from {args.params_in}")
        
    except FileNotFoundError :
        print(f"json parameters file '{params_file}' not found")
        args = argparse.Namespace()
        
    except json.decoder.JSONDecodeError as e : 
        print(f"invalid json file : {e}")
        args = argparse.Namespace()

    finally :
        #merge parameters from file and from cli
        args = parser.parse_args(args_list, args)
        
        #init working directory
        args.out  = Path(args.out)
        args.out.mkdir(exist_ok=True)
        
        #save params in a json file in out dir :
        with open(args.out / args.params_out,'w') as f :
            args.out = str(args.out)
            json.dump(args.__dict__ ,f, indent=2)
            args.out = Path(args.out)
            print(f"current parameters saved in {args.out / args.params_out}")
            
    return args

def load_data(args, device) :
    if args.input is None  :
        raise ValueError("No input")
    elif not Path(args.input).is_file() :
        raise ValueError("Invalid input file : {args.input}")
    
    extension = args.input.split('.')[-1]

    if extension == "lif" : 
        reader = read_lif.Reader(args.input)
        series = reader.getSeries()
        data = torch.stack([torch.from_numpy(img) \
                            for t,img in series[args.index].enumByFrame()])
      
    elif extension == "tif" : 
        data = imread(args.input)
        data = torch.from_numpy(data)
        
    else :
        data = read_video(args.input)[0][..., args.index]

    #take only the first n_img if this parameter is not None
    if args.n_images is not None :
        if args.n_images > len(data) : 
            raise ValueError(f"There are only {len(data)} images in this dataset but {args.n_images} are required from parameters")
        else : 
            data = data[:args.n_images]

    #send data to device and reshape it to pytorch standard (N, C, H, W)
    data = data.view(data.shape[0], 1, *data.shape[1:]).to(device)
    return data


def init_device(params) :
    if not torch.cuda.is_available() : 
        device = torch.device("cpu")
        print("cuda device unavailable")
    else :
        device = torch.device("cuda:0" if not params.cpu else "cpu")
    print(f"device selected : {device}")
    
    return device

def main(args, parser) : 
    params = load_parameters(args, parser)
    device = init_device(params)
    data   = load_data(params, device)
    
    
    print(f"loaded {len(data)} images from {params.input} with shape {tuple(data.shape[-2:])} to device {device}")
    print(f"output directory : {params.out}")
    
    print(data.shape)
    print(data.dtype)

if __name__ == "__main__" :
  
    parser = argparse.ArgumentParser()
    
    #input file : video or lif or tif image
    parser.add_argument("input", default = None,
                        help="input file : video or lif file or tiff image \
                        the image serie should be well aligned to yield good results")
    
    parser.add_argument("-i", "--index", type=int, default=0,
                        help="in case of lif input give the serie index \
                        in case of a video input give the channel index")
    parser.add_argument("-n_img", "--n_images", type=int, default=None,
                        help="take only the n_img first image from the input serie")
    parser.add_argument("-p", "--params_in", type=str, default="parameters.json",
                        help="load parameters from a json file in the output dir \
                        parameters are loaded in this order : \
                        1. CLI, 2. JSON file, 3. default")
    parser.add_argument("-o", "--params_out", type=str, default="parameters.json",
                        help="save current configuration in a json parameter file \
                        (creates or updates this file in the out dir)")
    parser.add_argument("-O", "--out", type=str, default="./", 
                        help="output directory, default : ./")
    
    #model for the generator
    parser.add_argument("-a", "--alpha", type=int, default=20, 
                        help="average number of photons per emitter")
    parser.add_argument("-w", "--kwidth", type=int, default=81, 
                        help="width of the convolution kernel")
    parser.add_argument("-s", "--ksigma", type=float, default=8.,
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
    
    main(sys.argv[1:], parser)