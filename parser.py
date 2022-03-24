#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import read_lif
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from pathlib import Path
from torchvision.io import read_video, write_video, read_image
from matplotlib.image import imsave

from Generator import Generator
from Discriminator import Discriminator
from train import train, MultipleOptimizer, ISTA_Optimizer, prox_l1, FISTA_Optimizer


def load_parameters(args, parsed_args, parser) : 
    #initialize a list containing paths to json parameters file to load
    params_src = []
    #initialize the dict for the different parameters
    params = {}
    #path to the input data
    data_path = None
    
    #check if the input exists
    path = Path(parsed_args.input)
    if not path.exists() :
        raise ValueError(f"input file or directory {path} not found")
    
    #if it is an existing directory: look for a parameter.json file inside
    elif path.is_dir() : 
        params_src.append(path / "parameters.json")
    
    else :         
        #if input is a json file then load parameters from it
        if path.suffix == ".json" : 
            params_src.append(path)

        else : 
            data_path = path
            
    #if an additional parameter file is provided, add it to the parameter sources
    if parsed_args.params is not None : 
        params_src.append(parsed_args.params)
      
    #load parameters from different sources in this order : 
    #1. default 2. input json file 3. additional parameter file 4. arguments 
    #each step overwrites the previous one
    for file in params_src : 
        print(f"loading parameters from {file}")
        with open(file,"r") as f : 
            params.update(json.load(f))
                
    #choose the right input data path from data_path if not None or from params
    if data_path is None : 
        if "input" in params : 
            data_path = Path(params["input"])
        else : 
            raise ValueError("no path to input data")
    
    #load CLI arguments 
    params = parser.parse_args(args, argparse.Namespace(**params)).__dict__
    params["input"] = data_path
        
    #save parameters in a json file in output directory 
    params["output"] = Path(params["output"])
    params["output"].mkdir(exist_ok=True)
    print(f"saving current parameters to {params['output'] / 'parameters.json'}")
    with open(params['output'] / 'parameters.json', "w") as f : 
        json.dump(params, f, default=lambda p : str(p.resolve()), indent=2)
    
    return params

def load_data(args, device) :
    if args["input"] is None  :
        raise ValueError("No input")
    elif not Path(args["input"]).is_file() :
        raise ValueError("Invalid input file : {args['input']}")
        
    extension = args["input"].suffix[1:]

    if extension == "lif" : 
        reader = read_lif.Reader(args["input"])
        series = reader.getSeries()
        data = torch.stack([torch.from_numpy(img) \
                            for t,img in series[args["index"]].enumByFrame()])
      
    elif extension == "tif" : 
        data = imread(args["input"])
        data = torch.from_numpy(data)
        
    elif extension in ["png", "jpeg", "JPG", "JPEG"] :
        #in case the input is an image, then generate a synthetic dataset
        data = gen_synthetic_data(args["input"])
        
    else :
        data = read_video(str(args["input"]))[0][..., args["index"]]

    #take only the first n_img if this parameter is not None
    if args["n_images"] is not None :
        if args["n_images"] > len(data) : 
            raise ValueError(f"There are only {len(data)} images in this dataset but {args['n_images']} are required from parameters")
        else : 
            data = data[:args["n_images"]]

    #send data to device and reshape it to pytorch standard (N, C, H, W)
    data = data.view(data.shape[0], 1, *data.shape[1:]).to(device)
    return data


def init_device(params) :
    if not torch.cuda.is_available() : 
        device = torch.device("cpu")
        print("cuda device unavailable")
    else :
        device = torch.device("cuda:0" if not params["cpu"] else "cpu")
    print(f"device selected : {device}")
    
    return device


def gen_synthetic_data(gt, params, device) : 
    """
    Generate a synthetic learning dataset from a ground truth image

    Parameters
    ----------
    gt : numpy.ndarray(shape=(h,w,c))
        input image.

    Returns
    -------
    Y : torch.tensor(shape=(n,h,w)).
        the output dataset with n images
    """
    #load image
    xreal = read_image(gt)[1,...].to(device)
    #normalize the image 
    xreal //= 10
    #compute shape of output image 
    out_shape = (xreal.shape[0] // params["undersampling"], xreal.shape[1] // params["undersampling"])
    #resize yreal to length and width multiples of stride (to avoid rounding problems)
    xreal = xreal[:params["undersampling"] * out_shape[0], 
                  :params["undersampling"] * out_shape[1]]
    #simulate real data
    b0 = torch.zeros(*out_shape, device=device).float() #uniform backgroud
    #"real data" generator and values
    G0 = Generator(xreal.shape, 
                   params["kwidth"],
                   params["ksigma"],
                   params["undersampling"],
                   params["alpha"],
                   params["esigma"],
                   x0=xreal.float(),
                   b0=b0).to(device)
    Y = G0(params["n_img"])
    return Y

def main(args, parsed_args, parser) :
    params = load_parameters(args, parsed_args, parser)
    device = init_device(params)
    data   = load_data(params, device)
    
    out = Path(params['output'])
    
    print(f"loaded {len(data)} images from {params['input']} with shape {tuple(data.shape[-2:])} to device {device}")
    print(f"output directory : {params['output']}")
    
    #initialize generator
    G = Generator(data.shape[2:],
                 kwidth         = params["kwidth"], 
                 ksigma         = params["ksigma"], 
                 undersampling  = params["undersampling"], 
                 alpha          = params["alpha"], 
                 esigma         = params["esigma"]
                 ).to(device)
    G.compute_phi()

    #init discriminator (critic)
    ysim = G(1)
    D = Discriminator(ysim.shape[2:],
                  c           = params["c"], 
                  n_conv      = params["n_conv"],
                  conv_k_size = params["conv_k_size"], 
                  max_pool_k_size = params["max_pool_k_size"]
                  ).to(device)

    #init learning set 
    Y = data.view(data.shape[0], 1, data.shape[-2], data.shape[-1])
    
    #init discriminator optimiser
    d_optimizer = torch.optim.Adam(D.parameters(), lr=params['d_lr'])
        
    #def the l2 fidelity term
    Ymean = Y.float().mean(axis=0)
    g_l2_func = lambda y : params['g_l2'] * ((y-Ymean)**2).sum()
    
    #def the discriminator fidelity term
    g_D_func = lambda y : params['g_D'] * (1 - D(y)).mean() * G.x.shape[0] * G.x.shape[1]
    
    #complete fidelity term (need a second generator to avoid collision with G)
    G2 = Generator(data.shape[2:],
                    kwidth         = params["kwidth"], 
                    ksigma         = params["ksigma"], 
                    undersampling  = params["undersampling"], 
                    alpha          = params["alpha"], 
                    esigma         = params["esigma"]
                    ).to(device)
    G2.bg.b.data = G.bg.b.data
    
    #smooth term to minimize
    def f(x) : 
        G2.x.data = x
        G2.compute_phi()
        ysim = G2(params['batch_size'])
        return g_l2_func(ysim) + g_D_func(ysim)
    
    #and L1 regularization
    def g(x):
        return x.abs().sum()
    
    #init FISTA generator optimizer
    prox_x = prox_l1(params['g_x_l1'])
    g_x_optimizer = FISTA_Optimizer(G.x,
                                    1 / params['g_x_lr'],
                                    f, g,
                                    params['g_x_eta'] ,
                                    prox=prox_x)
    g_b_optimizer = ISTA_Optimizer([G.bg.b],  [0], [params['g_b_lr']])
    
    # start training
    g_losses, d_losses, x, b  =  train(G, D, Y, 
                                        g_x_optimizer, g_b_optimizer, d_optimizer,
                                        params, out,
                                        g_l2_func, g_D_func)
    

    #save the pytorch models to restart learning later
    if params["save_models"] : 
        torch.save(G, out / 'G')
        torch.save(D, out / 'D')
    
    #plot ysim, yreal and final x
    if params["plot_results"] :
        plt.imshow(G(1)[0,0].detach().cpu());plt.colorbar();plt.title("ysim")
        plt.savefig(out / "ysim.png");plt.show()
        plt.imshow(Y[0,0].detach().cpu());plt.colorbar();plt.title("yreal") 
        plt.savefig(out / "yreal.png");plt.show()
        plt.imshow(G.x.detach().cpu());plt.colorbar();plt.title("x") 
        plt.savefig(out / "x.png");plt.show()
        plt.imshow(G.bg.b.detach().cpu());plt.colorbar();plt.title("b") 
        plt.savefig(out / "b.png");plt.show()
        imsave(out / "x-img.png", G.x.detach().cpu())
        imsave(out / "b-img.png", G.bg.b.detach().cpu())
        imsave(out / "x-img-bw.png", G.x.detach().cpu(), cmap="binary")
        imsave(out / "b-img-bw.png", G.bg.b.detach().cpu(), cmap="binary")
        plt.hist([float(x) for x in D(G(len(Y))).flatten()], histtype="step")
        plt.hist([float(x) for x in D(Y).flatten()], histtype="step")
        plt.savefig(out / "hist.png");plt.show()
    
    
    #plot learning curves
    if params["save_losses"] : 
        g_iter = torch.arange(len(g_losses['total']))
        for k,v in g_losses.items() : 
            plt.plot(g_iter, torch.tensor(v).log10(), label=k)
        plt.legend();plt.title("g_loss")
        plt.savefig(out / "g_loss.png");plt.show()
        
    if params["save_losses"] and params["n_epochs"] * params["n_generator"] > 50 :
        for k,v in g_losses.items() : 
            plt.plot(g_iter[50:], torch.tensor(v)[50:], label=k)
        plt.legend();plt.title("g_loss")
        plt.savefig(out / "g_loss_2.png");plt.show()
    
    d_iter = torch.arange(len(d_losses['total']))
    for k,v in d_losses.items() : 
        plt.plot(d_iter, torch.tensor(v).log10(), label=k)
    plt.legend();plt.title("d_loss")
    plt.savefig(out / "d_loss.png");plt.show()
 
        
    #save video with x evolution
    if params["save_video"] :
        video = torch.stack(x)
        video = (video / video.max() * 255).to(torch.uint8) \
                                           .view(*video.shape,1) \
                                           .repeat(1,1,1,3)
        write_video(str(out / 'video_apprentissage_x.mp4'), video.cpu(), fps=1)
        

if __name__ == "__main__" :
  
    parser = argparse.ArgumentParser()
    
    #input file : video or lif or tif image
    parser.add_argument("input", type=str,
                        help="main input, may be : \
                            1. a video or lif file or tiff image serie \
                            2. a json parameter file containing path to input data \
                            3. a directory containing a file called 'parameter.json' \
                            Remark 1 : the image serie should be well aligned to yield good results \
                            Remark 2 : this software creates or overwrite the output directory \
                                to save parameters and results (default : current directory)")
    
    parser.add_argument("-o", "--output", type=str, default="./", 
                        help="output directory, default : ./")
    
    parser.add_argument("-i", "--index", type=int, default=0,
                        help="in case of lif input give the serie index \
                        in case of a video input give the channel index")
    parser.add_argument("-n_img", "--n_images", type=int, default=None,
                        help="take only the n_img first image from the input serie")
    parser.add_argument("-p", "--params", type=str, default=None,
                        help="load parameters from a json file \
                        parameters are loaded in this order : \
                        1. CLI, 2. JSON file, 3. default")
    
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
    
    #Discriminator construction
    parser.add_argument("--c", type=int, default=32,
                        help="number of channels in the first conv layer \
                        (doubles after each next layer")
    parser.add_argument("--n_conv", type=int, default=3,
                        help="number of convolutional layers for Discriminator")
    parser.add_argument("--conv_k_size", type=int, default=5, 
                        help="kernel size for conv layers")
    parser.add_argument("--max_pool_k_size", type=int, default=2,
                        help="kernel size for max pool layer")
    
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
    parser.add_argument("-C", "--cpu", action="store_true",
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

    #main(sys.argv[1:], parser.parse_args(), parser)
    main(["test"], parser.parse_args(["test"]), parser)