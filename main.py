#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import json
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, Normalize, PowerNorm
import numpy as np

from pathlib import Path, PosixPath, WindowsPath
from torchvision.io import read_video, write_video, read_image
from matplotlib.image import imsave
from tifffile import imwrite 

from Generator import Generator
from Discriminator import Discriminator
from train import train, ISTA_Optimizer, prox_l1, FISTA_Optimizer

torch.cuda.empty_cache()


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
            p = json.load(f)
            
            #change paths into absolute paths
            for k,v in p.items() :
                if k in ["input", "output", "parameters", "D", "G"] and v is not None :
                    p[k] = Path(v)
                    if not p[k].is_absolute() :
                        p[k] = file.parent / v
            
            params.update(p)
                
    #choose the right input data path from data_path if not None or from params
    if data_path is None : 
        if "input" in params : 
            data_path = params["input"]
        else : 
            raise ValueError("no path to input data")
    
    #load CLI arguments 
    params = parser.parse_args(args, argparse.Namespace(**params)).__dict__
    params["input"] = data_path
    
    #compute PSF width from provided parameters
    if params["ksigma"] is None :
        if params["px"] is None :
            raise ValueError("You should provide at least ksigma or \
                             px / FWHM or px / NA / wavelength to modelize PSF")
        elif params["FWHM"] is None :
            if params["wavelength"] is None or params["NA"] is None :
                raise ValueError("You should provide at least ksigma or \
                                 px / FWHM or px / NA / wavelength to modelize PSF")
            params["FWHM"] = int(0.61 * params["wavelength"] / params["NA"])
        params["ksigma"] = params["FWHM"] / 2.335 / params["px"] * params["undersampling"]
    
    return params

def save_parameters(params):
        
    #save parameters in a json file in output directory 
    params["output"].mkdir(exist_ok=True)
    
    def save_path(p): 
        try : 
            res = p.relative_to(params['output'])
        except :
            res = p.resolve()
        finally :
            return str(res)
    
    with open(params['output'] / 'parameters.json', "w") as f : 
        json.dump(params, f, indent=2, default=save_path)
        
    print(f"saved current parameters to {params['output'] / 'parameters.json'}")

def load_data(params, device) :
    if params["input"] is None  :
        raise ValueError("No input")
    elif not Path(params["input"]).is_file() :
        raise ValueError(f"Invalid input file : {params['input']}")
        
    extension = params["input"].suffix[1:]

    if extension == "lif" : 
        from read_lif import Reader as lifReader
        reader = lifReader(params["input"])
        series = reader.getSeries()
        data = torch.stack([torch.from_numpy(img) \
                            for t,img in series[params["index"]].enumByFrame()])
      
    elif extension == "tif" : 
        from tifffile import imread as tifread
        data = tifread(params["input"])
        data = torch.from_numpy(np.float_(data))
        
    elif extension == "czi" :
        from czifile import imread as cziread
        data = cziread(params['input'])
        data = torch.from_numpy(np.float_(data[0,:,0,:,:,params['index']]))
        
    elif extension in ["png", "jpeg", "JPG", "JPEG"] :
        #in case the input is an image, then generate a synthetic dataset
        xreal = read_image(str(params["input"]))[params['index'],...].to(device)
        data = gen_synthetic_data(xreal, params, device)
        
    else :
        data = read_video(str(params["input"]))[0][..., params["index"]]

    #take only the first n_img if this parameter is not None
    if params["n_images"] is not None :
        if params["n_images"] > len(data) : 
            raise ValueError(f"There are only {len(data)} images in this dataset but {params['n_images']} are required from parameters")
        else : 
            data = data[:params["n_images"]]

    #crop the data,reshape it to pytorch standard (N, C, H, W) and send it to device
    if params['crop'] is not None :
        xmin, ymin, xmax, ymax = params['crop']
        data = data[:, xmin:xmax, ymin:ymax]
        
    #check if this portion of data is not empty
    if (data == 0).all() :
        raise ValueError("this portion of data contains only zeros")
        
    #estimate alpha if not provided
    if params["alpha"] is None : 
        params['alpha'] = float((data.std(axis=0)**2 / data.mean(axis=0)).mean())
        
    data = data.view(data.shape[0], 1, *data.shape[1:])
    data = data.to(device).to(torch.float)
    
    return data


def init_device(params) :
    if not torch.cuda.is_available() : 
        device = torch.device("cpu")
        print("cuda device unavailable")
    else :
        device = torch.device("cuda:0" if not params["cpu"] else "cpu")
    print(f"device selected : {device}")
    
    return device


def gen_synthetic_data(xreal, params, device) : 
    """
    Generate a synthetic learning dataset from a ground truth image

    Parameters
    ----------
    xreal : numpy.ndarray(shape=(h,w))
        input image.

    Returns
    -------
    Y : torch.tensor(shape=(n,h,w)).
        the output dataset with n images
    """
    #save ground truth image : 
    if params["plot_results"] : 
        plt.imshow(xreal.detach().cpu())
        plt.colorbar(orientation = 'horizontal' if xreal.shape[0] < xreal.shape[1] else 'vertical')
        plt.title("xreal")
        plt.savefig(params["output"] / "xreal.png"); plt.close()
    
    #compute shape of output image 
    out_shape = (xreal.shape[0] // params["undersampling"], xreal.shape[1] // params["undersampling"])
    #resize yreal to length and width multiples of stride (to avoid rounding problems)
    xreal = xreal[:params["undersampling"] * out_shape[0], 
                  :params["undersampling"] * out_shape[1]]
    #translate xreal values from 0 to 10
    xreal -= xreal.min()
    xreal = xreal.to(float) / 10
    
    #simulate real data
    b0 = torch.zeros(*out_shape, device=device).float() #uniform backgroud
    #"real data" generator and values
    G0 = Generator(xreal.shape, 
                   params["kwidth"],
                   params["ksigma"],
                   params["undersampling"],
                   params["alpha"],
                   params["esigma"],
                   params["fft"],
                   x0=xreal.float(),
                   b0=b0).to(device)
    
    #check if n_img is not None (mandatory in case of dataset generation)
    if params["n_images"] is None : 
        raise ValueError("A non null n_images parameter is mandatory for training set generation")
    
    #generate fake training set
    Y = G0(params["n_images"])
    print(f"fake training set generated from ground truth {params['input']}")
    
    return Y[:,0,:,:]

def main(args, parsed_args, parser) :
    params = load_parameters(args, parsed_args, parser)
    device = init_device(params)
    data   = load_data(params, device)
    save_parameters(params)
    
    out = Path(params['output'])
    
    print(f"loaded {len(data)} images from {params['input']} with shape {tuple(data.shape[-2:])} to device {device}")
    print(f"output directory : {params['output']}")
    
    #init learning set 
    Y = data.view(data.shape[0], 1, data.shape[-2], data.shape[-1])
    # Y *= 10
    # Y[:,:,:20,:] = 0
    # Y[:,:,40:,:] = 0
    
    #normalize Y background :
    Ymin = Y.mean(axis=0).min()
    Y = (Y - Ymin).abs()
    
    #normalize learning rates 
    params['g_x_lr'] *= Y.shape[2] * Y.shape[3] * float(Y.mean())
    params['g_b_lr'] *= Y.shape[2] * Y.shape[3] * float(Y.mean())
    
    #initialize generator (x and b initizalized to 0 or from provided archive)
    if params['G'] is not None : 
        G = torch.load(params['G']).to(device)
    else :
        G = Generator(data.shape[2:],
                     kwidth         = params["kwidth"], 
                     ksigma         = params["ksigma"], 
                     undersampling  = params["undersampling"], 
                     alpha          = params["alpha"], 
                     esigma         = params["esigma"], 
                     fft            = params["fft"],
                     b0             = 0*float(Y.mean(axis=0).quantile(.25)) * torch.ones(Y[0,0].shape)
                     ).to(device)

    #print PSF
    if params['plot_results'] : 
        plt.imshow(G.conv.kernel[0,0].detach().cpu())
        plt.title(f"Gaussian PSF, ksigma = {params['ksigma']} px")
        plt.savefig(out / "PSF.png"); plt.close()

    #init discriminator (critic) or load it from archive
    if params['D'] is not None :
        D = torch.load(params['D']).to(device)
    else:
        ysim = G(1)
        D = Discriminator(ysim.shape[2:],
                      c           = params["c"], 
                      n_conv      = params["n_conv"],
                      conv_k_size = params["conv_k_size"], 
                      max_pool_k_size = params["max_pool_k_size"],
                      margin      = 1 #params["kwidth"] // (2 * params["undersampling"])
                      ).to(device)

    #init discriminator optimiser
    d_optimizer = torch.optim.Adam(D.parameters(), lr=params['d_lr'])
        
    #def the l2 fidelity term
    Ymean = Y.float().mean(axis=0)
    g_l2_func = lambda y : params['g_l2'] * ((y-Ymean)**2).sum() / float((Ymean**2).sum())
    
    #def the discriminator fidelity term
    g_D_func = lambda y : params['g_D'] * (1 - D(y)).mean() #* G.x.shape[0] * G.x.shape[1]
    
    #complete fidelity term (need a second generator to avoid collision with G)
    G2 = Generator(data.shape[2:],
                    kwidth         = params["kwidth"], 
                    ksigma         = params["ksigma"], 
                    undersampling  = params["undersampling"], 
                    alpha          = params["alpha"], 
                    esigma         = params["esigma"], 
                    fft            = params["fft"],
                    b0             = 0 * float(Y.mean(axis=0).quantile(.25)) * torch.ones(Y[0,0].shape)
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
        return x.abs().mean()
    
    #init FISTA generator optimizer
    prox_x = prox_l1(params['g_x_l1'] / float(Ymean.sum()))
    g_x_optimizer = FISTA_Optimizer(G.x,
                                    1 / params['g_x_lr'],
                                    f, g,
                                    1 / params['g_x_eta'],
                                    prox=prox_x, 
                                    backtrackingmax=params['g_x_backtrackingmax'])
    # g_x_optimizer = ISTA_Optimizer([G.x], [params['g_x_l1']], [params['g_x_lr']])
    g_b_optimizer = ISTA_Optimizer([G.bg.b],  [0,0], 
                                   [params['g_b_lr']])
    
    # start training
    g_losses, d_losses, x, b, sums  =  train(G, D, Y, 
                                       g_x_optimizer, g_b_optimizer, d_optimizer,
                                       params, out,
                                       g_l2_func, g_D_func)
    

    #save the pytorch models to restart learning later
    if params["save_models"] : 
        torch.save(G, out / 'G')
        torch.save(D, out / 'D')
    
    #plot ysim, yreal and final x
    if params["plot_results"] :
        ysim = G(1)[0,0].detach().cpu()
        yreal = Y[0,0].detach().cpu()
        ymax = max(ysim.max(), yreal.max())
        o = 'vertical' if yreal.shape[0] > yreal.shape[1] else 'horizontal'
        plt.imshow(ysim, vmax=ymax, vmin=0);plt.colorbar(orientation=o);plt.title("ysim")
        plt.savefig(out / "ysim.png") ; plt.close()
        plt.imshow(yreal, vmax=ymax, vmin=0);plt.colorbar(orientation=o);plt.title("yreal") 
        plt.savefig(out / "yreal.png") ; plt.close()
        #mg = params['kwidth'] // 2
        xmax = max(G.x.detach().max().cpu(), 
                   G.bg.b.detach().max().cpu())
        #norm = SymLogNorm(vmin=0, vmax=xmax, linthresh=0.01)
        #norm = Normalize(vmin=0, vmax=xmax, clip=True)
        norm_x = PowerNorm(vmin=0, vmax=xmax, gamma=.5)
        norm_b = PowerNorm(vmin=0, vmax=xmax, gamma=.5)
        plt.imshow(G.x.detach().cpu(), norm=norm_x);plt.colorbar(orientation=o);plt.title("x") 
        plt.savefig(out / "x.png") ; plt.close()
        
        plt.imshow(G.bg.b.detach().cpu(), norm=norm_b);plt.colorbar(orientation=o);plt.title("b") 
        plt.savefig(out / "b.png") ; plt.close()

        imsave(out / "x-img.png", G.x.detach().cpu())
        imsave(out / "b-img.png", G.bg.b.detach().cpu())
        imsave(out / "x-img-bw.png", G.x.detach().cpu(), cmap="binary")
        imsave(out / "b-img-bw.png", G.bg.b.detach().cpu(), cmap="binary")
        imsave(out / "x-img-wb.png", G.x.detach().cpu(), cmap="gray")
        imsave(out / "b-img-wb.png", G.bg.b.detach().cpu(), cmap="gray")
        imsave(out / "yreal-img-wb.png", Y[0,0].detach().cpu(), cmap="gray")
        
        imwrite(out / "x.tif", G.x.detach().cpu().numpy())
        imwrite(out / "b.tif", G.bg.b.detach().cpu().numpy())
        imwrite(out / "yreal.tif", Y[0,0].detach().cpu().numpy())
    
    #plot stepsizes (for adaptative learning rate backtracking algo)
    if params['plot_results'] :
        plt.step(torch.arange(params['n_epochs'] * params['n_generator']),
                  g_x_optimizer.stepsizes)
        plt.yscale("log")
        plt.title("evolution of the adaptative FISTA learning rate")
        plt.ylabel("learning rate")
        plt.xlabel("epochs")
        plt.savefig(out / "stepsizes.png"); plt.close()
    
    #plot learning curves
    if params["save_losses"] : 
        g_iter = torch.arange(len(g_losses['total']))
        for k,v in g_losses.items() : 
            plt.plot(g_iter, v, label=k)
        plt.legend();plt.title("generator loss")
        plt.yscale("log")
        plt.savefig(out / "g_loss_log.png"); plt.close()
        
        
    if params["save_losses"] and params["n_epochs"] * params["n_generator"] > 50 :
        for k,v in g_losses.items() : 
            plt.plot(g_iter[50:], torch.tensor(v)[50:], label=k)
        plt.legend();plt.title("generator loss (excepted steps from 0 to 50)")
        plt.xlabel("optimization steps (=n_generator * epochs)")
        plt.savefig(out / "g_loss.png"); plt.close()
    
    d_iter = torch.arange(len(d_losses['total']))
    for k,v in d_losses.items() : 
        plt.plot(d_iter, v, label=k)
    plt.legend();plt.title("discriminator loss")
    plt.xlabel("optimization steps (=n_discriminator * epochs)")
    plt.savefig(out / "d_loss.png"); plt.close()
    
    #plot sum of the signal 
    if params['plot_results']:
        sumx, sumb, sumysim, sumyreal = sums
        plt.plot(range(len(sumx)), sumx, label='sum of x pixel values')
        plt.plot(range(len(sumb)), sumb, label='sum of b pixel values')
        plt.plot(range(len(sumysim)), sumysim, label='sum of ysim pixel values')
        plt.plot(range(len(sumyreal)), sumyreal, label='sum of yreal pixel values')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.savefig(out / "sum_signal.png"); plt.close()
    
    #plot histograms of discriminator outputs 
    if params['plot_results'] :
        plt.hist([float(D(G(1))[0,0]) for x in range(len(Y))], histtype="step", label="real images")
        plt.hist([float(D(y.repeat(1,1,1,1))) for y in Y], histtype="step", label="simulated images")
        plt.title("estimated distribution of discriminator ouputs")
        plt.legend()
        plt.savefig(out / "hist.png"); plt.close()
 
        
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
    parser.add_argument("input", type=Path, default=Path("./"),
                        help="main input, may be : \
                            1. a video or lif file or tiff image serie \
                            2. a json parameter file containing path to input data \
                            3. a directory containing a file called 'parameter.json' \
                            Remark 1 : the image serie should be well aligned to yield good results \
                            Remark 2 : for a quick configuration you can provide either FWHM or wavelength and numerical aperture. Then configure the pixel size. The only tricky parameter you have to find manually is the generator l1 regularization parameter g_x_l1. Choose a small crop of the images and try to find the best regularization without discriminator (n_discriminator = 0 and g_D = 0). Then try to use the discriminator on the complete images. \
                            Remark 3 : this software creates or overwrite the output directory \
                                to save parameters and results (default : current directory)")
    
    parser.add_argument("-o", "--output", type=Path, default=Path("./"), 
                        help="output directory, default : ./")
    parser.add_argument("-p", "--params", type=Path, default=None,
                        help="load parameters from a json file \
                        parameters are loaded in this order : \
                        1. CLI, 2. JSON file, 3. default")
    parser.add_argument("-G", "--G", type=Path, default=None, 
                        help="path to a pytorch archive containing a generator module, \
                        useful to restart training without restarting from zero. \
                        Remark : in this case parameters like alpha, ksigma or \
                        kwidth have no effect. \
                        Default generator is initialized with constant b and x equals to zero")
    parser.add_argument("-D", "--D", type=Path, default=None, 
                        help="path to a pytorch archive containing a discriminator module, \
                        useful to restart training without restarting from zero. \
                        Default discriminator is initialized with random values \
                        and its shape depends on the input data.")
    
    parser.add_argument("-i", "--index", type=int, default=0,
                        help="in case of lif input give the serie index \
                        in case of a video input give the channel index")
    parser.add_argument("-n_img", "--n_images", type=int, default=None,
                        help="take only the n_img first image from the input serie")
    parser.add_argument("-x", "--crop", type=int, default=None, nargs=4,
                        help="select only a region of the input data \
                        give the crop window coordinates xmin ymin xmax ymax")

    
    #model for the generator
    parser.add_argument("-a", "--alpha", type=int, default=None, 
                        help="average number of photons per emitter")
    parser.add_argument("-w", "--kwidth", type=int, default=61, 
                        help="width of the convolution kernel")
    parser.add_argument("-s", "--ksigma", type=float, default=None,
                        help="width of the gaussian in conv kernel")
    parser.add_argument("--NA", type=float, default=1.15,
                        help="numerical aperture \
                        (to compute PSF width if no alpha nor FWHM provided)")
    parser.add_argument("--px", type=int, default=103,
                        help="size (in nm) for the physical pixels \
                        (to compute PSF width if no alpha nor FWHM provided)")
    parser.add_argument("--wavelength", type=int, default=509,
                        help="fluorophore emission wavelength \
                        (to compute PSF width if no alpha nor FWHM provided)")
    parser.add_argument("--FWHM", type=int, default=None, 
                        help="full width at half maximum (in nm) \
                        (to compute PSF width if no alpha provided)")
    parser.add_argument("-u", "--undersampling", type=int, default=6,
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
    parser.add_argument("-d_gp", type=float, default=100,
                        help="regularization param for grad penalty (WGAN-GP)")
    
    #training generator
    parser.add_argument("-g_D", type=float, default=1,
                        help="weight for discriminator distance")
    parser.add_argument("-g_l2", type=float, default=10,
                        help="weight for L2 distance")
    parser.add_argument("-g_x_l1", type=float, default=1,
                        help="param for L1 reg on x")
    parser.add_argument("-g_b_grad", type=float, default=100,
                        help="param for smooth reg on b")
    
    
    #training
    parser.add_argument("-n_d", "--n_discriminator", type=int, default=1,
                        help="number of discriminator update during 1 epoch")
    parser.add_argument("-n_g", "--n_generator", type=int, default=1,
                        help="number of generator update during 1 epoch")
    parser.add_argument("-n_e", "--n_epochs", type=int, default=5000,
                        help="number of epochs")
    parser.add_argument("-B", "--batch_size", type=int, default=32,
                        help="batch size for D and G training")
    parser.add_argument("-C", "--cpu", action="store_true",
                        help="use CPU instead of GPU")
    parser.add_argument("-F", "--fft", action="store_true",
                        help="use FFT to compute convolution with the PSF")
    
    #learning rates
    parser.add_argument("-d_lr", type=float, default=1e-6,
                        help="learning rate for the discriminator update")
    parser.add_argument("-g_x_lr", type=float, default=1,
                        help="learning rate for x (inverse of lipschitz constant L)")
    parser.add_argument("-g_x_backtrackingmax", type=int, default=1,
                        help="max number of backtracking steps per FISTA iteration")
    parser.add_argument("-g_b_lr", type=float, default=1,
                        help="learning rate for b")
    parser.add_argument("-g_x_eta", type=float, default=.95,
                        help="factor to reduce g_x_lr during FISTA backtracking iterations")
    
    #visualization parameters
    parser.add_argument("-l", "--save_losses", action="store_true",
                        help="save G loss (requires extra computation)")
    parser.add_argument("-m", "--save_models", action="store_true",
                        help="save G and D pytorch modules after training")
    parser.add_argument("-v", "--save_video", action="store_true",
                        help="save a video to show x convergence")
    parser.add_argument("-r", "--plot_results", action="store_true",
                        help="plot and save x and b as plt with their scale")
    parser.add_argument("--interactive", action="store_true", 
                        help="use interactive matplotlib windows to plot results")

    main(sys.argv[1:], parser.parse_args(), parser)
    #main(["/home/mayeul/Desktop/shift_correction"], parser.parse_args(["/home/mayeul/Desktop/shift_correction"]), parser)
