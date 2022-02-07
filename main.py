#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from torchvision.io import write_video, read_video, read_image
from Generator import Generator
from Discriminator import Discriminator
from train import train, MultipleOptimizer, ISTA_Optimizer, prox_l1, FISTA_Optimizer
import csv
import ast
from pathlib import Path
from default_params import default_adam_params, default_fista_params
import argparse
import read_lif


# we use GPU if available, otherwise CPU
device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def read_params(file, fista=False) : 
    """
    Reads parameters from a csv file. If file does not exist or is not a 
    valid csv file then a new file is created containing default parameters
    Default parameters are different depending on chosen algorithm fista or adam
    """
    params = {}
    default_params = default_fista_params if fista else default_adam_params
    
    if Path(file).is_file() :
        with open(file, 'r') as f :
            reader = csv.reader(f, skipinitialspace=True)
            
            #read provided parameters
            for line in reader : 
                if len(line) == 2 :
                    params[line[0]] = ast.literal_eval(line[1])
    
    with open(file,'a') as f : 
        writer = csv.writer(f)
        
        #complete the file with default params
        for p in default_params :
            if p not in params : 
                params[p] = default_params[p]
                writer.writerow([p, params[p]])
    
    return params

    
def main(Y, rep, conf = "parameters") : 
    """
    initializes Generator and Discriminator modules, 
    loads parameters from file, 
    instanciates the optimizers, 
    start the optimization process
    plot and save the results
    
    Parameters
    ----------
    Y : torch.tensor :
        its shape should be (L,n,m) with L the number of images of size n*m
    rep : the directory to store the results (will be created if not exists)
    conf : the parameters file (created and filled with default values if not 
           exists). lacking values will be appended at the end of the file.
    """
    
    #create directory and conf file if they not exist
    rep = Path(rep)
    rep.mkdir(exist_ok=True)
    params = read_params(rep / conf)
    
    #initialize generator
    G = Generator(out_shape,
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
                  c = 32, 
                  n_conv = 3,
                  conv_k_size = 5, 
                  max_pool_k_size = 2).to(device)

    #init learning set 
    Y = Y.view(Y.shape[0], 1, Y.shape[-2], Y.shape[-1])
    
    #init discriminator optimiser
    d_optimizer = torch.optim.Adam(D.parameters(), lr=params['d_lr'])
        
    #def the l2 fidelity term
    Ymean = Y.mean(axis=0)
    g_l2_func = lambda y : params['g_l2'] * ((y-Ymean)**2).sum()
    
    #def the discriminator fidelity term
    g_D_func = lambda y : params['g_D'] * (1 - D(y)).mean() * G.x.shape[0] * G.x.shape[1]
    
    #complete fidelity term (need a second generator to avoid collision with G)
    G2 = Generator(out_shape,
                    kwidth         = params["kwidth"], 
                    ksigma         = params["ksigma"], 
                    undersampling  = params["undersampling"], 
                    alpha          = params["alpha"], 
                    esigma         = params["esigma"]
                    ).to(device)
    G2.bg.b.data = G.bg.b.data
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
                                        params, rep,
                                        g_l2_func, g_D_func)

        
    #save the pytorch models to restart learning later
    if params["save_models"] : 
        torch.save(G, rep / 'G')
        torch.save(D, rep / 'D')
    
    #plot ysim, yreal and final x
    if params["plot_results"] :
        plt.imshow(G(1)[0,0].detach().cpu());plt.colorbar();plt.title("ysim")
        plt.savefig(rep / "ysim.png");plt.show()
        plt.imshow(Y[0,0].detach().cpu());plt.colorbar();plt.title("yreal") 
        plt.savefig(rep / "yreal.png");plt.show()
        plt.imshow(G.x.detach().cpu());plt.colorbar();plt.title("x") 
        plt.savefig(rep / "x.png");plt.show()
        plt.imshow(G.bg.b.detach().cpu());plt.colorbar();plt.title("b") 
        plt.savefig(rep / "b.png");plt.show()
        imsave(rep / "x-img.png", G.x.detach().cpu())
        imsave(rep / "b-img.png", G.bg.b.detach().cpu())
        imsave(rep / "x-img-bw.png", G.x.detach().cpu(), cmap="binary")
        imsave(rep / "b-img-bw.png", G.bg.b.detach().cpu(), cmap="binary")
        plt.hist([float(x) for x in D(G(len(Yval))).flatten()], histtype="step")
        plt.hist([float(x) for x in D(Yval).flatten()], histtype="step")
        plt.savefig(rep / "hist.png");plt.show()
    
    
    #plot learning curves
    if params["save_losses"] : 
        g_iter = torch.arange(len(g_losses['total']))
        for k,v in g_losses.items() : 
            plt.plot(g_iter, torch.tensor(v).log10(), label=k)
        plt.legend();plt.title("g_loss")
        plt.savefig(rep / "g_loss.png");plt.show()
        
    if params["save_losses"] and params["n_epochs"] * params["n_generator"] > 50 :
        for k,v in g_losses.items() : 
            plt.plot(g_iter[50:], torch.tensor(v)[50:], label=k)
        plt.legend();plt.title("g_loss")
        plt.savefig(rep / "g_loss_2.png");plt.show()
    
    d_iter = torch.arange(len(d_losses['total']))
    for k,v in d_losses.items() : 
        plt.plot(d_iter, torch.tensor(v).log10(), label=k)
    plt.legend();plt.title("d_loss")
    plt.savefig(rep / "d_loss.png");plt.show()
 
        
    #save video with x evolution
    if params["save_video"] :
        video = torch.stack(x)
        video = (video / video.max() * 255).to(torch.uint8) \
                                           .view(*video.shape,1) \
                                           .repeat(1,1,1,3)
        write_video(str(rep / 'video_apprentissage_x.mp4'), video.cpu(), fps=1)
        


if __name__ == "__main__" : 
    # #parameters 
    undersampling = 4
    alpha = 20
    ksigma = 8
    kwidth = 81
    esigma = 0.
    frames = 500
    
    #load lif file
    # reader = read_lif.Reader("../donnees_ostreopsis/ArgoLight.lif")
    # series = reader.getSeries()
    # data = torch.stack([torch.from_numpy(img) for t,img in series[0].enumByFrame()]).to(device)
    # Y = torch.zeros(500,1,50,630).to(device)
    # dy = 0
    # for i in range(500): 
    #     if i % 86 == 0 :
    #         dy -= 1
    #     Y[i,0] = data[i, 200:250, 20+dy:650+dy].float()
    # #Y = data[100:600,200:250,200:250].view(500,1,50,50).float()
    # Yval = data[600:650,200:250,20:650].view(50,1,50,630).float()
    # out_shape = Y.shape[2:]
    
    #load image
    xreal = read_image("../tests/xreal.png")[1,...].to(device)
    #normalize the image 
    xreal //= 10
    #compute shape of output image 
    out_shape = (xreal.shape[0] // undersampling, xreal.shape[1] // undersampling)
    #resize yreal to length and width multiples of stride (to avoid rounding problems)
    xreal = xreal[:undersampling * out_shape[0], :undersampling * out_shape[1]]
    #simulate real data
    b0 = 5*torch.zeros(40,40, device=device).float() #uniform
    # b0 = 1 + torch.arange(40, device=device).float().expand(40,40) / 40 #linear
    #"real data" generator and values
    G0 = Generator(out_shape,kwidth,ksigma,undersampling,alpha,esigma,x0=xreal.float(), b0=b0).to(device)
    Y = G0(frames)
    Yval = G0(frames)
    
    #use real data from video
    # Y = read_video("../donnees_ostreopsis/ostreo.m4v")[0][:500,50:-50,150:-150,0].to(device)
    # Y = Y.float().view(500,1,*Y.shape[1:])
    # Yval = read_video("../donnees_ostreopsis/ostreo.m4v")[0][500:550,50:-50,150:-150,0].to(device)
    # Yval = Yval.float().view(50,1,*Yval.shape[1:])
    # out_shape = Y.shape[2:]
    
    main(Y,"../tests/fantome14")
    
    
    