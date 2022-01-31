#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt

def jaccard(img, ref, thres=10):
    """compute jaccard index of the two images img and ref
    It compares the support of both images defined as non null 
    pixels given a threshold thres"""
    
    inter = ((img > thres) * (ref > thres)).sum()
    union = ((img > thres) + (ref > thres)).sum()
    
    return inter / union


def PSNR(img, ref) : 
    """compute peaked signal noise ratio to compare an image img 
    with a reference image ref."""
    
    mse = ((img.float() - ref.float())**2).mean()
    return 10 * torch.log10(ref.max().float()**2 / mse)


def main(rep, xreal="../tests/xreal.png", epochs=500) : 
    """process a directory containing results 
    to return list containing jaccard indices and 
    PSNR function of the number of epochs"""
    
    xreal = read_image(xreal)[1]
    
    J = []
    psnr = []
    
    for i in range(50, epochs, 50) : 
        x = read_image(rep + f"x{i}-img.png")[1]
        J.append(jaccard(x, xreal, 10))
        psnr.append(PSNR(x, xreal))

    x = read_image(rep + "x-img.png")[1]
    J.append(jaccard(x, xreal, 10))
    psnr.append(PSNR(x, xreal))
    
    return J, psnr
    

if __name__ == "__main__" : 
    
    J0, psnr0 = main("../tests/l2_only/")
    J1, psnr1 = main("../tests/donnes_synthetiques_3/")
    J2, psnr2 = main("../tests/bg_0_bis/")
    J3, psnr3 = main("../tests/frames_250/")
    
    plt.plot(list(range(50, 550, 50)), J0, label="frames = 500, l2 only")
    plt.plot(list(range(50, 550, 50)), J1, label="frames = 500")
    #plt.plot(list(range(50, 550, 50)), J2, label="no background")
    plt.plot(list(range(50, 550, 50)), J3, label="frames = 250")
    plt.ylabel("Jaccard index")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()
    
    plt.plot(list(range(50, 550, 50)), psnr0, label="frames = 500, l2 only")
    plt.plot(list(range(50, 550, 50)), psnr1, label="frames = 500")
    #plt.plot(list(range(50, 550, 50)), psnr2, label="no background")
    plt.plot(list(range(50, 550, 50)), psnr3, label="frames = 250")
    plt.xlabel("epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.show()