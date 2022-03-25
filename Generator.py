#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from fft_conv import fft_conv


class Generator(nn.Module):
    """
    Generator class defined by the aquisition model 
    with x the super resolution image we search as parameter
    """
    
    def __init__(self, 
                 out_shape:tuple, 
                 kwidth:int = 51, 
                 ksigma:float = 5., 
                 undersampling:int = 4, 
                 alpha:int = 12, 
                 esigma:float = 0.5, 
                 x0:torch.Tensor = None, 
                 b0:torch.Tensor = None) -> None :
        """
        

        Parameters
        ----------
        
        """
        super().__init__()
        
        #initialisation of x and b
        if x0 is None : 
            h,w = out_shape
            x0 = torch.ones((h*undersampling + 2*(kwidth //2), w*undersampling + 2*(kwidth//2)))
            
        if b0 is None :
            b0 = torch.zeros(out_shape, device=x0.device, dtype=x0.dtype)
        
        #Other parameters
        self.alpha = alpha
        
        #The layers for simulator
        self.conv = FFTConv2dConstKernel(kwidth, ksigma, undersampling, x0.dtype, x0.device) #diffraction
        self.relu = nn.ReLU() 
        self.bg = Background(b0) #out of focus emitters
        self.poisson = PoissonProcess(alpha) #emission
        self.noise = GaussianNoise(esigma) #acquisition noise
        self.relu2 = nn.ReLU()
        
        #The variables we want to optimize on
        self.x = nn.Parameter(x0, requires_grad=True)
        self.compute_phi()
        
    def compute_phi(self) : 
        """
        precompute the first steps of computation because they do not 
        change at each evaluation.
        """
        phi = self.x.view(1,1,*self.x.shape)
        phi = self.conv(phi)
        phi = self.relu(phi)
        phi = self.bg(phi)
        phi = self.relu(phi)
        self.phi = phi
        
    def to(self, *args, **kargs) : 
        """ 
        rewrite the general "to" method to update phi computation
        """
        res = super().to(*args,**kargs)
        self.compute_phi()
        return res
        
    def update_x(self,x) :
        self.x.data = x
        self.compute_phi()
        
    def forward(self, frames=10):
        y = self.phi.repeat(frames,1,1,1)
        y = self.poisson(y)
        y = self.noise(y)
        y = self.relu2(y)
        return y
    
    def compute_grad(self, f, n_exp_estimate=10): 
        """
        Compute an estimate of the gradient of the expectancy of f(ysim) where 
        ysim is the result of forward pass after poisson sampling. 
        Then it updates self.x.grad

        Parameters
        ----------
        f : y -> c
            we want to estimate gradient of f(y) with regard to self.x
        n_exp_estimate : int
            number of realizations to estimate expectancy with empirical average.
        """
        #generate n_exp_estimate simulated images as references
        x = self.x.view(1,1,*self.x.shape)
        phi = self.relu(self.conv(x))
        y = self.noise(self.poisson(phi.repeat(n_exp_estimate,1,1,1)))
        
        ref = f(y)
        n,_,h,w = y.shape
        
        grad = torch.zeros_like(y[0,0])
        
        #compute gradient for each pixel i as mean(f_i(y) - f(y)) * alpha
        for i in range(h) : 
            #print((100*i)//h, end=" ")
            for j in range(w) : 
                y[:,0,i,j] += 1
                grad[i,j] = float((f(y) - ref).mean())
                y[:,0,i,j] -= 1
        
        grad *= self.alpha
        
        #if self.x does not already have a gradient computed
        if self.x.grad is None : 
            self.x.grad = torch.zeros_like(self.x.data)
        
        #Multiply with jacobian matrix containing derivatives of y wrt self.x
        self.x.grad += torch.autograd.grad(outputs=phi[0,0], 
                                           inputs=self.x, 
                                           grad_outputs=grad)[0]
        
        
    def compute_grad_2(self, f, ysim, inputs): 
        """
        alternative faster way to compute grad based on pytorch autograd
        Instead of generating several times a batch of simulated images 
        you can reuse the same ysim batch several times when you are 
        estimating derivatives with different f functions
        
        Be careful : ysim should be computed with current self.phi to 
        give relevant gradient
        """
        #separate sim images from upstream grad computation tree
        y = ysim.detach_().requires_grad_()
        
        #compute associated costs
        c = f(y)
        
        #compute grad c = f(y) wrt y
        c.backward(gradient = torch.ones_like(c), inputs=y)
        
        #estimate expectancy of grad f(y) wrt phi
        grads_mean = self.alpha * y.grad.mean(axis = 0, keepdim = True)
        
        #multiply with jacobian matrix containing derivatives of phi wrt x and b
        self.phi.backward(gradient = grads_mean, 
                          retain_graph = True, 
                          inputs = inputs)
        

class Background(nn.Module):
    """
    This layer introduces a smooth background to images
    It aims to model out of focus sources
    """
    
    def __init__(self, b0):
        """ 
        To build the background layer you have to provide an initial tensor
        b0 for background. It should have the same dtype, device like the 
        signal estimate x. The size should be divided by the undersampling 
        ratio. 
        """
        super().__init__()
        
        self.b = nn.Parameter(b0, requires_grad=True)
        
        #def laplacian filter 
        laplacian = torch.tensor([[[[ 0,-1, 0],
                                    [-1, 4,-1],
                                    [ 0,-1, 0]]]],
                                 device = self.b.device,
                                 dtype = self.b.dtype)
        self.laplacian = nn.Parameter(laplacian, requires_grad=False)
        
        sx = torch.tensor([[[[ -1, 1]]]],
                                 device = self.b.device,
                                 dtype = self.b.dtype)
        self.sx = nn.Parameter(sx, requires_grad=False)
        
        sy = torch.tensor([[[[ -1 ],
                             [  1 ]]]],
                                 device = self.b.device,
                                 dtype = self.b.dtype)
        self.sy = nn.Parameter(sy, requires_grad=False)
        
        self.pad = nn.ReplicationPad2d(1)
        
    def forward(self,x):
        return x + self.b.expand(1,1,-1,-1)
    
    def spatial_grad_norm_derivative(self):
        b = self.b.detach()
        b = b.expand(1,1,-1,-1)
        b = self.pad(b)
        
        return 2 * nn.functional.conv2d(b, self.laplacian)
    
    def spatial_grad_norm(self):
        """
        Computes the squared 2 norm of the spatial gradient of the image b
        It can be used as a regularization parameter to force b being smooth
        """
        b = self.b.expand(1,1,-1,-1)
        
        gx = nn.functional.conv2d(b, self.sx, padding='same')
        gy = nn.functional.conv2d(b, self.sy, padding='same')
        
        return gx**2 + gy**2

        

class Conv2dConstKernel(nn.Module):
    """ 
    A conv layer with constant gaussian kernel to modelize diffraction
    Default conv is computed with nn.functionnal.conv2d 
    but may be replaced with othe
    """
    
    def __init__(self,
                 kwidth,
                 ksigma,
                 undersampling,
                 dtype,
                 device,
                 conv=nn.functional.conv2d) : 
        """
        Builder function for Conv2dConstKernel layer. It generates a 
        constant kernel that will be used then for convolution. The 
        kernel can be generated to compute undersampling at the same time
        with convolution. In this case the ouput is the convolved image
        on a coarser grid. Each pixel is the sum of several fine pixels
        corresponding to finer input grid.

        Parameters
        ----------
        kwidth : int (odd)
            The width of the convolution kernel (must be odd for padding 
            reasons, the kernel is a square matrix).
        ksigma : float
            Standard error sigma for gaussian kernel
        undersampling : int
            Reduction ratio to apply for undersampling during convolution
            Remark : it is different from pytorch stride since it sums values  
            of pixels in fine grid to get value for coarse grid. The stride 
            parameter is different because the result is a selection of one 
            pixel out of given number of pixel instead of the sum.
        dtype : type
            dtype of the kernel data (default: float)
        conv : function for (fft) conv with same signature like conv2d
            default nn.functionnal.conv2d but may be replaced with fft_conv 
        """
        super().__init__()
        
        self.kwidth = kwidth
        self.undersampling = undersampling 
        self.conv = conv
        
        #build gaussian kernel
        ax = torch.linspace(-kwidth//2, kwidth//2, kwidth, dtype=dtype, device=device)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(- (xx**2 + yy**2) / 2 / ksigma**2)
        kernel /= kernel.sum()
        
        #reshape it to a 4D tensor
        kernel = kernel.view(1,1,kwidth,kwidth)
        
        #convolve it with undersampling matrix of ones
        ones = torch.ones((1,1,undersampling, undersampling), dtype=dtype, device=device)
        kernel = nn.functional.conv2d(kernel, ones, padding='same')
        
        #save it as a constant parameter
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        
    def forward(self,x) :
        """
        Applies conv with constant kernel and stride to input x

        Parameters
        ----------
        x : tensor of shape (n_images, n_channels, height, width)

        Returns
        -------
        y : tensor of same shape like x
        """
        return self.conv(x, self.kernel, \
                        padding=(0,0), \
                        stride=self.undersampling)
            

class FFTConv2dConstKernel(Conv2dConstKernel):
    """ 
    A fft conv layer with constant gaussian kernel to modelize diffraction
    It is a wrapper for Conv2dConstKernel with default conv function fft_conv
    """
    
    def __init__(self,*args,conv=fft_conv,**kargs) : 
        super().__init__(*args,conv,**kargs)
    
            

class PoissonProcess(nn.Module):
    def __init__(self,alpha):
        super().__init__()
        self.alpha = alpha
    def forward(self,x): 
        return torch.poisson(self.alpha * x)

class GaussianNoise(nn.Module) : 
    def __init__(self,sigma) :
        super().__init__()
        self.sigma = sigma
    def forward(self,x):
        return x + torch.normal(0, self.sigma, 
                                dtype=x.dtype, 
                                size=x.shape, 
                                device=x.device)
