#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from matplotlib.image import imsave
from tqdm import tqdm
from pathlib import Path


class ISTA_Optimizer:
    """
    Optimizer implementing ISTA proximal algorithm with L1 reg on each 
    parameter in params which requires grad with coefficient given in coeffs
    """
    def __init__(self, params, coeffs, lr):
        self.lr = lr
        self.coeffs = list(coeffs)
        self.params = list(params)
        
    def step(self):
        for p,c,tau in zip(self.params, self.coeffs, self.lr) :
            if p.requires_grad :
                p.data = torch.relu(p.data - tau * p.grad - c * tau)

def l1(x) : 
    return x.abs().sum()

def prox_l1(coeff) : 
    return lambda L, x : torch.relu(x - x.grad/L - coeff/L)

class FISTA_Optimizer:
    """
    This optimizer implements FISTA proximal algorithm, which is an fast 
    version of ISTA, given a proximal operator prox.
    """
    def __init__(self, x, L, f, g=l1, eta=1.1, prox=prox_l1, t0=1, Lmax=float('inf')) :
        """
        Init a FISTA optimizer that minimizes F(x) = f(x) + g(x) with 
        f a smooth fidelity term and g a non smooth regularization (example :
        l1 norm). 
            
        Be careful : like other pytorch optimizers, F(x) is supposed to be 
        computed outside the optimize and backwarded before to call the step 
        method.
        
        Parameters
        ----------
        x : nn.Parameter
            The image we want to optimize.
        L : float > 0
            Initial descent stepsize L0.
        eta : float > 1
            The multiplier to update L and find optimized stepsize eta**k * L.
        f : torch.Tensor -> float
            Fidelity function (eg. L2 distance)
        g : torch.Tensor -> float
            Regularization function (eg. L1 norm)
        prox : float, torch.Tensor -> torch.Tensor
            Corresponding proximal operator which need the step L 
        t : float, optional
            Initialization of parameter t in FISTA algorithm. The default is 1.
        Lmax : float, optional
            max value for the stepsize L, default : inf
        """
        self.eta = eta
        self.L = L
        self.Lmax = Lmax
        self.stepsizes = []
        self.prox = prox
        
        self.t0 = torch.tensor(t0)
        self.t = torch.tensor(t0)
        
        #self.x is a reference to the image we optimize on 
        self.x = x.data.clone().requires_grad_()

        #self.x0 is a copy of this image at the begining of each step
        self.x0 = x.data.clone().requires_grad_()
        
        #self.u is an intermediate between x0 and x at the end of the step
        self.u = x
        
        #F is the loss function we want to minimize
        self.F = lambda x : f(x) + g(x)
        
        #Q is the quadratic approx. It needs gradient of f wrt x0. 
        #This gradient is contained in u.grad and not x0.grad because 
        #it is not update since the last backward() method,
        #that means during the whole step method.
        self.Q = lambda L,x,x0 : f(x0) + ((x-x0) * x0.grad).sum() \
                                       + L/2 * ((x-x0)**2).sum() + g(x)
        self.Q_2 = lambda L,x,x0,fx0 : fx0 + ((x-x0) * x0.grad).sum() \
                                       + L/2 * ((x-x0)**2).sum() + g(x)
                                       
        self.loss = float("inf")
        
    def step(self) : 
        #set grad of x0 equals to last current gradient
        self.x0.grad = self.u.grad
        
        #update x with proximal operator
        self.x = self.prox(self.L, self.x0)
        
        #find the right L :
        while float(self.F(self.x)) > float(self.Q(self.L, self.x, self.x0)):
            self.L *= self.eta
            self.x = self.prox(self.L, self.x0) 
            
        #update t, compute next image and save it in x0
        self.t = (1 + torch.sqrt(1 + 4 * self.t0**2))/2
        self.x0 = self.x * (1 + (self.t0 - 1) / self.t) \
                  - self.u.data * (self.t0 - 1) / self.t
        self.u.data = self.x
        self.t0 = self.t
        
    def backtracking_step(self, loss) : 
            #update x with proximal operator
            self.x = self.prox(self.L, self.u)
            
            #find the right L :
            while float(self.F(self.x)) > float(self.Q_2(self.L, self.x, self.u, loss)) \
                and self.L < self.Lmax :
                    
                self.L *= self.eta
                self.x = self.prox(self.L, self.u) 
            
            #save current L
            self.stepsizes.append(self.L)
            
            #update t, compute next image and save it in x0
            self.t = (1 + torch.sqrt(1 + 4 * self.t0**2))/2
            self.u.data = self.x * (1 + (self.t0 - 1) / self.t) \
                      - self.x0 * (self.t0 - 1) / self.t
            self.x0 = self.x
            self.t0 = self.t
        

def train(G, D, Y, 
          g_x_optimizer, g_b_optimizer, d_optimizer,
          params, rep, 
          g_l2_func, g_D_func):
    """
    Parameters
    ----------
    G : x -> ysim nn.Module
        Generator function.
    D : y -> P(y real) nn.Module
        Discriminator network.
    Y : ndarray(K,w*h)
        Array of input images (blury, noisy and under sampled).
    params : dict
        dict containing the different parameters value for training
        see read_param function in main to see default values
    rep : path or str
        directory to save intermediate x and b calculations
    

    Returns
    -------
    g_losses : list containing the loss for generator at each iteration
    d_losses : same for the discriminator (but not the same len)
    x : list of estimate of the superresolution image every 50 steps
    b : list of estimates of the background every 50 steps

    """
    
    d_losses = {"d_gp" : [], "total" : []}
    g_losses = {"g_x_l1" : [], "g_l2" : [], "g_D" : [], "g_b_grad" : [], "total" : []}
    
    x = []
    b = []
    
    rep = Path(rep)
    batch_size = params['batch_size']
            
    #for each epoch, train first the discriminator and then update x
    for epoch in tqdm(range(params['n_epochs']), desc="Learning...") :
        
        #plot x each 50 epochs : 
        if epoch % 50 == 0 :
            x.append(G.x.detach().clone())
            b.append(G.bg.b.detach().clone())
            
            if epoch > 0 : imsave(rep / f"x{epoch}-img.png", G.x.detach().cpu())

        
        #train D for n_critic steps
        for ii in range(params['n_discriminator']) : 
            D.zero_grad() 
            
            #sample real data
            indices = torch.multinomial(torch.ones(len(Y)), batch_size, replacement=True)
            yreal = Y[indices].detach()
                        
            #sample simulated data
            ysim = G(batch_size).detach()
            
            #mix the simulated and real data with eps ratio
            eps = torch.rand((batch_size, 1, 1, 1), device=ysim.device)
            ymix = eps * yreal + (1 - eps) * ysim
            
            #compute gradient of D(ymix) wrt to ymix to get then grad penalty
            ymix.requires_grad_(True)
            cmix = D(ymix)
            ones = torch.ones_like(cmix)
            grads = torch.autograd.grad(outputs=cmix, inputs=ymix, grad_outputs=ones)[0].flatten(start_dim=1)
            d_gp = params['d_gp'] * ((grads.norm(2, dim=1) - 1) ** 2).sum()
            
            #compute loss and backward gradient wrt D parameters
            d_loss = (D(ysim) - D(yreal)).sum() + d_gp
            d_loss.backward()
            
            #gradient clipping 
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=.001, norm_type=2.)
            
            #save loss
            d_losses['d_gp'].append(float(d_gp))
            d_losses['total'].append(float(d_loss))
            
            d_optimizer.step()
            
        #train G for n_generator steps
        for ii in range(params['n_generator']) :
            G.zero_grad()
        
            #compute grads to regularize b and keep it smooth
            G.bg.b.grad = params['g_b_grad'] * G.bg.spatial_grad_norm_derivative()[0,0]
            
            #sample a batch of simulated images 
            ysim = G(batch_size)

            #compute l2 fidelity term and its gradient
            G.compute_grad_2(g_l2_func, ysim, [G.x, G.bg.b])
            
            #compute gradient of expectation of the distance wrt x
            G.compute_grad_2(g_D_func, ysim, [G.x, G.bg.b])

            #compute explicit loss to plot them and for FISTA next step update
            g_x_l1   = params['g_x_l1']   *  G.x.abs().sum()
            g_b_grad = params['g_b_grad'] *  G.bg.spatial_grad_norm().sum()
            
            g_l2     = g_l2_func(ysim)
            g_D      = g_D_func(ysim)

            #save loss
            g_losses['g_x_l1'].append(float(g_x_l1))
            g_losses['g_b_grad'].append(float(g_b_grad))
            g_losses['g_l2'].append(float(g_l2))
            g_losses['g_D'].append(float(g_D))
            g_losses['total'].append(float(g_b_grad +  g_l2 + g_D + g_x_l1))
            
            #update G.x with optimizer
            g_x_optimizer.backtracking_step(g_losses['total'][-1])
            #g_x_optimizer.step()
            g_b_optimizer.step()
            G.compute_phi()
                
                
    return g_losses, d_losses, x, b
