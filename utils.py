import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim.optimizer import Optimizer
import torch.optim as optim
from Operators import *
import tifffile

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
def SNR(gt,x):
    return 10*np.log10(np.power(gt,2).sum()/np.power(x - gt,2).sum())

class costL1():
    def __init__(self,regL1): #'anisotropic
        self.regL1 = regL1        
    def apply(self,x):
        loss = (x).abs().sum()
        return self.regL1*loss

class costTVH2O2():
    def __init__(self,regTV,M,n_iter=15,ndims=2,index=None,type='anisotropic',bounds=(-float('inf'),float('inf')),epsi=1e-8,res = torch.tensor((1,1,1))): #'anisotropic
        self.regTV = regTV
        self.ndims = ndims
        self.n_iter = n_iter
        self.D = LinOpGrad(ndims=ndims,index=index,bc='mirror',res=res[:ndims])
        self.gam = 1/self.D.norm**2 #for 2D, 12 for 3D, Lipschitz of finite difference
        self.type = type
        self.bounds = torch.tensor(bounds)
        self.M = M
        self.epsi = epsi
        
    def apply(self,x):
        Dxout = self.D.apply(x)
        Dx = torch.stack([Dxout[...,n][self.M] for n in range(self.D.lgthidx)])
        if self.type=='isotropic':
            loss = (Dx.pow(2).sum(dim=0) + self.epsi).sqrt().sum()
        else: #anisotropic
            loss = (Dx).abs().sum()
        return self.regTV*loss
    
    def project_ball(self,x):
        if self.type=='isotropic':
            #Project L2 ball
            x.div_(torch.maximum(x.pow(2).sum(dim=self.ndims).sqrt(),torch.tensor(1.)).unsqueeze(self.ndims))
        else: #anisotropic
            x.div_(torch.maximum(x.abs(),torch.tensor(1.)))
        return x
    
    def proj(self,x):
        return torch.minimum(torch.maximum(x,self.bounds[0]),self.bounds[1])
    
    def prox(self,y,rho):
        sz = y.shape
        tau = self.regTV*rho

        ilip = self.gam/tau #1./(self.tv_ct*rho)
        x = torch.zeros((*sz,self.D.lgthidx)) #np.zeros(self.size*self.ndim)
        x_prev = torch.zeros_like(x)
        t_prev = torch.tensor(1.)

        for k in range(self.n_iter):
            x_temp = x + ilip*self.D.apply(self.proj(y-tau*self.D.adjoint(x)) ) 
            
            self.project_ball(x_temp) #in-place, but not sure
            #x_temp = self.project_ball(x_temp)
            t = 0.5 * (1 + np.sqrt(1 + 4 * t_prev ** 2))
            
            x = x_temp + (x_temp - x_prev) * (t_prev - 1) / t

            x_prev = x_temp
            t_prev = t
        return self.proj(y-tau*self.D.adjoint(x_temp))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_and_monitor(ahat,H, g, fold_res, itr, losses,dt_ratio, M, Ma, save_ahat, Mhat=None):
    
    ghat = H(ahat).cpu()
    ahat = ahat.clone().cpu()
    if Mhat is not None:
        ghat = Mhat(ghat)
    ahat[~Ma] = 0  
    #print('min/max ghat vs GT: {:.6e}/{:.6e} vs {:.6e}/{:.6e}'.format(ghat.min(),ghat.max(), g.min(),g.max()), end ="\n")
    
    #print('min/max ahat: {:.6e}/{:.6e}'.format(ahat.min(),ahat.max()), end ="\n")
    
    #figure estimated concentration
    toi = int(ahat.shape[0]//2)
    xoi = int(ahat.shape[1]//2)
    yoi = int(ahat.shape[2]//2)
    ax = []
    f = plt.figure()
    
    ax.append(plt.subplot(2, 2, 1))
    plt.imshow(ahat[toi].cpu())
    
    ax.append(plt.subplot(2, 2, 3))
    plt.imshow(ahat[:,xoi].cpu())
    
    ax.append(plt.subplot(2, 2, 2))
    plt.imshow(ahat[...,yoi].cpu())
    
    for cax in ax:
        plt.sca(cax)
        plt.axis('off')
        plt.colorbar(ax=cax)
    #plt.show()
    plt.savefig('{}/ahat_orthoview_{:d}.png'.format(fold_res,itr), bbox_inches='tight',dpi=300)
    plt.close(f)
    
    #figure estimated fluorescence
    ghatdec = ghat[::dt_ratio].clone()
    ghatdec[~M] = float('nan')
    g[~M] = float('nan')
    toi = int(ghatdec.shape[0]//2)
    xoi = int(ghatdec.shape[1]//2)
    #yoi = int(ghatdec.shape[2]//2)
    ax = []
    f = plt.figure()
    
    ax.append(plt.subplot(2, 2, 1))
    plt.imshow(ghatdec[toi].cpu())
    
    ax.append(plt.subplot(2, 2, 2))
    plt.imshow(ghatdec[:,xoi].cpu())
    
    ax.append(plt.subplot(2, 2, 3))
    plt.imshow(g[toi].cpu())
    
    ax.append(plt.subplot(2, 2, 4))
    plt.imshow(g[:,xoi].cpu())
    
    for cax in ax:
        plt.sca(cax)
        plt.axis('off')
        plt.colorbar(ax=cax)
    #plt.show()
    plt.savefig('{}/g_{:d}.png'.format(fold_res,itr), bbox_inches='tight',dpi=300)
    plt.close(f)
    
    losses = np.squeeze(np.stack(losses))

    f = plt.figure()
    plt.plot(losses.squeeze())
    #plt.show()
    plt.savefig('{}/loss_{}.png'.format(fold_res, itr), bbox_inches='tight',dpi=200)
    plt.close(f)
    
    if save_ahat:
        tifffile.imwrite('{}/ahat_{:d}.tif'.format(fold_res,itr), ahat.cpu().numpy())
        tifffile.imwrite('{}/ghat_{:d}.tif'.format(fold_res,itr), ghat.cpu().numpy())