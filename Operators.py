import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

class LinOpGrad():
    '''
    Implements finite difference and its adjoint
    '''
    
    def __init__(self,ndims=2,index=None,bc = 'zeros',res=1):
        self.name ='LinOpGrad'
        self.isInvertible = False
        self.isDifferentiable = True
        self.bc=bc
        self.ndms = ndims
        if not index:
            self.index = range(0,self.ndms)
        else:
            self.index = index
        self.lgthidx = len(self.index)
        self.res= res*torch.ones(self.ndms)
        self.norm = 2*torch.sqrt( torch.sum(1/self.res**2) )
        
        if self.bc not in ['mirror', 'circular', 'zeros']:
            error('boundary conditions not valid')


    def apply(self, x):
        szx = x.shape
        y = torch.zeros((*szx,self.lgthidx)).to(x.device) #,dtype=torch.double, if float, some deviations at 1e-5 precision
        for k in range(self.lgthidx):
            indoi = self.index[k]
            if self.bc=='zeros':
                bcs = torch.zeros((*szx[:indoi],1,*szx[indoi+1:]))
            elif self.bc=='circular':
                bcs = torch.narrow(x,indoi,0,1) #first element
            elif self.bc=='mirror':
                bcs = torch.narrow(x,indoi,szx[indoi]-1,1) #last element-> last element of y always 0
            y[...,k] = torch.diff(x, n=1, dim=indoi,append=bcs)/self.res[indoi]
        return y
            
    def adjoint(self,x):
        szx = x.shape
        y = torch.zeros(szx[:-1]) #,dtype=torch.double
        for k in range(self.lgthidx):
            indoi = self.index[k]
            if self.bc == 'zeros':
                #y= [[-x(1)  (-x(2:end-1)+x(1:end-2))]  x(end-1)-x(end)]/self.res(1)
                prebcs = torch.zeros((*szx[:indoi],1,*szx[indoi+1:-1]))#,dtype=torch.double
                appbcs = torch.narrow(x[...,k],indoi,szx[indoi]-1,1) #x(end)
            elif self.bc == 'circular':
                #y= [[x(end)-x(1)  (-x(2:end-1)+x(1:end-2))]  (x(end-1)-x(end))]/self.res(1)
                prebcs = torch.narrow(x[...,k],indoi,szx[indoi]-1,1) #x(end)
                appbcs = torch.narrow(x[...,k],indoi,szx[indoi]-1,1) #x(end)
            elif self.bc == 'mirror':
                #y= [[-x(1)  (-x(2:end-1)+x(1:end-2))]  x(end-1)]/self.res(1)
                prebcs = torch.zeros((*szx[:indoi],1,*szx[indoi+1:-1]))#,dtype=torch.double
                appbcs = torch.zeros((*szx[:indoi],1,*szx[indoi+1:-1]))#,dtype=torch.double
            y.add_(-torch.diff(x[...,k].narrow(indoi,0,szx[indoi]-1),n=1,dim=indoi,prepend=prebcs,append=appbcs),alpha=1/self.res[indoi])
        return y