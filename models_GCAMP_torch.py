#!/bin/env python
from libraries import *


class OpEquilibrium(nn.Module):
    def __init__(self, kf, kb, fact, gmin, qe, Hill=1):
        super().__init__()
        self.kf = nn.Parameter(torch.tensor(kf))
        self.kb = nn.Parameter(torch.tensor(kb))
        self.fact = fact
        self.gmin = nn.Parameter(gmin)
        self.qe = nn.Parameter(qe)
        self.Hill = nn.Parameter(torch.tensor(Hill))
    
    def forward(self, a):
        if self.fact > 1:
            a = nn.AvgPool2d(self.fact,padding=0)(a)
        
        return self.gmin + self.qe*a**self.Hill/(a**self.Hill + self.kb/self.kf)
    
class OpGCAMP(nn.Module):
    def __init__(self, kf, kb, dt, fact, gmin, qe, g0, Hill=1):
        super().__init__()
        self.kf = nn.Parameter(torch.tensor(kf))
        self.kb = nn.Parameter(torch.tensor(kb))
        self.dt = dt
        self.fact = fact
        self.gmin = nn.Parameter(gmin)
        self.qe = nn.Parameter(qe)
        self.g0 = nn.Parameter(g0)
        self.Hill = nn.Parameter(torch.tensor(Hill))
    
    def forward(self, a):
        #solves dg(x,t)/dt = k_f a(x,t) (1 − g(x,t)) − k_{b}g(x,t) with backward Euler
        # ghat[t] = (ghat[t-1] + self.dt*self.kf*a[t]/(1 + self.dt*(k_f*a[t] + self.kb))
        if self.fact > 1:
            a = nn.AvgPool2d(self.fact,padding=0)(a)
        ghat = torch.zeros_like(a)
        ghat[0] = self.g0
        for t in range(1, a.shape[0]):
            ghat[t] = (ghat[t-1] + self.dt*self.kf*a[t].pow(self.Hill))/(1 + self.dt*(self.kf*a[t].pow(self.Hill) + self.kb))
        ghat = self.gmin + self.qe*ghat #conversion from binded sensor to fluorescence (per pixel)
        return ghat
    #g(t)*(1 + dt*(kf*a + kb)) = g(t-1) + dt*kf*a


class LatentVector(nn.Module):
    def __init__(self, Np: int, mode: str ='linear', Nnodes: int = 2, sigma = 1, NdimIn=3):
        '''
        Attributes
        ----------
        Np : int
            The number of time points
        mode : str (optional, default='linear')
            str for type of latent space ('linear', 'geometrical', 'splineshelix','splinesline', 'helix')
        NnodesFactor : integer (optional, default=True)
            Np / NnodesFactors are the number of nodes to create the latent space
        sigma : float
            std of the randomly-initialized latent vecors for 'linear', 'geometrical'
        NdimIn : integer (optional, default=3)
            Length of the output vector (and input possible as well (can be of dim 1))

        Methods
        -------
        '''
        super(LatentVector,self).__init__()
        self.Np = Np
        self.mode = mode
        #self.NnodesFactor = NnodesFactor
        self.Nnodes = Nnodes
        self.invh = self.Nnodes - 1
        self.sig = sigma
        self.NdimIn = NdimIn
        if 'helix' in mode:
            self.TotalRev = 16 #tmp
            self.omega = nn.parameter.Parameter(torch.tensor(2*torch.pi*self.TotalRev/self.Nnodes))
        self.z = []
        self.zsig = [] ##
        for n in torch.arange(self.Nnodes):
            if mode in {'linear','geometrical'}:
                self.z.append(self.sig*torch.randn(self.NdimIn))
            elif 'splines' in mode:
                #assume B-spline order 1 (tri, easier) TO check
                if 'line' in mode: #draw a line between (-sig) and (sig) as initial knots position (NdimIn x 1)
                    self.z.append((-self.sig*torch.ones(self.NdimIn)
                                                + n/(self.Nnodes-1)*(2*self.sig*torch.ones(self.NdimIn))))
                    if 'gauss' in mode:
                        self.zsig.append((0.05*self.sig*torch.ones(self.NdimIn))) ##
                elif 'helix' in mode: #draw an helicoidal trajectory, did it for NdimIn==3, not sure
                    self.z.append(((torch.cos(self.omega*n),torch.sin(self.omega*n),torch.tensor(-1 + 2*n/(self.Nnodes-1)))))
                #self.fcc = nn.Linear(self.NdimIn, torch.prod(torch.tensor(self.szlat)).int())
        if mode in {'linear','geometrical'}:
            self.z = torch.stack(self.z,dim=0)
        else:
            self.z = nn.parameter.Parameter(torch.stack(self.z,dim=0))
            if 'gauss' in mode:
                self.zsig = nn.parameter.Parameter(torch.stack(self.zsig,dim=0))
        
    def forward(self, t):
        #t: Nsamples x NdimsIn \in [0,1] or Nsamples x 1 \in [0,1], would be repeated
        t = t.squeeze()
        alp = t*self.invh % 1 #gives the distance between the neighboring nodes for each entry \in [0,1)
        correctmod = torch.isclose(alp,torch.ones(1,device=t.device))
        alp[correctmod] = 0 #floor division inaccurate because of float
        
        alp = alp.unsqueeze(1).unsqueeze(1).unsqueeze(1) # 4D normally?
        alps = alp + torch.arange(2,-3,step=-1,device=t.device).reshape((1,-1,1,1)) #add along channels a vector of length 5
        
        bv = self.Bsplines(alps, n=3)
        indf = torch.floor(t*self.invh).long()
        indf[correctmod] += 1
                
        indc = torch.floor(1+t*self.invh).long()
        correctmod = torch.isclose((1+t*self.invh) % 1,torch.ones(1, device=t.device)) #inaccurate floor
        indc[correctmod] += 1
        indcc = indc + 1
        
        indff = indf - 1
        indfff = indff - 1
        
        z0 = self.z[indf].to(t.device)
        
        zm2 = torch.zeros_like(z0)
        zm2[indfff>=0] = self.z[indfff[indfff>=0]].to(t.device)
        
        zm2[indfff==-1] = 2*self.z[0] - self.z[1] #natural bc
        zm1 = torch.zeros_like(z0)
        zm1[indff>=0] = self.z[indff[indff>=0]].to(t.device)
        zm2[indff==-1] = 2*self.z[0] - self.z[1] #natural bc
        z1 = torch.zeros_like(z0)
        z1[indc <= self.invh] = self.z[indc[indc <= self.invh]].to(t.device)
        z1[indc==self.Nnodes] = 2*self.z[-1] - self.z[-2] #natural bc
        z2 = torch.zeros_like(z0)
        z2[indcc<=self.invh] = self.z[indcc[indcc<=self.invh]].to(t.device)
        z2[indcc==self.Nnodes] = 2*self.z[-1] - self.z[-2] #natural bc
        
        if self.mode=='linear':
            return alp*z0 + (1-alp)*z1 
        elif self.mode=='geometrical':
            return torch.sqrt(alp)*z0 + torch.sqrt(1-alp)*z1
        elif 'splines' in self.mode:
            if 'gauss' in self.mode: #lazy to adapt
                zsigf = self.zsig[indf].to(t.device)
                zsigc = self.zsig[indc].to(t.device)
            if bv.shape[0]==1:
                bv = bv.squeeze().unsqueeze(0)
            else:
                bv = bv.squeeze().unsqueeze(1)
            if 'gauss' in self.mode: #sampling from Gaussian distribution ? Weird idea I had
                zin = bv[2]*(z0 + zsigf*torch.randn_like(zsigf)) + bv[3]*(z1 + zsigc*torch.randn_like(zsigc))
            else:
                zin = bv[...,0]*zm2 + bv[...,1]*zm1 + bv[...,2]*z0 + bv[...,3]*z1 + bv[...,4]*z2
            return zin
    
    def Bsplines(self,alps,n=3):#max n=3
        #alps are alp + 2, alp + 1, alp, alp - 1, alp - 2 with alp \in [0,1]
        #e.g., alp = 0. => 2., 1., 0., -1, -2
        #alps = alps + (n+1)/2, because we define their support from 0 to 1, except n=1
        v0 = torch.zeros((1),device=alps.device)
        if n==3: #cubic B-splines,
            alps = alps + 2
            bv = torch.where((alps > 0) & (alps < 1),alps**3/6.,v0) + torch.where((alps >= 1) & (alps < 2),(-3*alps**3 + 12*alps**2 - 12*alps + 4)/6,v0)  + torch.where((alps >= 2) & (alps < 3),(3*alps**3 - 24*alps**2 + 60*alps - 44)/6.,v0) + torch.where((alps >=3) & (alps < 4),(-alps**3 + 12*alps**2 - 48*alps + 64)/6.,v0)
        elif n==2: 
            alps = alps + 1.5
            bv = (alps > 0 & alps < 1)*alps**2/2 + (alps >= 1 & alps < 2)*(-2*alps**2 + 6*alps - 3)/2 + (alps >= 2 & alps < 3)*(3 - alps)**2/2
        else: #default n = 1
            bv = (1 - alps.abs()).maximum(0)
        return bv
    
#Neural networks
class ImageGenerator(nn.Module):
    def __init__(self, in_dim, image_size_high_res, latentdim=64,constant_width = 128, 
                 interp_mode = 'nearest', use_normal_conv = True, 
                 use_separate_decoder = False, nl_layer = nn.ReLU(), use_norm = True, 
                 hidden_dim = 64, block_num = 2, use_mapnet = True,last_nl = None): #nn.Sigmoid()
        super(ImageGenerator, self).__init__()

        # hidden_dim = 64 #  for FC layer
        self.block_num = block_num # for decoder
        self.mid_channel_reduction_factor = 2 # for bottleneck residual block

        self.constant_width = constant_width
        self.interp_mode = interp_mode
        self.use_normal_conv = use_normal_conv
        self.use_separate_decoder = use_separate_decoder
        self.nl_layer = nl_layer
        self.last_nl = last_nl
        self.use_norm = use_norm
        self.use_mapnet = use_mapnet
        self.latentdim = latentdim
        self.latsz =  np.array((np.sqrt(latentdim),np.sqrt(latentdim)),dtype=int)

        (NchanOut,m, n) = image_size_high_res
        up_repeat_num = round(np.log2(min(m, n) / self.latsz[0]))
        self.NchanOut = NchanOut
        
        if self.use_mapnet:
            if use_separate_decoder:
                self.manifold_transform = nn.ParameterList()
                for i in torch.arange(self.NchanOut):
                    self.manifold_transform.append(MapNet(in_dim, out_dim = latentdim, hidden_dim = hidden_dim, nl_layer = nl_layer))
            else:
                self.manifold_transform = MapNet(in_dim, out_dim = latentdim, hidden_dim = hidden_dim, nl_layer = nl_layer)
        

        # Decoder: from 8x8 to mxn
        scaled_size_before_final = self.latsz[0] * (2 ** (up_repeat_num - 1))
        self.scale_list = []
        
        if use_separate_decoder:
            decoder_layers=[]
            for i in torch.arange(self.NchanOut):
                decoder_layers.append(self.get_first_conv_layer_list())
        else:
            decoder_layers = self.get_first_conv_layer_list()
        
        
        for i in range(up_repeat_num):
            if i < (up_repeat_num - 1):
                up_scale_factor = 2
                scaled_size = None
            else:
                up_scale_factor = (m / scaled_size_before_final, n / scaled_size_before_final)
                scaled_size = (m, n)
            self.scale_list.append(up_scale_factor)
            
            if use_separate_decoder:
                for i in torch.arange(self.NchanOut):
                    decoder_layers[i].append(self.get_decoder(up_scale_factor, scaled_size))
            else:
                decoder_layers.append(self.get_decoder(up_scale_factor, scaled_size))
             
        if use_separate_decoder:
            self.decoder = nn.ParameterList()
            for i in torch.arange(self.NchanOut):
                self.decoder.append(nn.Sequential(*decoder_layers[i]))
        else:
            self.decoder = nn.Sequential(*decoder_layers)
            
        padding_mode = 'reflect'
        
        if use_separate_decoder:
            self.final_conv = nn.ParameterList()
            for i in torch.arange(self.NchanOut):
                if self.last_nl==None:
                    self.final_conv.append(nn.Conv2d(constant_width, 1,
                                kernel_size = 3, stride = 1, padding = 1, bias = True, padding_mode = padding_mode))
                else:
                    self.final_conv.append(nn.Sequential(nn.Conv2d(constant_width, 1,
                                kernel_size = 3, stride = 1, padding = 1, bias = True, padding_mode = padding_mode),self.last_nl))
        else:
            if self.last_nl==None:
                self.final_conv = nn.Conv2d(constant_width, self.NchanOut, kernel_size = 3, stride = 1, padding = 1, bias = True, padding_mode = padding_mode)
            else:
                self.final_conv = nn.Sequential(nn.Conv2d(constant_width, self.NchanOut, kernel_size = 3, stride = 1, padding = 1, bias = True, padding_mode = padding_mode),self.last_nl)
        
    def get_first_conv_layer_list(self):
        return conv3x3(1, self.constant_width, nl_layer = self.nl_layer, use_norm = self.use_norm)

    def get_decoder(self, up_scale_factor, scaled_size):
        return DecoderBlock(self.constant_width, self.constant_width, 
                            nl_layer = self.nl_layer, up_scale_factor = up_scale_factor, 
                            scaled_size = scaled_size,
                            interp_mode = self.interp_mode, use_normal_conv = self.use_normal_conv, 
                            mid_channel_reduction_factor = self.mid_channel_reduction_factor, 
                            block_num = self.block_num, use_norm = self.use_norm)

    def forward(self, z):
        
        if self.use_separate_decoder:
            '''
            if self.use_mapnet:
                zsep = []
                for i in torch.arange(self.NchanOut):
                    zsep.append(self.manifold_transform[i](z[i]))
                z = torch.stack(zsep)
            '''
            
                
            z = z.view(-1, 1, self.latsz[0], self.latsz[1])
            xout = []
            for i in torch.arange(self.NchanOut):
                zn = self.decoder[i](z)
                xout.append(self.final_conv[i](zn))
            xout = torch.cat(xout,dim=-3)
        else:
            
            if self.use_mapnet:
                z = self.manifold_transform(z)
            z = z.view(-1, 1, self.latsz[0], self.latsz[1])
            z = self.decoder(z)
            xout = self.final_conv(z)
        
        return xout
    
    
class MapNet(nn.Module):
    def __init__(self, in_dim, out_dim = 64, hidden_dim = 512, nl_layer = nn.ReLU()):
        super(MapNet, self).__init__()
        self.fc1 = FCBlock(in_dim, hidden_dim, nl_layer = nl_layer, use_norm = False)
        # self.fc2 = FCBlock(hidden_dim, out_dim, nl_layer = nl_layer, use_norm = False)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias = True)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class FCBlock(nn.Module):
    def __init__(self, in_size, out_size, nl_layer = nn.ReLU(), use_norm = False):
        super(FCBlock, self).__init__()
        if use_norm:
            # TO DO:
            # layer norm?
            module_list = [nn.Linear(in_size, out_size, bias = False), 
                           nn.GroupNorm(out_size,out_size), nl_layer] #nn.BatchNorm1d(out_size), nl_layer]
        else:
            module_list = [nn.Linear(in_size, out_size, bias = True), 
                           nl_layer]
        self.fc_block = nn.Sequential(*module_list)
        
    def forward(self, x):
        return self.fc_block(x)

class DecoderBlock(nn.Module):
    """
    the Bottleneck Residual Block in ResNet
    """

    def __init__(self, in_planes, out_planes, nl_layer = nn.ReLU(), up_scale_factor = 2, 
                 scaled_size = None,
                 interp_mode = 'nearest', use_normal_conv = True, 
                 mid_channel_reduction_factor = 2, block_num = 1, use_norm = True):
        super(DecoderBlock, self).__init__()
        if scaled_size is None:
            up_sampling = nn.Upsample(scale_factor = up_scale_factor, mode = interp_mode)
        else:
            up_sampling = nn.Upsample(size = scaled_size, mode = interp_mode)
        # up_sampling.recompute_scale_factor=True
        if use_normal_conv:
            layers = [up_sampling, 
                      conv3x3(in_planes, out_planes, nl_layer = nl_layer, use_norm = use_norm)]
        else:
            layers = [up_sampling, 
                      ConvBottleNeck(in_planes, out_planes, nl_layer = nl_layer, reduction_factor = mid_channel_reduction_factor)]
        
        for i in range(block_num - 1):
            if use_normal_conv:
                layers.append(conv3x3(out_planes, out_planes, nl_layer = nl_layer, use_norm = use_norm))
            else:
                layers.append(ConvBottleNeck(out_planes, out_planes, nl_layer = nl_layer, reduction_factor = mid_channel_reduction_factor))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)

def conv3x3(in_planes, out_planes, nl_layer = nn.ReLU(), use_norm = True, padding_mode = 'reflect'):
    """3x3 convolution with zero padding"""

    conv = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = 1, 
                        padding = 1, bias = False, padding_mode = padding_mode)
    #nn.init.xavier_uniform(conv.weight.data)
    nn.init.kaiming_uniform_(conv.weight.data, nonlinearity = 'relu')
    #nn.init.zeros_(conv.bias.data)
    layers = [conv]
    if use_norm: 
        #RN = nn.BatchNorm2d(out_planes,affine=True)
        RN = nn.GroupNorm(out_planes,out_planes)
        layers.append(RN)
        
    layers.append(nl_layer)
    return nn.Sequential(*layers)

class ConvBottleNeck(nn.Module):
    """
    the Bottleneck Residual Block in ResNet
    """

    def __init__(self, in_channels, out_channels, nl_layer = nn.ReLU(), reduction_factor = 2):
        super(ConvBottleNeck, self).__init__()
        self.nl_layer = nl_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels // reduction_factor, kernel_size=1, bias = False)
        self.conv2 = nn.Conv2d(out_channels // reduction_factor, out_channels // reduction_factor, kernel_size=3, padding=1, bias = False)
        self.conv3 = nn.Conv2d(out_channels // reduction_factor, out_channels, kernel_size=1, bias = False)

        #RN = nn.BatchNorm2d(out_channels // reduction_factor)
        RN = nn.GroupNorm(out_channels//reduction_factor, out_channels//reduction_factor)
        self.norm1 = RN
        self.norm2 = RN
        #RN=nn.BatchNorm2d(out_channels)
        RN = nn.GroupNorm(out_channels,out_channels)
        self.norm3 = RN 
        

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        identity = x

        y = self.conv1(x)
        y = self.norm1(y)
        y = self.nl_layer(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.nl_layer(y)

        y = self.conv3(y)
        y = self.norm3(y)

        if self.in_channels != self.out_channels:
            identity = self.skip_conv(identity)
        y += identity
        y = self.nl_layer(y)
        return y

def init(DIP,z,x0,Ma,Nepoch=100,bs=4,lr=1e-5,lr_min=1e-9,wd=1e-6):
    #[0.01, 0.99, 0, 4e-3,1, 1]
    dev = z.get_device()
    optimizer = torch.optim.Adam(DIP.parameters(),lr=lr,amsgrad=True,weight_decay=wd)
    
    criterion = lambda x,y: ((x - y).abs()**2).mean() #nn.MSELoss()

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = np.sqrt(0.1), patience = 20,threshold=1e-4,min_lr = lr_min)
    
    train_dataloader = DataLoader(x0, batch_size=bs, shuffle=True)
    prevl = 1e10
    t0 = time.time()
    for niter in torch.arange(Nepoch):
        for (b, x0b) in train_dataloader:
            optimizer.zero_grad()
            
            zb = z[b].to(dev)
            x0hat = DIP(zb)
            DataLoss = criterion(x0hat.squeeze()[Ma[b]], x0b.squeeze()[Ma[b]])
            loss = DataLoss.clone()
            loss.backward()

            optimizer.step()
            
        sched.step(loss.item())
        if niter % (Nepoch//10)==0:
            with torch.no_grad():
                strdisp = 'Iter {:d} | Loss {:.3e}| lr {:.2e} | time {:.2f} sec'.format(niter,DataLoss.item(),sched._last_lr[0],time.time() - t0)
                print(strdisp)
        if niter > 0 and np.abs(prevl - loss.item()/loss.item()) < 1e-4:
            print('No change ! Prev vs New loss {:.2e} vs {:.2e}'.format(prevl, loss.item()))
            break
        prevl = loss.item()