#!/bin/env python
from libraries import *
from models_GCAMP_torch import OpGCAMP, ImageGenerator, LatentVector, OpEquilibrium

from GCAMPparam import get_param

torch.manual_seed(0)

par = get_param()
print('Saving in {}'.format(par.fold_res))
makedirs(par.fold_res)
if par.gpu>=0:
    ordi = platform.system()
    if ordi=='Darwin': #mac
        print('On mac')
        device = torch.device("mps")
    else:
        device = torch.device('cuda:' + str(par.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

if 'jgcamp8s' in par.sensor.casefold():
    par.kb = 3.681079025810e+00
    par.kf = 8.089243550527982e-04 
    if '479571_cell02' in par.id:
        g = torch.from_numpy(tifffile.imread('../data//movie_reg_jGCaMP8s_ANM{}_mn_{:d}.ome.tif'.format(par.id,par.mn))).to(torch.float32) # 10000 x 512 x 128,479571_cell02
        roim = (70,96,60,68)
        par.exp_start = 3.504800000e+02
    elif '479572_cell09' in par.id:
        g = torch.from_numpy(tifffile.imread('../data//movie_reg_jGCaMP8s_ANM{}_mn_{:d}.ome.tif'.format(par.id,par.mn))).to(torch.float32)
        roim = (215,64,60,64)
        if par.mn==0:
            par.exp_start = 0.00
        elif par.mn==1:
            par.exp_start = 205.333000
        par.fs = 121.96668231804678
    elif '472181_cell01' in par.id:
        g = torch.from_numpy(tifffile.imread('../data//movie_reg_jGCaMP8s_ANM{}_mn_{:d}.ome.tif'.format(par.id,par.mn))).to(torch.float32)
        roim = (215,96,17,96)
        if par.mn==0:
            par.exp_start = 0.00
        par.fs = 120.12841104927844
    par.Hill = 1.
elif 'jgcamp7f' in par.sensor.casefold():
    if '471991_cell04' in par.id:
        if par.mn==0:
            g = torch.from_numpy(tifffile.imread('../data//movie_reg_jGCaMP7f_ANM{}_mn_{:d}.ome.tif'.format(par.id,par.mn)))
            roim = (152,72,23,72) #mn 0
            par.exp_start = 0.0
        elif par.mn==1:
            g = torch.from_numpy(tifffile.imread('../data//movie_reg_jGCaMP7f_ANM{}_mn_{:d}.ome.tif'.format(par.id,par.mn)))
            roim = (95,52,44,52) #mn 1
            par.exp_start = 1.18616700e+03
    elif '478406_cell01' in par.id:
        g = torch.from_numpy(tifffile.imread('../data//movie_reg_jGCaMP7f_ANM{}_mn_{:d}.ome.tif'.format(par.id,par.mn)))
        roim = (193,64,58,64) #mn 0 normally
        par.exp_start = 0.0
        par.fs = 127.87204383929603
    else:
        exit('Data doesn''t exist!')
    par.kb = 7.342660811
    par.kf = 1.318167390611477e-06 
    par.Hill = 1.
elif 'jgcamp8m' in par.sensor.casefold():
    g = torch.from_numpy(tifffile.imread('../data//movie_reg_jGCaMP8m_ANM{}_mn_{:d}.ome.tif'.format(par.id,par.mn))).to(torch.float32)
    
    #for jgcamp8m
    if '472180_cell05' in par.id:
        roim = (170,64,50,64)
        par.exp_start = 243.43103899999915
        par.fs = 121.96973720798644
    elif  '472180_cell04' in par.id:
        if par.mn == 0:
            roim = (310,64,30,42) #(310,64,30,42)
            par.exp_start = 0.0
            par.fs = 121.9694628307145
        elif par.mn==1:
            roim = (306,64,22,64)
            par.exp_start = 392.594
            par.fs = 121.96603321894956
    elif '479117_cell02' in par.id:
        roim = (364,42,28,52)
        if par.mn==1:
            par.exp_start = 191.188001
            par.fs = 121.96444804135785
        elif par.mn==3:
            par.exp_start = 596.235000
            par.fs = 121.96415844605859
    par.kb = 1.824071527789e+01
    par.kf = 2.274407369575e-03 
    par.Hill = 1.
elif 'simGCAMP'.casefold() in par.sensor.casefold():
    
    #a_gt = torch.from_numpy(tifffile.imread('../data/astrocyte_simulation.tif')).to(device)
    a_gt = torch.from_numpy(tifffile.imread(par.fileoi)).to(device)
    a_gt = a_gt[int(par.Toi[0]):int(par.Toi[-1])]
    a_gt = a_gt*par.concs
    
    g0gt = par.gmin0*torch.ones(a_gt.shape[1:],device=device)
    gmin0gt = par.gmin0*torch.ones(a_gt.shape[1:],device=device)
    qe0gt = par.qegt*torch.ones(a_gt.shape[1:],device=device)
    Hgt = OpGCAMP(par.kf,par.kb,par.dt[1],par.fact,gmin0gt,qe0gt,(g0gt - gmin0gt)/qe0gt,Hill=par.Hill).to(device)
    Hgt.to(device)
    
    g = Hgt(a_gt).detach()
    gclean = g.clone().cpu().numpy()
    g = g + torch.maximum(torch.sqrt(g/par.noise[0]),torch.tensor(par.noise[1]))*torch.randn_like(g)
    g = g.maximum(torch.zeros_like(g)).cpu()
    
    print('Measurements SNR {:.2e}'.format(SNR(gclean,g.numpy())))
    Hgt.to('cpu')
elif 'complex' in par.sensor.casefold():
    fname = '{}/sim_complex.npy'.format(par.fold_res)
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            g = torch.from_numpy(np.load(f))
            gclean = np.load(f)
            a_gt = torch.from_numpy(np.load(f))
            #a_gt_cmpl = torch.from_numpy(np.load(f))
    else:
        roim = (0,32,0,32)
        dts = 1e-5
        factNt = int(par.dt[1]/dts)
        Nt = int(par.Toi[1]/dts)
        simshape = (Nt, roim[1] - roim[0],1)
        g0gt = 0.1*torch.ones(simshape[1:],device=device) #uM#300
        gmin0gt = par.gmin0*torch.ones(simshape[1:],device=device)
        qe0gt = par.qegt*torch.ones(simshape[1:],device=device)
        
        Hgt = OpGCAMP6(dts, gmin0gt, qe0gt ,g0gt)
        Hgt.to(device)
        Nsec = 4
        AAs = 1.7*500*np.repeat(0.25*np.ones((simshape[1]//Nsec)).cumsum(),(Nsec),axis=0)#uM 0.1
        AAs = np.expand_dims(AAs,1)
        
        ca0_init = 1e-4*np.ones_like(AAs)
       
        times = np.arange(0,Nt)*Hgt.dt
        
        g, a_gt, ctot, fakefreec = Hgt(AAs,Nt,ca0_init)
        a_gt = a_gt*torch.ones((1,1,roim[3] - roim[2])).to(device=device)
        fakefreec = torch.from_numpy(fakefreec).to(device)*torch.ones((1,1,roim[3] - roim[2])).to(device=device)
        g = g*torch.ones((1,1,roim[3] - roim[2])).to(device=device)
        gclean = g.detach().clone().cpu().numpy()
        if par.noise[1] >= 0:
            g = g + torch.maximum(torch.sqrt(g/par.noise[0]),torch.tensor(par.noise[1]))*torch.randn_like(g)
        g = g.maximum(torch.zeros_like(g)).cpu()
        print('Measurements SNR {:.2e}'.format(SNR(gclean,g.numpy())))
        Hgt.to('cpu')
        
        for idx in range(0,simshape[1],Nsec):
            cf,axes=plt.subplots(2,2, figsize=(10, 10))
            
            axes[0,0].plot(times, a_gt[:,idx,0].cpu(),linestyle='dotted',label='Free C sim')
            axes[0,0].plot(times, ctot[:,idx,0],linestyle='dashed',label='C total sim')
            axes[0,0].plot(times, fakefreec[:,idx,0].cpu(),linestyle='dashed',label='Simple free C sim')
            axes[0,0].legend()
            axes[0,1].plot(times,fakefreec[:,idx,0].cpu()**4,label='GT**Hill')
            axes[0,1].legend()
            axes[1,0].plot(times,np.array([Hgt.dca_input(t,AAs[idx,0]) for t in times]),label='Derivative continuous')
            axes[1,0].legend()
            axes[1,1].plot(times,g[:,idx,0],linestyle='dotted',label='Measured Fluorescence')
            axes[1,1].plot(times,gclean[:,idx,0],label='Clean Fluorescence')
            axes[1,1].legend()
            plt.show()
            plt.savefig('{}/sims_{:d}.png'.format(par.fold_res,idx),dpi=400)
            print('Ca <0 {}; G4 < 0 {}'.format(np.any(a_gt[:,idx,0].cpu().numpy().squeeze() < 0.), np.any(gclean[:,idx,0].squeeze() < 0.)))
        a_gt_cmpl = a_gt.clone()
        a_gt = fakefreec #a_gt for DUSK
        g = g[::factNt]
        gclean = gclean[::factNt]
        a_gt = a_gt[::factNt]
        a_gt_cmpl = a_gt_cmpl[::factNt]
        tifffile.imwrite('{}/a_gt_cmpl.ome.tif'.format(par.fold_res),a_gt_cmpl.cpu().squeeze().numpy())
        with open(fname, 'wb') as f:
            np.save(f, g.cpu().numpy())
            np.save(f, gclean)
            np.save(f, a_gt.cpu().numpy())
            np.save(f, a_gt_cmpl.cpu().numpy())
    
elif 'custom'.casefold() in par.sensor.casefold():
    g = torch.from_numpy(tifffile.imread('{}'.format(par.fileoi))).to(torch.float32)
    if par.roim is None:
        roim = (0,g.shape[1],0,g.shape[2])
else:
    exit('Data doesn''t exist!')

fs = par.fs
exp_start = par.exp_start

print('Loading {}'.format(par.sensor))

if 'simGCAMP'.casefold() in par.sensor.casefold():
    par.fold_res = '{}/{}_{}_{}_Toi_{:.2f}_{:.2f}_ds_{:d}_qe0_{:1.2f}_mp_{:d}_dt_{:.2e}_{:.2e}_dst_{:d}_N_{:d}_L_{}_lr_{:.0e}_{:.0e}_{:d}_opK_{:d}_rA_{:.1e}_rQe_{:.1e}_q_{:.2f}_z_{:d}_epC_{:.1e}_{:d}'.format(
    par.fold_res,par.sensor,par.id,par.method,par.Toi[0],par.Toi[1],par.ds,par.qe0,par.mapnet,par.dt[0],par.dt[-1],par.dst,par.Nepoch,par.criterion,par.lr[0],par.lr_min[0],par.lr_patience[0],par.optiK,par.regA,par.regQe,par.quant,par.paramZ,par.epsiC[0],par.epsiC[1])
else:
    par.fold_res = '{}/{}_{}_mn_{:d}_{}_Toi_{:.2f}_{:.2f}_ds_{:d}_qe0_{:1.2e}_mp_{:d}_dt_{:.2e}_{:.2e}_dst_{:d}_N_{:d}_L_{}_lr_{:.0e}_{:.0e}_{:d}_opK_{:d}_rA_{:.1e}_rQe_{:.1e}_q_{:.2f}_z_{:d}_epC_{:.1e}_{:d}'.format(
        par.fold_res,par.sensor,par.id,par.mn,par.method,par.Toi[0],par.Toi[1],par.ds,par.qe0,par.mapnet,par.dt[0],par.dt[-1],par.dst,par.Nepoch,par.criterion,par.lr[0],par.lr_min[0],par.lr_patience[0],par.optiK,par.regA,par.regQe,par.quant,par.paramZ,par.epsiC[0],par.epsiC[1])
if par.optiK:
    par.fold_res = '{}_lr_{:.1e}_{:.1e}_{:d}'.format(par.fold_res,par.lr[-1],par.lr_min[-1],par.lr_patience[-1])
if 'simGCAMP'.casefold() in par.sensor.casefold():
    par.fold_res = '{}_sim_{:1.2e}_{:1.2f}_{:1.2f}_n_{:.2e}_{:.2e}_eq_{:d}'.format(par.fold_res,\
                                                                            par.concs,par.gmin0,par.qegt,
                                                                            par.noise[0],par.noise[1],par.equi)
if 'complex'.casefold() in par.sensor.casefold():
    par.fold_res = '{}_sim_{:1.2f}_{:1.2f}_n_{:.2e}_{:.2e}_eq_{:d}'.format(par.fold_res,\
                                                                            par.gmin0,par.qegt,
                                                                            par.noise[0],par.noise[1],par.equi)

print('Data params: start time {} | kb/kf {}/{} | Hill {} | Toi {} to {}'.format(par.exp_start,par.kb,par.kf,par.Hill,par.Toi[0],par.Toi[-1],par.fs))

Tn = (int((par.Toi[0] - exp_start)*fs),int((par.Toi[1] - exp_start)*fs))
print('{:d} Samples : {:d} to {:d}'.format(Tn[1] - Tn[0],Tn[0],Tn[1]))

if 'simGCAMP'.casefold() not in par.sensor.casefold() and  'complex'.casefold() not in par.sensor.casefold():
    par.fold_res = '{}_ri_{:d}_{:d}_{:d}_{:d}'.format(par.fold_res,roim[0],roim[1],roim[2],roim[3])
    g = g[Tn[0]:Tn[1], roim[0]:roim[0]+roim[1], roim[2]:roim[2]+roim[3]] 

makedirs(par.fold_res)
if os.path.exists('{}/res.mat'.format(par.fold_res)):
    if par.saveNet:
        if os.path.exists('{}/dip.pth'.format(par.fold_res)) and os.path.exists('{}/latentvector.pth'.format(par.fold_res)) and os.path.exists('{}/H.pth'.format(par.fold_res)):
            print('Using existing optimized network to interpolate in time')
            loadRes = True
        else:
            loadRes = False
    else:
        exit("Already done")
else:
    loadRes = False
print('Res in {}'.format(par.fold_res))


print('g shape ', g.shape)
g = np.maximum(g,0) #negative fluorescence comes from registration artifact done by (Zhang 2023)

normalizer = np.quantile(g,par.quant,axis=None)
g = g/normalizer # for numerical purpose (DIP)

print('min max normalizer',g.min().item(),g.max().item(),normalizer)

g = nn.functional.interpolate(g.unsqueeze(0),size=(g.shape[1]//par.ds,g.shape[2]//par.ds),mode='area').squeeze()
a0 = g
g0 = g[0]
qe0 = par.qe0*torch.ones(g.shape[1:])
gmin0 = g.amin(0,keepdim=True)[0]

if par.equi:
    H = OpEquilibrium(par.kf, par.kb, par.fact, gmin0, qe0, Hill=par.Hill)
    H.to(device)
    H.Hill.requires_grad = False
    H.kf.requires_grad = False
    H.kb.requires_grad = False
    if not par.optiK:
        H.gmin.requires_grad = False
        H.qe.requires_grad = False
else:
    H = OpGCAMP(par.kf,par.kb,par.dt[1],par.fact,gmin0,\
        qe0, (g0-gmin0)/qe0, Hill=par.Hill)
    H.to(device)
    H.Hill.requires_grad = False
    H.kf.requires_grad = False
    H.kb.requires_grad = False
    if not par.optiK:
        H.gmin.requires_grad = False
        H.qe.requires_grad = False
        H.g0.requires_grad = False

if 'MSE' in par.criterion:
    criterion = nn.MSELoss(reduction='mean')
elif 'Wass' in par.criterion:
    criterion = lambda x,y: (torch.cumsum(x,0) - torch.cumsum(y,0)).abs().mean()
else:
    criterion = nn.L1Loss(reduction='mean')

ups_mode = 'trilinear' #'nearest' #trilinear
OpSpatialResize = nn.Upsample(scale_factor=(1,par.fact,par.fact),mode=ups_mode)
OpTimeResize = lambda x: nn.Upsample(scale_factor=(np.round(par.dt[0]/par.dt[-1]),1,1),mode=ups_mode)(x.unsqueeze(0).unsqueeze(0)).squeeze()

g = g.to(device)
a0ups = OpTimeResize(a0)
gups =  OpTimeResize(g)
    

M = (g.sum(0)!=0).expand_as(g)
a = a0ups
a = torch.cat((torch.zeros((par.apad0,*a.shape[1:])),a),dim=0)
Mhat = lambda x: x.narrow(0,par.apad0, x.shape[0]-par.apad0)
torch.cat((torch.zeros((par.apad0,*M.shape[1:]),device=device),M),dim=0).to(bool)


print('g, a, Mhat, M ',g.shape,a.shape,Mhat(a).shape,M.shape)

a[~M[0].expand_as(a)]=0

if par.fact != 1:
    a = OpSpatialResize(a.unsqueeze(0).unsqueeze(0)).squeeze()

szM = M.shape
sz = a.shape

if par.optiK:
    optiK = optim.Adam(H.parameters(), lr = par.lr[-1], weight_decay = par.wd, eps=1e-12,amsgrad=True)
    schedK = torch.optim.lr_scheduler.ReduceLROnPlateau(optiK, mode='min', factor=par.lr_factor[-1], patience=par.lr_patience[-1],threshold=par.lr_thres[-1],min_lr=par.lr_min[-1])

#Regularization
if par.fact != 1:
    Ma = OpSpatialResize(M.to(torch.float32).unsqueeze(0).unsqueeze(0)).squeeze()
else:
    Ma = M.to(torch.float32)

Ma = torch.from_numpy(Ma[0].to(bool).cpu().numpy()).expand_as(a)

costA = costL1(par.regA)
costQe = costTVH2O2(par.regQe,Ma[0].to(bool).to(device))

if 'dip' in par.method:
    hidden_dim = par.hidden_dim
    block_num = par.block_num
    cwdt = par.cwdt
    upsnet = par.upsnet
    if par.mapnet:
        zdim = par.zdim
        latentdim = par.latentdim #must be the square of an integer, 64
    else:
        zdim = 8*8
        latentdim = zdim
    Fa = ImageGenerator(zdim, (1,*sz[1:]), latentdim = latentdim, constant_width = cwdt,
                 interp_mode = upsnet, use_normal_conv = True, 
                 use_separate_decoder = False, nl_layer = nn.LeakyReLU(), use_norm = True, 
                 hidden_dim = hidden_dim, block_num = block_num, use_mapnet = par.mapnet, last_nl=nn.ReLU())
    strnet = 'z_{:d}_latent_{:d}_hid_{:d}_block_{:d}_width_{:d}_ups_{}_nl_leaky_{}'.format(zdim,latentdim,hidden_dim,block_num,cwdt,upsnet,Fa)
    Fa.to(device)
    t = torch.linspace(0,1,sz[0],device=device).unsqueeze(1)
    
    if par.paramZ>0:
        z = LatentVector(sz[0], mode ='splinesline', Nnodes = sz[0]//par.paramZ, sigma = 2., NdimIn=zdim) #2*(par.Toi[1] - par.Toi[0])/2.
        z.to(device)
        t = t.squeeze()
    else:
        z = t*torch.rand((zdim),device=device) + (1-t)*torch.rand((zdim),device=device)

if loadRes:
    z.load_state_dict(torch.load('{}/latentvector.pth'.format(par.fold_res)))
    z.eval()
    H.load_state_dict(torch.load('{}/H.pth'.format(par.fold_res)))
    H.eval()
    Fa.load_state_dict(torch.load('{}/dip.pth'.format(par.fold_res)))
    Fa.eval()
else:
    if 'dip' in par.method:
        if par.paramZ > 0:
            optiA = optim.Adam([{'params':Fa.parameters(), 'lr': par.lr[0]},
                                {'params':z.parameters(), 'lr': par.lr[-1]}], lr = par.lr[0], weight_decay = par.wd, eps=1e-12, amsgrad=True)
        else:
            optiA = optim.Adam(Fa.parameters(), lr = par.lr[0], weight_decay = par.wd, eps=1e-12, amsgrad=True)
        
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optiA, mode='min', factor = par.lr_factor[0], patience = par.lr_patience[0],threshold=par.lr_thres[0],min_lr = par.lr_min[0])
        

    tifffile.imwrite('{}/gups.tif'.format(par.fold_res), gups.cpu().numpy())
    tifffile.imwrite('{}/g.tif'.format(par.fold_res), g.cpu().numpy())
    if 'gclean' in locals():
        tifffile.imwrite('{}/gclean.tif'.format(par.fold_res), gclean)
    tifffile.imwrite('{}/a0ups.tif'.format(par.fold_res), a0ups.cpu().numpy())

    losses = []

    strp = 'Number of parameters {:d}\n'.format(count_parameters(Fa))
    if par.paramZ > 0:
        strp = '{} and z params {:d}\n'.format(strp,count_parameters(z))
    print(strp, end="")

    with open('{}/params.txt'.format(par.fold_res), 'w') as f:
        json.dump(par.__dict__, f, indent=2)
        
    with open('{}/res.txt'.format(par.fold_res), 'w') as f:
        f.write('Starting optimization for {}\n'.format(par.fold_res))
        f.write('{}\n'.format(strnet))
        f.write(strp)
    end0 = time.time()
    end_time = end0
    count = 0
    lastloss = 1e15

    for niter in range(par.Nepoch):
        with torch.set_grad_enabled(True):
            
            optiA.zero_grad()
            if par.optiK:
                optiK.zero_grad()
            if 'dip' in par.method:
                if par.paramZ>0:
                    ahat = Fa(z(t)).squeeze()
                else:
                    ahat = Fa(z).squeeze()
                    
            ghat = H(ahat)

            ghat = Mhat(ghat[::np.round(par.dt[0]/par.dt[-1]).astype(int)])
            loss_train = criterion(ghat[::par.dst],g[::par.dst])
            
            if par.regA > 0:
                loss_RegA = costA.apply(ahat)
                loss_train += loss_RegA
            if par.regQe > 0:
                loss_qe = costQe.apply(H.qe)
                loss_train += loss_qe

            loss_train.backward()
            optiA.step()
                    
            if isinstance(sched,torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(loss_train)
            elif sched.get_last_lr()[0]>par.lr_min[0]:
                sched.step()
            
            if par.optiK:
                optiK.step()
                if isinstance(schedK,torch.optim.lr_scheduler.ReduceLROnPlateau):
                    schedK.step(loss_train)
                else:
                    schedK.step()
                H.gmin.data = H.gmin.data.maximum(torch.zeros_like(H.gmin.data))
                H.qe.data = H.qe.data.maximum(1e-6*torch.ones_like(H.qe.data))
                if not par.equi:
                    H.g0.data = H.g0.data.maximum(1e-6*torch.ones_like(H.g0.data))
                
            #Convergence criterion       
            if niter > 0:
                rerr = (aprev - ahat).pow(2).sum().sqrt()/aprev.pow(2).sum().sqrt()
                if rerr < par.epsiC[0]:
                    count = count + 1
                    print('Count {:d} | solution did not change significantly {:.2f}'.format(count,rerr))
                else:
                    count = 0
            aprev = ahat.clone()
            
            losses.append(loss_train.item())
            
            if niter % par.disp[0] ==0:
                with torch.no_grad():
                    
                    ahat = ahat[par.apad0:]
                    striter = 'Iter {:04d} | Training Loss {:.6e} | Time {:.6f} '.\
                            format(niter, loss_train, time.time() - end0)
                    if par.regA > 0:
                        striter += '| Loss Reg {:.3e}'.format(loss_RegA)
                    if par.regQe > 0:
                        striter += '| Loss Qe {:.3e} and avg+-std {:.3e}+-{:.3e}'.format(loss_qe,H.qe.mean(),H.qe.std())
                    
                    if 'sched' in locals():
                        if isinstance(sched,torch.optim.lr_scheduler.ReduceLROnPlateau):
                            striter += '| lr a {:.3e}'.format(*sched._last_lr)
                        else:
                            striter += '| lr a {:.3e}'.format(*sched.get_last_lr())
                    if 'schedK' in locals():
                        if isinstance(schedK,torch.optim.lr_scheduler.ReduceLROnPlateau):
                            striter += '| lrK a {:.3e}'.format(*schedK._last_lr)
                        else:
                            striter += '| lrK a {:.3e}'.format(*schedK.get_last_lr())
                    striter += '\n'
                    print(striter,end="")
                    with open('{}/res.txt'.format(par.fold_res), 'a') as f:
                        f.write(striter)
                    if 'dip' in par.method:
                        if par.paramZ>0:
                            ahat = Fa(z(t)).squeeze()
                        else:
                            ahat = Fa(z).squeeze()
                    save_and_monitor(ahat, H, g.clone(), par.fold_res, niter, losses, np.round(par.dt[0]/par.dt[-1]).astype(int), M, Ma, niter % (par.disp[-1]) == 0, Mhat)
                    if par.paramZ>0:
                        with open('{}/zt.txt'.format(par.fold_res), 'w') as f:
                                f.write('{}'.format(z(t).cpu().numpy()))
            if count >= par.epsiC[1]:
                break
        
    end = time.time()

    with torch.no_grad():
        if 'dip' in par.method:
            if par.paramZ>0:
                zt = z(t)
                ahat = Fa(zt).squeeze()
                zt = zt.detach().cpu().numpy()
                ck = z.z.detach().cpu().numpy()
            else:
                ahat = Fa(z).squeeze()
                zt = z.detach().cpu().numpy()
                ck = 0
        ahatM = ahat[par.apad0:]
        ghat = H(ahat)
        save_and_monitor(ahat,H, g.clone(), par.fold_res,niter, losses, np.round(par.dt[0]/par.dt[-1]).astype(int), M, Ma, True, Mhat)
        niter = niter+1
        save_and_monitor(ahatM, H, g.clone(), par.fold_res, niter, losses, np.round(par.dt[0]/par.dt[-1]).astype(int), M, Ma, True, Mhat)
        if 'a_gt' in locals():
            a_gt = a_gt.cpu().numpy()
        else:
            a_gt = []
        if par.optiK:
            kf = H.kf.detach().cpu().squeeze().numpy()
            kb = H.kb.detach().cpu().squeeze().numpy()
            
            gmin = H.gmin.detach().cpu().squeeze().numpy()
            qe = H.qe.detach().cpu().squeeze().numpy()
            savemat('{}/res.mat'.format(par.fold_res), {'gups':gups.cpu().numpy(),'ghat':ghat.detach().cpu().squeeze().numpy(),'ahat':ahatM.detach().cpu().squeeze().numpy(),'a0ups':a0ups.cpu().numpy(),'losses':np.squeeze(np.stack(losses)),'kf':kf,'kb':kb,'gmin':gmin,'qe':qe,'normalizer':normalizer,'z':zt,'ck':ck,'a_gt':a_gt})
            tifffile.imwrite('{}/gmin.tif'.format(par.fold_res), gmin)
            tifffile.imwrite('{}/qe.tif'.format(par.fold_res), qe)
        else:
            savemat('{}/res.mat'.format(par.fold_res), {'gups':gups.cpu().numpy(),'ghat':ghat.detach().cpu().squeeze().numpy(),'ahat':ahatM.detach().cpu().squeeze().numpy(),'a0ups':a0ups.cpu().numpy(),'losses':np.squeeze(np.stack(losses)),'normalizer':normalizer,'z':zt,'ck':ck,'a_gt':a_gt})

with torch.no_grad():
    if par.saveNet:
        #z, H,
        if par.paramZ>0:
            torch.save(z.state_dict(), '{}/latentvector.pth'.format(par.fold_res))
        else:
            zog = z.clone()
            z = lambda t: t*zog[0] + (1-t)*zog[-1]
        torch.save(H.state_dict(), '{}/H.pth'.format(par.fold_res))
        torch.save(Fa.state_dict(), '{}/dip.pth'.format(par.fold_res))
    if par.interpF > 1:
        interpF = par.interpF
        print('Saving network and super res {:d}'.format(interpF))
        tsr = torch.linspace(0,1,sz[0]*interpF,device=device)
        gsr = nn.functional.interpolate(g.unsqueeze(0).unsqueeze(0),size=(g.shape[0]*interpF,g.shape[1],g.shape[2]),mode='trilinear').squeeze()
        ahatsr = torch.zeros((interpF*g.shape[0],*g.shape[1:]))
        indcut = ahatsr.shape[0]//2
        ahatsr[0:indcut] = Fa(z(tsr[0:indcut])).squeeze().cpu()
        ahatsr[indcut:] = Fa(z(tsr[indcut:])).squeeze().cpu()

        ghat = H(Fa(z(t))).squeeze()
        ghatsr0 = nn.functional.interpolate(ghat.unsqueeze(0).unsqueeze(0),size=(ghat.shape[0]*interpF,ghat.shape[1],ghat.shape[2]),mode='trilinear').squeeze().cpu()
        H.dt = H.dt/interpF
        ghatsr1 = H(ahatsr.to(device)).cpu()
        savemat('{}/sr_{:d}.mat'.format(par.fold_res,interpF), {'ahatsr':ahatsr.squeeze().cpu().numpy(),'gsr':gsr.cpu().squeeze().numpy(),                                                   'ghatsr0':ghatsr0.cpu().squeeze().numpy(),'ghatsr1':ghatsr1.cpu().squeeze().numpy(),'interpF':interpF})


print('Optimization is over')