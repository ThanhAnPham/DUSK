from argparse import ArgumentParser
import numpy as np
def get_param():
    parser = ArgumentParser('recon_GCAMP')
    parser.add_argument('--fold_res', type=str, default='../results',help='folder in which the res will be saved')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index, -1 for cpu')
    parser.add_argument('--Toi', type=float, nargs='+', default=(473.5, 475.5),help='') #number of samples in dt[0] (data)
    parser.add_argument('--apad0', type=int, default=0,help='Temporal pre-padding. Allows DUSK to generate few frames free of measurements.')
    parser.add_argument('--qe0', type=float, default=1, help='Initial value of qe')
    parser.add_argument('--ds', type=float, default=1, help='Downsample (local average) the measurements')
    parser.add_argument('--criterion', type=str, default='L1',help='Loss function L1') #'MSE','L1','Wass'
    

    parser.add_argument('--noise', type=float, nargs='+',default=(12.5,0.1),help = '(Np,sigmin)| Np:Photon budget in simulated g | sigmin: minimum noise (Gaussian readout)') #poui 25
    parser.add_argument('--gmin0',type=float,default=0.25,help='For simulated data, add a background of fluorescence (see paper)')
    parser.add_argument('--qegt',type=float,default=10., help='Ground-truth Qe for simulation')
    parser.add_argument('--concs',type=float,default=1e4, help='For simulation, scale the ground-truth concentration')
    parser.add_argument('--quant',type = float, default=0.99,help = 'Quantile value to normalize the measurements')
    
    parser.add_argument('--fs',type=float, default= 121.97176094076032) #sampling rate of (loaded) raw data
    parser.add_argument('--dt', type=float, nargs='+',default=(1./121.97176094076032,1./121.97176094076032),help='(Acquisition sampling time step, Forward time step). Forward time step should be smaller or equal than the first and acquisition sampling time step should be a multiple of the forward time step.')
    parser.add_argument('--dst', type=int,default=1,help='Downsampling factor for both measurements and forward.')
    parser.add_argument('--equi',type=int,default=0,help='Boolean for using equilibrium model or ODE')#forward model
    ######################################## Sensors related ################################
    parser.add_argument('--sensor',type=str,default='jgcamp8s',help='Sensor type: jgcamp7f, jgcamp8s, jgcamp8m, simGCAMP, customXXX')
    parser.add_argument('--fileoi',type=str,default='../data/astrocyte_simulation.tif',help='TIFF file used for simGCAMP or custom')
    parser.add_argument('--id',type=str,default='479571_cell02')
    parser.add_argument('--mn',type=int,default=0)
    parser.add_argument('--roim',type=int,nargs='+',default=None,help='spatial region of interest (a,b,w,h) with a,b the top left indices of the window of interest and w,h')
    #related to data, starting time point
    parser.add_argument('--exp_start',type=float, default= 3.504800000e+02,help='Experiment starting time')
    
    #Below only used in simulation and custom. Overwritten in main script for real data
    #k_d backward rate [s^(-1)]
    parser.add_argument('--kb', type=float, default= 3.681079025810,help='kinetic parameter: backward reaction rate')
    #k_d forward rate [nM^(-1) s^(-1)]
    parser.add_argument('--kf', type=float, default= 8.089243550527982e-04,help='kinetic parameter: forward reaction rate')
    ########
    parser.add_argument('--Hill', type=float, default=1.,help='Hill coefficient, leave it at 1 for DUSK') #"Hill coeff", DIP estimates [CA]^Hill directly, so Hill is set at 1.

    #DIP architecture related (from zdim to hidden dim ot latentdim)
    parser.add_argument('--hidden_dim',type=int,default=16) 
    parser.add_argument('--block_num',type=int,default=1)
    parser.add_argument('--cwdt',type=int,default=64)
    parser.add_argument('--upsnet',type=str,default='bilinear')
    parser.add_argument('--zdim',type=int,default=3)
    parser.add_argument('--latentdim',type=int,default=36)
    parser.add_argument('--paramZ',type=int,default=16)
    
    parser.add_argument('--fact', type=float, default = 1.,help='Upsampling factor of the reconstructed image')
    
    #Number of batches per epoch (might not want to run through all the dataset each time, too slow), will be capped by len(train_gen)
    parser.add_argument('--Nepoch', type=int, default=10000,help = 'Number of epochs')
    parser.add_argument('--regA', type=float, default=0,help='Regularization tradeoff parameter on concentration (L1 norm)')
    parser.add_argument('--regQe', type=float, default=1e-5,help='Regularization tradeoff parameter for TV on qe')
    parser.add_argument('--wd', type=float, default=0,help='weight decay')
    
    parser.add_argument('--method', type=str, default='dip')

    parser.add_argument('--mapnet', type=int, default=1)
    parser.add_argument('--lr', type=float, nargs='+',default=(1e-2, 1e-2),help='Initial learning rate (LR) for concentration and binding variables') #(A,k rates or alpha) 
    parser.add_argument('--lr_min', type=float, nargs='+', default=(1e-2, 1e-2),help='Smallest LR achievable for concentration and binding variables') #(A,k rates)
    parser.add_argument('--lr_factor', type=float, default=(np.sqrt(0.1),np.sqrt(0.1)),help ='LR is reduced by #1 (#2) for concentration (binding variables)') #(A,k rates)
    parser.add_argument('--lr_thres', type=float, default=(1e-3,1e-3)) #(A,k rates)
    parser.add_argument('--lr_patience', type=int, default=(500, 500),help='LR scheduler. Every #1 epochs (#2) for concentration (binding variables, resp.), LR is reduced') #(A,k rates)
    parser.add_argument('--epsiC', type=float, nargs='+', default=(0.01,5),help='stopping criterion. Stop if the relative change of the concentration norm is smaller than #1 for #2 consecutive epochs')
    parser.add_argument('--disp', type=int, nargs="+",default=(100, 2000),help = 'display every #1 and save every #2')
    parser.add_argument('--optiK', type=int, default=1,help='optimization of binding variables g0min, g0, qe')
    parser.add_argument('--saveNet', type=int, default=1, help= 'boolean for saving optimized network and latent space')
    parser.add_argument('--interpF', type=int, default=0,help='save (temporally) interpolated frames with factor #1')
    par = parser.parse_args()
    par.equi = bool(par.equi)
    par.saveNet = bool(par.saveNet)
    par.optiK = bool(par.optiK)
    par.mapnet = bool(par.mapnet)
    par.epsiC = (par.epsiC[0],int(par.epsiC[-1]))
    
    return par

if __name__ == '__main__':
    get_param()
