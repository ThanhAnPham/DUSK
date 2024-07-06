from sys import exit
import os
import platform
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json

from utils import *

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import tifffile

#from scipy.ndimage.morphology import binary_erosion #distance_transform_cdt, distance_transform_edt
#from skimage.morphology import skeletonize
#from sklearn.decomposition import NMF

#import torchvision
#from torchvision.transforms import GaussianBlur

#import numpy as np
#import matplotlib.pyplot as plt



#for FENICS

# Note to myself: correct conda environment is fenicsproject
# Import fenics, torch_fenics, and at the end import fenics_adoint to override necessary data structures with fenics_adjoint
# conda installation: fenics, torch_fenics, dolfin_adjoint (torch_fenics force install the old dolfin_adjoint 2018.1)
#from utils import *
#from fenics import *
#from fenics_adjoint import *

from scipy.io import savemat