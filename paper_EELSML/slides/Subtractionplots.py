import numpy as np
import math
import scipy
from scipy import optimize
import matplotlib
from matplotlib import gridspec
from  matplotlib import rc
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd


########## Load data ####################################

mean_rep = pd.read_csv('data/Subtracted_spectrum.csv')


# Define the optimal parameters for dE1 = 1.8 eV #
# [ amp, E_bg, b ] #

amp = 2717
pars = [amp, 1.59, 1.257]
pars_high = [amp, 1.59, (1.26 + 0.3)]
pars_low = [amp, 1.59, (1.26 - 0.74)]
nbins = 493

########################################################################

def bandgap(x, amp, BG,b):
    return amp * (x - BG)**(b)

########################################################################


nrows, ncols = 1,1
gs = matplotlib.gridspec.GridSpec(nrows,ncols)
plt.figure(figsize=(ncols*9,nrows*7))

cm_subsection = np.linspace(0,1,24) 
colors = [cm.viridis(x) for x in cm_subsection]

hfont = rc('font',**{'family':'sans-serif','sans-serif':['Sans Serif']})
amp = 2700
pars = [amp, 1.59, 1.257]
pars_high = [amp, 1.59, (1.26 + 0.3)]
pars_low = [amp, 1.59, (1.26 - 0.74)]
          
for i in range(1):
    ax = plt.subplot(gs[i])
    ax.set_xlim([1,5])
    ax.set_ylim([-1e3,1e4])
    ax.set_ylabel('Intensity (a.u.)', fontsize=18)
    ax.set_xlabel('Energy loss (eV)', fontsize=18)
    
    ax.tick_params(which='major',direction='in',length=10, labelsize=16)
    ax.tick_params(which='minor',length=10)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    
    ax.plot(np.linspace(-.3, 12, nbins), mean_rep['spectrum14'], 'k--', label='Raw spectrum 14')
    ax.plot(np.linspace(-.3, 12, nbins), mean_rep['match14_median'], color='red', label='Predicted ZLP')
    ax.fill_between(np.linspace(-.3, 12, nbins), mean_rep['match14_low'], mean_rep['match14_high'], \
                    color='red', alpha=.2)
    
    ax.plot(np.linspace(-.3, 12, nbins), mean_rep['dif14_median'], 'k-', label='Subtracted spectrum')
    ax.fill_between(np.linspace(-.3, 12, nbins), mean_rep['dif14_low'], mean_rep['dif14_high'], color='black', alpha=.2)
    

    ax.legend(loc='upper left', fontsize = 14)
    
    axins = ax.inset_axes([0.50, 0.5, 0.5, 0.45])
    
    axins.get_xaxis().set_visible(True)
    axins.get_yaxis().set_visible(True)
    axins.spines['right'].set_visible(True)
    axins.spines['top'].set_visible(True)
    axins.set_xticks([1, 2, 3, 4])
    axins.set_xlim([1,4])
    axins.set_ylim([-1e3, 5e3])
    
    axins.plot(np.linspace(-.3, 12, nbins), mean_rep['dif14_median'], 'k-', alpha=.8, label='Subtracted spectrum')
    axins.fill_between(np.linspace(-.3, 12, nbins), mean_rep['dif14_low'], \
                       mean_rep['dif14_high'], color='black', alpha=.1)
    
   
    
    x = np.linspace(1.5, 2.7, 100)
    axins.plot(x, bandgap(x, *pars), label='Polynomial fit')
    axins.fill_between(x, bandgap(x, *pars_low), \
                       bandgap(x, *pars_high), color='steelblue', alpha=.5)
    axins.tick_params(which='both',direction='in', labelsize=12,right=True)
    axins.tick_params(which='major',length=10)
    axins.tick_params(which='minor',length=10)
    axins.set_yticks([])
    axins.legend(loc='upper left', fontsize=12, frameon=False)

    
    
   