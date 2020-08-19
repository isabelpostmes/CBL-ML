import numpy as np
from numpy import loadtxt
import math
import scipy
#import sklearn
from scipy import optimize
from scipy import signal
from scipy import interpolate
from scipy.optimize import leastsq
from io import StringIO
from scipy.interpolate import UnivariateSpline
from matplotlib import gridspec
from  matplotlib import rc
from  matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

###################### Load data ################################################

df_dx = pd.read_csv('data/derivatives.csv')

ratio1 = np.divide(df_dx['derivative y14'], df_dx['derivative y17'])
ratio2 = np.divide(df_dx['derivative y14'], df_dx['derivative y22'])
ratio3 = np.divide(df_dx['derivative y14'], df_dx['derivative y23'])

total_ratio = (ratio1 + ratio2 + ratio3)/3

########################################################################

ncols,nrows=1,1
fig = plt.figure(figsize=(ncols*9,nrows*6))
rescolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#py.suptitle(plottitle, fontsize=20)
gs = gridspec.GridSpec(nrows,ncols)


nsp = 14
ipl = 0

def smooth(x, window_len, window='hanning'):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': 
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    index = int(window_len/2)
    return y[(index-1):-(index)]


nrows, ncols = 3,1
gs = matplotlib.gridspec.GridSpec(nrows,ncols)
plt.figure(figsize=(ncols*9,nrows*7))

cm_subsection = np.linspace(0,1,24) 
colors = [cm.viridis(x) for x in cm_subsection]

hfont = rc('font',**{'family':'sans-serif','sans-serif':['Sans Serif']})
          
for i in range(1):
    ax = plt.subplot(gs[i])
    ax.set_xlim([0,6])
    ax.tick_params(which='major',direction='in',length=7)
    ax.tick_params(which='minor',length=8)
    plt.axhline(y=0, color='black', linewidth=1, alpha=.8)
    plt.axvline(x=0, color='darkgray', linestyle='--', linewidth = 1)
    
        
    for j in ([14]):   
        if i == 0: 
            ax.axhline(y=1, linestyle='--', color='gray')
        
            p1 = ax.plot(df_dx['x%(i)s'% {"i": j}], ratio1, color='lightgray', label='14/17')
            p1 = ax.plot(df_dx['x%(i)s'% {"i": j}], ratio2, color='lightgray', label='14/22')
            p1 = ax.plot(df_dx['x%(i)s'% {"i": j}], ratio3, color='lightgray', label='14/23')
            p1 = ax.plot(df_dx['x%(i)s'% {"i": j}], total_ratio, label='Mean spectrum 14/vacuum')
            ax.set_title('Derivatives ratio', fontsize=24)
            ax.set_ylim([-1, 2])
            ax.set_xlim([1,3.5])   
            ax.set_ylabel('$dI/dE_{sp14}$ / $dI/dE_{vac}$', fontsize=20)
            ax.set_xlabel('$\Delta$E (eV)', fontsize=24)
            ax.legend(fontsize=16)  
    
   
    if i == 0:
        ax.set_xlabel('Energy loss (eV)', fontsize=24)
        ax.tick_params(length= 10, labelsize=18)
        ax.tick_params(which='major', length= 10, labelsize=20)
        ax.tick_params(which='minor', length= 10, labelsize=10)
    
plt.tight_layout()
plt.savefig("Derivatives.pdf")
plt.show()