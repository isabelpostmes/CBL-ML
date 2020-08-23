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

fwhmvalues_mean_200 = np.loadtxt('data/fwhmvalues_time_mean_2.0')
fwhmvalues_mean_60 = np.loadtxt('data/fwhmvalues_time_mean_0.6')
up200 = np.loadtxt('data/fwhmvalues_time_up_2.0')
up60 = np.loadtxt('data/fwhmvalues_time_up_0.6')
low200 = np.loadtxt('data/fwhmvalues_time_down_2.0')
low60 = np.loadtxt('data/fwhmvalues_time_down_0.6')


################################################################################

def smooth(x, window_len, window='hanning'):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': 
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    index = int(window_len/2)
    return y[(index-1):-(index)]

################################################################################





hfont = rc('font',**{'family':'sans-serif','sans-serif':['Sans Serif']})

nrows, ncols = 1,1

gs = matplotlib.gridspec.GridSpec(nrows,ncols)
wl = 4
color1 = 'red'
color2 = 'orange'
plt.figure(figsize=(ncols*9,nrows*7))
ax = plt.subplot(gs[0])
ax.plot(np.linspace(0,200,21), smooth(fwhmvalues_mean_200 * 1000, wl), color = color2, label = '200 keV')
ax.fill_between(np.linspace(0,200,21), smooth(up200 * 1000, wl), smooth(low200 * 1000, wl), color = color2, alpha=.2, label = '200 keV')
ax.plot(np.linspace(0,200,21), smooth(fwhmvalues_mean_60 * 1000, wl*2), color=color1, label = '60 keV')
ax.fill_between(np.linspace(0,200,21), smooth(up60 * 1000, wl*2), smooth(low60 * 1000, wl), color=color1, alpha=.2, label = '60 keV')
ax.axvline(x=10, linestyle='--', color='gray')
ax.axvline(x=100, linestyle='--', color='gray')
ax.tick_params(which='major',direction='in',length=7, labelsize=14)
ax.tick_params(which='minor',length=8, labelsize=14)
ax.set_yticks([ 15, 20, 25, 30, 35, 40])
ax.set_ylabel('FWHM (meV)', fontsize = 18)
ax.set_xlim([0, 200])
ax.set_xlabel('t (ms)', fontsize = 18)
lables, handles = ax.get_legend_handles_labels() 
ax.legend([(lables[0], lables[2]), (lables[1], lables[3])], [(handles[0]), handles[1]], loc='upper right', fontsize=16)
ax.set_title('Time extrapolation', fontsize=18)
plt.savefig('Time_extrapolation.pdf')
