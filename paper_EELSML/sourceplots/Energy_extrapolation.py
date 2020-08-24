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


fwhmvalues_mean_100 = np.loadtxt('data/fwhmvalues350_mean_1.0')
fwhmvalues_mean_10 = np.loadtxt('data/fwhmvalues350_mean_0.1')
up100 = np.loadtxt('data/fwhmvalues350_up_1.0')
up10 = np.loadtxt('data/fwhmvalues350_up_0.1')
low100 = np.loadtxt('data/fwhmvalues350_down_1.0')
low10 = np.loadtxt('data/fwhmvalues350_down_0.1')

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

color2 = 'indigo'

plt.figure(figsize=(ncols*9,nrows*7))
ax = plt.subplot(gs[0])
ax.plot(np.linspace(0,350,36), smooth(fwhmvalues_mean_100 * 1000, wl), color = color2, label = '100 ms')
ax.fill_between(np.linspace(0,350,36), smooth(up100 * 1000, wl), smooth(low100 * 1000, wl), color = color2, alpha=.2, label = '100 ms')
ax.plot(np.linspace(0,350,36), smooth(fwhmvalues_mean_10 * 1000, wl*2), label = '10 ms')
ax.fill_between(np.linspace(0,350,36), smooth(up10 * 1000, wl*2), smooth(low10 * 1000, wl),  alpha=.2, label = '10 ms')
ax.axvline(x=60, linestyle='--', color='gray')
ax.axvline(x=200, linestyle='--', color='gray')
ax.tick_params(which='major',direction='in',length=7, labelsize=14)
ax.tick_params(which='minor',length=8, labelsize=14)
ax.set_yticks([10, 15, 20, 25, 30, 35])
ax.set_ylabel('FWHM (meV)', fontsize = 18)
ax.set_xlim([0, 350])
ax.set_xlabel('$E_{beam}$ (keV)', fontsize = 18)
lables, handles = ax.get_legend_handles_labels() 
ax.legend([(lables[0], lables[2]), (lables[1], lables[3])], [(handles[0]), handles[1]], loc='upper right', fontsize=16)
ax.set_title('$E_{beam}$ extrapolation', fontsize=18)

plt.savefig('Ebeam_extrapolation.pdf')