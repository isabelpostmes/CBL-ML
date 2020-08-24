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


energy_file = pd.read_csv('data/Prediction_120keV')

groups = energy_file.groupby(['time', 'energy'])


ncols, nrows = 2,1
gs = matplotlib.gridspec.GridSpec(nrows,ncols)

plt.figure(figsize=(ncols*9,nrows*7))



for name, group in groups:
    mean_prediction = group.iloc[:, 4:].mean(axis=1)
    std_prediction = group.iloc[:, 4:].std(axis=1)
    
    
    if group['time'].max() == .1:
        i = 0
        ax = plt.subplot(gs[i])
        ax.set_title('10 ms exposure time', fontsize = 18)
        ax.set_ylabel('Intensity (a.u.)', fontsize = 18)
        ax.set_ylim([0, 1])
        ax.plot(group.x*1000, (mean_prediction), label=int(name[1] * 100))
        ax.fill_between(group.x*1000, mean_prediction + std_prediction, mean_prediction - std_prediction, alpha=.3)
    if group['time'].max() == 1:
        i = 1
        ax = plt.subplot(gs[i])
        ax.set_title('100 ms exposure time', fontsize = 18)
        ax.plot(group.x*1000, (mean_prediction),  label=int(name[1] * 100))
        ax.set_ylim([0, 1.2])
        ax.fill_between(group.x*1000, mean_prediction + std_prediction, mean_prediction - std_prediction, alpha=.3)
    ax.set_xlim([-90, 90])
    ax.set_yticks([])
    ax.set_xticks([-80, -40, 0, 40, 80])
    ax.set_xlabel('Energy loss (meV)', fontsize = 18)
   
    #ax.set_yticklabels(fontsize=1)
    ax.tick_params(which='major',direction='in',length=10, labelsize=16)
    ax.tick_params(which='minor',length=10, labelsize=14)

    ax.legend(fontsize = 14)
plt.tight_layout()
plt.savefig('120keV.pdf')
