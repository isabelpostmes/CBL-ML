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

energy_file_full = pd.read_csv('data/prediction_120keV_full')
energy_file_cut = pd.read_csv('data/prediction_120keV_cut')


hfont = rc('font',**{'family':'sans-serif','sans-serif':['Sans Serif']})

groups_full = energy_file_full.groupby(['time', 'energy'])
groups_cut = energy_file_cut.groupby(['time', 'energy'])
ncols, nrows = 1,1

gs = matplotlib.gridspec.GridSpec(nrows,ncols)
plt.figure(figsize=(ncols*9,nrows*7))
ax = plt.subplot(gs[0])


for name, group in groups_cut:
    mean_prediction = group.iloc[:, 4:].mean(axis=1).to_numpy()
    std_prediction = group.iloc[:, 4:].std(axis=1).to_numpy()
    
    if group['time'].max() == 1 and group['energy'].max() == 2:
        ax.plot(group.x*1000, np.divide(mean_prediction, mean_prediction), label='Prediction on cut data')
        ax.fill_between(group.x*1000, np.divide(mean_prediction + std_prediction, mean_prediction), \
                        np.divide(mean_prediction - std_prediction, mean_prediction), alpha=.3)
        
for name, group in groups_full:
    mean_prediction = group.iloc[:, 4:].mean(axis=1).to_numpy()
    std_prediction = group.iloc[:, 4:].std(axis=1).to_numpy()
    
    if group['time'].max() == 1 and group['energy'].max() == 2:
        ax.plot(group.x*1000, np.divide(mean_prediction, mean_prediction), label='Prediction on full range')
        ax.fill_between(group.x*1000, np.divide(mean_prediction + std_prediction, mean_prediction), \
                        np.divide(mean_prediction - std_prediction, mean_prediction), alpha=.3)        
        
ax.axvline(x=100, linestyle='--', color='black') 
ax.axvline(x=800, linestyle='--', color='black') 
ax.set_title('$t_{exp}$ = 100ms, $E_b$ = 200keV', fontsize = 18)
ax.set_ylabel('Intensity (normalized)', fontsize = 18)  
ax.set_xlim([-90, +1000])
ax.set_ylim([0.9, 1.1])
ax.tick_params(which='major',direction='in',length=10, labelsize=16)
ax.tick_params(which='minor',length=10, labelsize=14)
ax.legend(fontsize = 14, loc = 'upper right')

plt.tight_layout()

plt.savefig('Data/Prediction_with_cut_10ms.pdf')