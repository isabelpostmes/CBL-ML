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


fnamebase="data/"


####################################################################
####################################################################
#
# First series
#

# Read and preprocess data
# Now plot the spectra
import matplotlib.pyplot as plt
# Number of spectra
nsp=4
offset=[1e2, 5e3, 1e4, 1.4e4]


ncols,nrows=1,1
fig = plt.figure(figsize=(ncols*5,nrows*3.5))
rescolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#py.suptitle(plottitle, fontsize=20)
gs = gridspec.GridSpec(nrows,ncols)

# loop over spectra
ipl=0
for isp in range(3,nsp):

    ax = plt.subplot(gs[0])

    fname = fnamebase+"EELS_Sp0"+str(isp)+"_m3d868eV-to-45d332eV.txt"
    print(isp, "  ",fname)
    EEL = np.loadtxt(fname)
    l=len(EEL)
    print("Number of data points = ",l)
    # Min and max values of the EEL spectra
    deltaE_min = -3.868 # eV, adjust for each spectrum
    deltaE_max = 45.332 # eV, adjust for each spectrum
    # Assign values of energy loss
    deltaE = np.zeros(l)
    for i in range(l):
        deltaE[i] = deltaE_min + (deltaE_max-deltaE_min)*(1.0*i)/(l-1.0)
        EEL[i] = EEL[i] + offset[ipl]

    # Make the plot
    plt.plot(deltaE, EEL,color=rescolors[0],\
             ls="solid",linewidth=1,marker="D",markersize=0.0)

    axins = ax.inset_axes([0.20, 0.55, 0.47, 0.42])
    axins.get_xaxis().set_visible(True)
    axins.get_yaxis().set_visible(True)
    axins.plot(deltaE, EEL,color=rescolors[0],\
             ls="solid",linewidth=2,marker="D",markersize=0.0)
    axins.tick_params(which='both',direction='in',labelsize=8,right=True)
    axins.tick_params(which='major',length=3)
    axins.tick_params(which='minor',length=3)
    axins.set_yticks([])
    axins.set_ylim(0,4.5e6)
    axins.set_xlim(-2,2)
    
    
    ipl = ipl+1

# Now produce the plot        
plt.xlabel(r"Energy loss (eV)",fontsize=18)
plt.ylabel(r"Intensity (a.u.)",fontsize=18)
plt.xlim(0,35)
plt.ylim(0,4.5e4)
plt.tick_params(which='both',direction='in',labelsize=12,right=True)
plt.tick_params(which='major',length=7)
plt.tick_params(which='minor',length=4)
#plt.grid(True)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.yticks([])
plt.tight_layout(pad=1, w_pad=1, h_pad=1.0)
plt.savefig("EELS_example.pdf")

exit()


####################################################################
####################################################################

