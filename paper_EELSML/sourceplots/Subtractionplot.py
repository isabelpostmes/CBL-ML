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

mean_rep = pd.read_csv('Data/Subtracted_spectra_1.45.csv')


import matplotlib
from matplotlib import rc

nbins = 1861
rescolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

########################################################################

def bandgap(x, amp, BG,b):
    return amp * (x - BG)**(b)

########################################################################


nrows, ncols = 1,2
gs = matplotlib.gridspec.GridSpec(nrows,ncols)
plt.figure(figsize=(ncols*9,nrows*7))

cm_subsection = np.linspace(0,1,24) 
colors = [cm.viridis(x) for x in cm_subsection]

hfont = rc('font',**{'family':'sans-serif','sans-serif':['Sans Serif']})
          
for i in range(1):
    ax = plt.subplot(gs[i])
    ax.set_xlim([0.8,2.5])
    ax.set_ylim([-1e1,2e3])
    ax.set_ylabel('Intensity (a.u.)', fontsize=31)
    ax.set_xlabel('Energy loss (eV)', fontsize=28)
    
    ax.tick_params(which='major',direction='in',length=10, labelsize=20)
    ax.tick_params(which='minor',length=10)
    ax.set_xticks([.75, 1, 1.25,1.5,1.75, 2, 2.5, 3])
    
    p1=ax.plot(np.linspace(-.3, 9, nbins), mean_rep['spectrum4'], 'k--',ls="dashdot",lw=2)

    
    p2=ax.plot(np.linspace(-.3, 9, nbins), mean_rep['match4_median'], color=rescolors[1],ls="dashed",lw=2)
    ax.fill_between(np.linspace(-.3, 9, nbins), mean_rep['match4_low'], mean_rep['match4_high'], \
                    color=rescolors[1], alpha=.2)
    p2b=ax.fill(np.NaN,np.NaN,color=rescolors[1],alpha=0.2)
    
    p3 = ax.plot(np.linspace(-.3, 9, nbins), mean_rep['dif4_median'], 'k-',color=rescolors[0],lw=2)
    ax.fill_between(np.linspace(-.3, 9, nbins), mean_rep['dif4_low'], mean_rep['dif4_high'], color=rescolors[0], alpha=.2)
    p3b=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    ax.set_yticks([])

    ax.set_xlim([0.7,2.1])

    ax.legend([(p1[0]),(p3[0],p3b[0]),(p2[0],p2b[0])],['original','subtracted',"ZLP"],loc='upper right', fontsize = 22)
    

########## Load data ###############################################

fnamebase = 'data/'
mean_rep = pd.read_csv(fnamebase + 'Subtracted_spectra_1.45.csv')

##############################################################
def smooth(x, window_len, window='hanning'):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': 
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    index = int(window_len/2)
    return y[(index-1):-(index)]

##################################################################

lines = ['solid', 'dotted', 'dashed', 'dashdot']

nbins = 1861
wl = 4
          
#for i in ([4,5,6,7]):
for i in ([4,5,6]):
    j = i - 4
    ax = plt.subplot(gs[1])
    ax.set_xlim([.95,2.1])
    ax.set_ylim([-1e1,1.5e3])
    #ax.set_ylabel('Intensity (a.u.)', fontsize=18)
    ax.set_xlabel('Energy loss (eV)', fontsize=28)
    
    ax.tick_params(which='major',direction='in',length=10, labelsize=20)
    ax.tick_params(which='minor',length=10)
    ax.set_xticks([1,1.25, 1.5,1.75, 2, 2.5])
    label= '%(i)s'%{"i":str(i)}
    
    p1 = ax.plot(np.linspace(-.3, 9, nbins), smooth(mean_rep['dif%(i)s_median' % {"i": i}], wl), linestyle=lines[j],color=rescolors[j],lw=2, label= 'subtracted (sp %(i)s)'%{"i":i})
    ax.fill_between(np.linspace(-.3, 9, nbins), smooth(mean_rep['dif%(i)s_low'% {"i": i}], wl), smooth(mean_rep['dif%(i)s_high'% {"i": i}], wl), color=rescolors[j], alpha=.2)
    p1b =ax.fill(np.NaN,np.NaN,color=rescolors[j],alpha=0.2)
    ax.set_yticks([])
    ax.legend(loc='upper left', fontsize=22)
    plt.text(1.7,250,r"$\Delta E_{\rm I}=1.45~{\rm eV}$",fontsize=28)
    ax.set_xlim([.95,2.1])

    
plt.tight_layout()
plt.savefig('../plots/subtractedEELS_plot_sampleB_sp4.pdf')
print("Saved fig = ../plots/subtractedEELS_plot_sampleB_sp4.pdf")


