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
amp = 2700
pars = [amp, 1.59, 1.257]
pars_high = [amp, 1.59, (1.26 + 0.3)]
pars_low = [amp, 1.59, (1.26 - 0.74)]
          
for i in range(1):
    ax = plt.subplot(gs[i])
    ax.set_xlim([1.3,4.5])
    ax.set_ylim([-1e1,4.5e3])
    ax.set_ylabel('Intensity (a.u.)', fontsize=26)
    ax.set_xlabel('Energy loss (eV)', fontsize=25)
    #ax.set_xticklabels(labels="x",rotation=0, fontsize=20)
    #plt.rc('xtick',labelsize=50)
    
    ax.tick_params(which='major',direction='in',length=7)
    ax.tick_params(which='minor',length=8)
        
    
    ax.set_xticks([1.5, 2, 2.5, 3, 3.5 , 4, 4.5])
    
    p1=ax.plot(np.linspace(-.3, 12, nbins), mean_rep['spectrum14'], 'k--',ls="dashdot",lw=2)

    
    p2=ax.plot(np.linspace(-.3, 12, nbins), mean_rep['match14_median'], color=rescolors[1],ls="dashed",lw=2)
    ax.fill_between(np.linspace(-.3, 12, nbins), mean_rep['match14_low'], mean_rep['match14_high'], \
                    color=rescolors[1], alpha=.2)
    p2b=ax.fill(np.NaN,np.NaN,color=rescolors[1],alpha=0.2)
    
    p3 = ax.plot(np.linspace(-.3, 12, nbins), mean_rep['dif14_median'], 'k-',color=rescolors[0],lw=2)
    ax.fill_between(np.linspace(-.3, 12, nbins), mean_rep['dif14_low'], mean_rep['dif14_high'], color=rescolors[0], alpha=.2)
    p3b=ax.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    ax.set_yticks([])

    ax.legend([(p1[0]),(p3[0],p3b[0]),(p2[0],p2b[0])],['original','subtracted',"ZLP"],loc='upper right', fontsize = 19)
    ax.tick_params(length= 10, labelsize=18)
    ax.tick_params(which='major', length= 10, labelsize=20)
    ax.tick_params(which='minor', length= 10, labelsize=10)
    
    axins = ax.inset_axes([0.55, 0.10, 0.43, 0.40])
    
    axins.get_xaxis().set_visible(True)
    axins.get_yaxis().set_visible(True)
    axins.spines['right'].set_visible(True)
    axins.spines['top'].set_visible(True)
    axins.set_xticks([1, 2, 3, 4])
    axins.set_xlim([1,3.02])
    axins.set_ylim([-1e3, 5e3])
    
    p1=axins.plot(np.linspace(-.3, 12, nbins), mean_rep['dif14_median'], 'k-', alpha=.8, label='subtracted',color=rescolors[0])
    axins.fill_between(np.linspace(-.3, 12, nbins), mean_rep['dif14_low'], \
                       mean_rep['dif14_high'], color=rescolors[0],alpha=0.2,lw=2)
    p1b=axins.fill(np.NaN,np.NaN,color=rescolors[0],alpha=0.2)
    
    
    x = np.linspace(1.5, 2.7, 100)
    p2=axins.plot(x, bandgap(x, *pars), label='model fit',color=rescolors[2],lw=2,ls="dashed")
    axins.fill_between(x, bandgap(x, *pars_low), \
                       bandgap(x, *pars_high), color=rescolors[2], alpha=.2)
    p2b=axins.fill(np.NaN,np.NaN,color=rescolors[2],alpha=0.2)
    axins.tick_params(which='both',direction='in', labelsize=12,right=True)
    axins.tick_params(which='major',length=10)
    axins.tick_params(which='minor',length=10)
    axins.set_yticks([])
    axins.legend([(p1[0],p1b[0]),(p2[0],p2b[0])],['subtracted','model fit'],loc='upper left', fontsize=15)
    


###################### Load data ################################################

df_dx = pd.read_csv('data/derivatives.csv')

ratio1 = np.divide(df_dx['derivative y14'], df_dx['derivative y17'])
ratio2 = np.divide(df_dx['derivative y14'], df_dx['derivative y22'])
ratio3 = np.divide(df_dx['derivative y14'], df_dx['derivative y23'])

total_ratio = (ratio1 + ratio2 + ratio3)/3

########################################################################

ax = plt.subplot(gs[1])

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

cm_subsection = np.linspace(0,1,24) 
colors = [cm.viridis(x) for x in cm_subsection]

hfont = rc('font',**{'family':'sans-serif','sans-serif':['Sans Serif']})

    
ax.set_xlim([0,6])
ax.tick_params(which='major',direction='in',length=7)
ax.tick_params(which='minor',length=8)
plt.axhline(y=0, color='darkgray', linewidth=1.5,linestyle='--')
plt.axhline(y=0.9, color='darkgray', linewidth=1.5,linestyle='--')
plt.axvline(x=0, color='darkgray', linestyle='--', linewidth = 1.5)

for j in ([14]):   
    ax.axhline(y=1, linestyle='--', color='gray')
    
    p1 = ax.plot(df_dx['x%(i)s'% {"i": j}], ratio1, color='black', label='Individual',lw=2,ls="dashed")
    p1 = ax.plot(df_dx['x%(i)s'% {"i": j}], ratio2, color='black',ls="dashed",lw=2,)
    p1 = ax.plot(df_dx['x%(i)s'% {"i": j}], ratio3, color='black',ls="dashed",lw=2,)
    p1 = ax.plot(df_dx['x%(i)s'% {"i": j}], total_ratio, label='Average',lw=3)
    # ax.set_title('Derivatives ratio', fontsize=24)
    ax.set_ylim([-0.3, 1.5])
    ax.set_xlim([1,3.2])   
    ax.set_ylabel(r'$dI/d \Delta E |_{\rm sample} \,/\,dI/d \Delta E|_{\rm vacuum} $', fontsize=25)
    ax.legend(fontsize=23)  
    ax.set_xlabel('Energy loss (eV)', fontsize=25)
    ax.tick_params(length= 10, labelsize=18)
    ax.tick_params(which='major', length= 10, labelsize=20)
    ax.tick_params(which='minor', length= 10, labelsize=10)
    ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
    

plt.tight_layout()
plt.savefig('SubtractedEELS_plot_sp14.pdf')
print("Saved fig = SubtractedEELS_plot_sp14.pdf")
