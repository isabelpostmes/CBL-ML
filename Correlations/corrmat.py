# Call all relevant libraries
import numpy as np
from numpy import loadtxt
import math
import scipy
#import sklearn
from scipy import optimize
from scipy.optimize import leastsq
from io import StringIO
import matplotlib.pyplot as plt
rescolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from matplotlib import gridspec
from  matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text',usetex=True)
###################################################

# Read the data
spectra = np.loadtxt("Data/Spectra.txt")
print(spectra)

# Compute corrmat
ndat=20
nsp=15
mean=np.zeros(ndat)
sigma=np.zeros(ndat)
bbin=np.zeros(ndat)
for idat in range(ndat):
    summ=0
    summ2=0
    bbin[idat]=idat
    for isp in range(nsp):
        summ = summ + spectra[idat][isp]/nsp
        summ2 = summ2 + pow(spectra[idat][isp],2)/nsp
    mean[idat]=summ
    sigma[idat] = pow(summ2 -summ**2, 0.5)


for idat in range(ndat):
    print(idat, sigma[idat]/mean[idat])


rho=np.zeros((ndat,ndat))
for idat in range(ndat):
    for jdat in range(ndat):
        sumd=0
        for isp in range(nsp):
            sumd = sumd + spectra[idat][isp]*spectra[jdat][isp]/nsp
        rho[idat][jdat] = (sumd - mean[idat]*mean[jdat])/( sigma[idat] * sigma[jdat] )

print(rho)


fig, ax = plt.subplots(figsize=(12,12))
im = ax.imshow(rho)

for i in range(ndat):
    for j in range(ndat):
        text = ax.text(j, i, np.round(rho[i, j], 2),
                       ha="center", va="center", color="black")

cbar = ax.figure.colorbar(im,shrink=0.76)
cbar.ax.set_ylabel(r"Correlation Coefficient~~$\rho(\Delta E_i, \Delta E_j)$", rotation=-90, va="bottom",fontsize=22)
        
ax.set_xlabel(r'$\Delta E$ bin', fontsize=26)
ax.set_ylabel(r'$\Delta E$ bin', fontsize=26)
ax.set_xticks(np.arange(ndat))
ax.set_yticks(np.arange(ndat))
#ax.legend(fontsize=20)

fig.tight_layout()
plt.savefig("corrmat.pdf")
print("saved plot: corrmat.pdf")

print("\n\n ****** Plot all spectra separately")

plt.clf()


fig, ax = plt.subplots(figsize=(5,3.5))

for isp in range(nsp):

    spc=np.zeros(ndat)
    for idat in range(ndat):
        spc[idat]=spectra[idat][isp]

    p=ax.plot(bbin,spc,color=rescolors[0],ls="dashed",lw=0.7)
    ax.tick_params(which='both',direction='in',labelsize=13,right=True)
    ax.tick_params(which='major',length=3)
    ax.tick_params(which='minor',length=2)
    ax.set_xlabel(r'$\Delta E~{\rm bin}$', fontsize=20)
    ax.set_ylabel(r'$\rm Intensity (a.u.)$', fontsize=20)
    ax.set_title(r"${\rm Vacuum},~t_{\exp}=100~{\rm ms},~E_b=200~{\rm keV}$",fontsize=18)
    ax.axes.yaxis.set_ticks([])

fig.tight_layout()
plt.savefig("spectra.pdf")
print("saved plot: spectra.pdf")
