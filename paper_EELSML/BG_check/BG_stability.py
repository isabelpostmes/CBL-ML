#!/usr/bin/env python
# coding: utf-8

import os,sys
import lhapdf
import numpy as np
import matplotlib.pyplot as py
from matplotlib import gridspec
from  matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text',usetex=True)
from pylab import *

ncols,nrows=2,1
py.figure(figsize=(ncols*5,nrows*3.5))
#py.suptitle(plottitle, fontsize=20)
gs = gridspec.GridSpec(nrows,ncols)
rescolors = py.rcParams['axes.prop_cycle'].by_key()['color']

ax = py.subplot(gs[0])

BG=np.loadtxt("BG_results.txt")

nsp=6
spl=['14','15','16','19','20','21']

position = [0.85, 0.7, 0.55, 0.40,0.25,0.10]

for i in range(nsp):

    ax.errorbar((BG[i,2]+BG[i,3])/2,nsp-i,xerr=(BG[i,2]-BG[i,3])/2, marker="o",markersize=0, lw=2,fmt='o',color=rescolors[i],capthick=2,capsize=5)
    ax.plot(BG[i,1],nsp-i, marker="o",markersize=8, lw=2,color=rescolors[i])
    label="sp"+spl[i]
    ax.text(0.830,position[i],label,fontsize=12,transform=ax.transAxes,color=rescolors[i])


    
ax.tick_params(which='both',direction='in',labelsize=15,right=True)
ax.set_ylim(0,nsp+1)
ax.set_yticks([])
ax.set_xlabel(r"$b$",fontsize=20)
ax.set_xlim(0.27,2.5)

ax = py.subplot(gs[1])


for i in range(nsp):

    ax.errorbar((BG[i,5]+BG[i,6])/2,nsp-i,xerr=(BG[i,5]-BG[i,6])/2, marker="o",markersize=0, lw=2,fmt='o',color=rescolors[i],capthick=2,capsize=5)
    ax.plot(BG[i,4],nsp-i, marker="o",markersize=8, lw=2,color=rescolors[i])
    label="sp"+spl[i]
    ax.text(0.830,position[i],label,fontsize=12,transform=ax.transAxes,color=rescolors[i])

    
ax.tick_params(which='both',direction='in',labelsize=15,right=True)
ax.set_ylim(0,nsp+1)
ax.set_yticks([])
ax.set_xlabel(r"$E_{\rm BG}~({\rm eV})$",fontsize=20)
ax.set_xlim(0.5,2.5)



py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig('bg_stability.pdf')
print('output plot: bg_stability.pdf')








