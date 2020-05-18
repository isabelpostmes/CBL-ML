###############################################
import numpy as np
import random
from numpy import loadtxt
import math
import scipy
import sklearn
from scipy import optimize
from scipy.optimize import leastsq
from io import StringIO
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
fig = plt.gcf()
###############################################


########### APPLY WINDOW TO SPECTRA #########
def window(x,y, minval, maxval):
    """Function applies a window to arrow"""
    
    low = next(i for i, val in enumerate(x) if val > minval)
    treshold_min = str(low)
    treshold_min = int(treshold_min)
    up = next(i for i, val in enumerate(x) if val > maxval)
    treshold_max = str(up)
    treshold_max = int(treshold_max)
    x = x[treshold_min:treshold_max]
    y = y[treshold_min:treshold_max]
    
    return x,y
    
# Spectrum1, read the intensity
ZLP_200_y1 = np.loadtxt("Data/May/MoS2_EELS-Spectrum_01_m1eV-9d24eV.txt")
ndat=int(len(ZLP_200_y1))
# Energy loss values
ZLP_200_x1 = np.zeros(ndat)
Eloss_min = -1 # eV
Eloss_max = +9.24 # eV

i=0

while(i<ndat):
    ZLP_200_x1[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat) 
    i = i + 1
    
    
# Vacuum peak
ZLP_200_yvac = np.loadtxt("Data/May/ZLP_200keV_m0d4eV-3d696eV.txt")
ndat=int(len(ZLP_200_yvac))
# Energy loss values
ZLP_200_xvac = np.zeros(ndat)
Eloss_min = -0.4 # eV
Eloss_max = +3.696 # eV
i=0

while(i<ndat):
    ZLP_200_xvac[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
    

ZLP_200_y2 = np.loadtxt("Data/May/MoS2_EELS-Spectrum_02_m1eV-9d24eV.txt")
ZLP_200_y3 = np.loadtxt("Data/May/MoS2_EELS-Spectrum_03_m1eV-9d24eV.txt")
ZLP_200_y4 = np.loadtxt("Data/May/MoS2_EELS-Spectrum_04_m1eV-9d24eV.txt")
ZLP_200_y5 = np.loadtxt("Data/May/MoS2_EELS-Spectrum_05_m1eV-9d24eV.txt")
ZLP_200_y6 = np.loadtxt("Data/May/MoS2_EELS-Spectrum_06_m1eV-9d24eV.txt")
ZLP_200_y7 = np.loadtxt("Data/May/MoS2_EELS-Spectrum_07_m1eV-9d24eV.txt")
ZLP_200_y8 = np.loadtxt("Data/May/MoS2_EELS-Spectrum_08_m1eV-9d24eV.txt")
ZLP_200_y9 = np.loadtxt("Data/May/MoS2_EELS-Spectrum_09_m1eV-9d24eV.txt")

ZLP_200_x2 = ZLP_200_x1
ZLP_200_x3 = ZLP_200_x1
ZLP_200_x4 = ZLP_200_x1
ZLP_200_x5 = ZLP_200_x1
ZLP_200_x6 = ZLP_200_x1
ZLP_200_x7 = ZLP_200_x1
ZLP_200_x8 = ZLP_200_x1
ZLP_200_x9 = ZLP_200_x1




window_min = -0.5
window_max = +9

ZLP_200_x1, ZLP_200_y1 = window(ZLP_200_x1, ZLP_200_y1, window_min, window_max)
ZLP_200_x2, ZLP_200_y2 = window(ZLP_200_x2, ZLP_200_y2, window_min, window_max)
ZLP_200_x3, ZLP_200_y3 = window(ZLP_200_x3, ZLP_200_y3, window_min, window_max)
ZLP_200_x4, ZLP_200_y4 = window(ZLP_200_x4, ZLP_200_y4, window_min, window_max)
ZLP_200_x5, ZLP_200_y5 = window(ZLP_200_x5, ZLP_200_y5, window_min, window_max)
ZLP_200_x6, ZLP_200_y6 = window(ZLP_200_x6, ZLP_200_y6, window_min, window_max)
ZLP_200_x7, ZLP_200_y7 = window(ZLP_200_x7, ZLP_200_y7, window_min, window_max)
ZLP_200_x8, ZLP_200_y8 = window(ZLP_200_x8, ZLP_200_y8, window_min, window_max)
ZLP_200_x9, ZLP_200_y9 = window(ZLP_200_x9, ZLP_200_y9, window_min, window_max)
#ZLP_200_xvac, ZLP_200_yvac = window(ZLP_200_xvac, ZLP_200_yvac, window_min, window_max)

log_y1, log_y2, log_y3, log_y4, log_y5, log_y6, log_y7, log_y8, log_y9 = np.log(ZLP_200_y1), np.log(ZLP_200_y2), \
np.log(ZLP_200_y3), np.log(ZLP_200_y4), np.log(ZLP_200_y5), np.log(ZLP_200_y6), np.log(ZLP_200_y7), np.log(ZLP_200_y8), np.log(ZLP_200_y9)

import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))

plt.plot(ZLP_200_x1 + 0.2025, ZLP_200_y1,color="tan",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="Sample 1")
plt.plot(ZLP_200_x2+ 0.2025, ZLP_200_y2,color="salmon",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="Sample 2")
plt.plot(ZLP_200_x3+ 0.2025, ZLP_200_y3,color="tomato",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="Sample 3")
plt.plot(ZLP_200_x4+ 0.2025, ZLP_200_y4,color="darksalmon",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="Sample 4")
plt.plot(ZLP_200_x5+ 0.2025, ZLP_200_y5,color="brown",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="Sample 5")
plt.plot(ZLP_200_x6+ 0.2025, ZLP_200_y6,color="moccasin",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="Sample 6")
plt.plot(ZLP_200_x7+ 0.2025, ZLP_200_y7,color="orange",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="Sample 7")
plt.plot(ZLP_200_x8+ 0.2025, ZLP_200_y8,color="khaki",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="Sample 8")
plt.plot(ZLP_200_x9+ 0.2025, ZLP_200_y9,color="red",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="Sample 9")

#plt.plot(ZLP_200_xvac, ZLP_200_yvac,color="steelblue",ls="solid",linewidth=2.0,marker="D",markersize=0.0, label='Vacuum')
plt.xlabel(r"Energy loss (eV)",fontsize=13)
plt.ylabel(r"Intensity (a.u.)",fontsize=13 )
#plt.ylim(0,8e5)
plt.xlim(-0.15,0.5)
plt.grid(True)
plt.title('MOS2')
plt.legend(fontsize=8, bbox_to_anchor=(1, 1))
plt.show()

plt.figure(figsize=(15,5))
plt.plot(ZLP_200_x1, log_y1,color="tan",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="Sample")
plt.plot(ZLP_200_x1, log_y2,color="salmon",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_200_x1, log_y3,color="tomato",ls="dotted",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_200_x1, log_y4,color="darksalmon",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_200_x1, log_y5,color="brown",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_200_x1, log_y6,color="moccasin",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_200_x1, log_y7,color="orange",ls="dotted",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_200_x1, log_y8,color="khaki",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_200_x1, log_y9,color="red",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)

plt.xlabel(r"Energy loss (eV)",fontsize=13)
plt.ylabel(r"Intensity (a.u.)",fontsize=13 )
#plt.ylim(0,8e5)
#plt.xlim(-0.15,0.15)
plt.grid(True)
plt.title('log')
plt.legend(fontsize=8, bbox_to_anchor=(1, 1))
plt.show()


def truncate(n, decimals=0):
    multiplier = 200 ** decimals
    return int(n * multiplier) / multiplier

def sequence(datafiles):
    x = list(np.linspace(0,datafiles-1, datafiles))
    return x

def prepare_x_data():
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = ZLP_200_x1, ZLP_200_x2, ZLP_200_x3, ZLP_200_x4, ZLP_200_x5, ZLP_200_x6, ZLP_200_x7, ZLP_200_x8, ZLP_200_x9
  
    datafiles = 9
    array = sequence(datafiles)
    mix_x = array
        
    for n, i in enumerate(array1):
            if i == 0:
                mix_x[n] = x1
            if i == 1:
                mix_x[n] = x2
            if i == 2:
                mix_x[n] = x3
            if i == 3:
                mix_x[n] = x4
            if i == 4:
                mix_x[n] = x5
            if i == 5:
                mix_x[n] = x6
            if i == 6:
                mix_x[n] = x7
            if i == 7:
                mix_x[n] = x8
            if i == 8:
                mix_x[n] = x9
            
    mix_x = np.concatenate(mix_x)
    
    return mix_x
    
def prepare_y_data():
        
    y1, y2, y3, y4, y5, y6, y7, y8, y9 = ZLP_200_y1, ZLP_200_y2, ZLP_200_y3, ZLP_200_y4, ZLP_200_y5, ZLP_200_y6, ZLP_200_y7, ZLP_200_y8, ZLP_200_y9
    
    normalization = max(y1)
    datafiles = 9
        
    array2 = sequence(datafiles)
    mix_y = array2

    for n, i in enumerate(array2):
            if i == 0:
                mix_y[n] = np.divide(y1, normalization)
            if i == 1:
                mix_y[n] = np.divide(y2, normalization)
            if i == 2:
                mix_y[n] = np.divide(y3, normalization)
            if i == 3:
                mix_y[n] = np.divide(y4, normalization)
            if i == 4:
                mix_y[n] = np.divide(y5, normalization)   
            if i == 5:
                mix_y[n] = np.divide(y6, normalization)
            if i == 6:
                mix_y[n] = np.divide(y7, normalization)  
            if i == 7:
                mix_y[n] = np.divide(y8, normalization)
            if i == 8:
                mix_y[n] = np.divide(y9, normalization)
                
    mix_y = np.concatenate(mix_y)

    return mix_y
    

def prepare_mix_data():

    x_train = prepare_x_data()
    y_train = prepare_y_data()  
    
    df = np.stack((x_train, y_train)).T
    df = np.matrix(df)
    
    x_train = df[:,0]
    y_train = df[:,1]

    return x_train, y_train

x1, x2, x3, x4, x5, x6, x7, x8, x9 = ZLP_200_x1, ZLP_200_x2, ZLP_200_x3, ZLP_200_x4, ZLP_200_x5, ZLP_200_x6, ZLP_200_x7, ZLP_200_x8, ZLP_200_x9

y1, y2, y3, y4, y5, y6, y7, y8, y9 = ZLP_200_y1, ZLP_200_y2, ZLP_200_y3, ZLP_200_y4, ZLP_200_y5, ZLP_200_y6, ZLP_200_y7, ZLP_200_y8, ZLP_200_y9

logy1, logy2, logy3, logy4, logy5, logy6, logy7, logy8, logy9 = log_y1, log_y2, log_y3, log_y4, log_y5, log_y6, log_y7, log_y8, log_y9
    


