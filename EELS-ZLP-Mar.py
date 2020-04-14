###############################################
import numpy as np
from numpy import loadtxt
import math
import scipy
import sklearn
from scipy import optimize
from scipy.optimize import leastsq
from io import StringIO
from scipy.signal import savgol_filter
###############################################

mode = 'Mixture'



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
    

#""""""""""""#######"""""DATA t = 100 ms""""""########""""""""""""

# Spectrum1, read the intensity

ZLP_100_y1 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum1_m0d9567eV_8p6eV.txt")
ndat=int(len(ZLP_100_y1))
# Energy loss values
ZLP_100_x1 = np.zeros(ndat)
Eloss_min = -0.9567 # eV
Eloss_max = +8.6 # eV
i=0
while(i<ndat):
    ZLP_100_x1[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum2, read the intensity
ZLP_100_y2 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum2_m0d947eV_8p4681eV.txt")
ndat=int(len(ZLP_100_y2))
# Energy loss values
ZLP_100_x2 = np.zeros(ndat)
Eloss_min = -0.947 # eV
Eloss_max = +8.468 # eV
i=0
while(i<ndat):
    ZLP_100_x2[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum3, read the intensity
ZLP_100_y3 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum3_m0d94eV_8p3603eV.txt")
ndat=int(len(ZLP_100_y3))
# Energy loss values
ZLP_100_x3 = np.zeros(ndat)
Eloss_min = -0.94 # eV
Eloss_max = +8.3603 # eV
i=0
while(i<ndat):
    ZLP_100_x3[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum4, read the intensity
ZLP_100_y4 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum4_m0d961eV_8p7343eV.txt")
ndat=int(len(ZLP_100_y4))
# Energy loss values
ZLP_100_x4 = np.zeros(ndat)
Eloss_min = -0.961 # eV
Eloss_max = +8.7343 # eV
i=0
while(i<ndat):
    ZLP_100_x4[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum5, read the intensity
ZLP_100_y5 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum5_m0d951eV_8p5967eV.txt")
ndat=int(len(ZLP_100_y5))
# Energy loss values
ZLP_100_x5 = np.zeros(ndat)
Eloss_min = -0.951 # eV
Eloss_max = +8.5967 # eV
i=0
while(i<ndat):
    ZLP_100_x5[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum6, read the intensity
ZLP_100_y6 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum6_m0d852eV_7p4173eV.txt")
ndat=int(len(ZLP_100_y6))
# Energy loss values
ZLP_100_x6 = np.zeros(ndat)
Eloss_min = -0.852 # eV
Eloss_max = +7.4173 # eV
i=0
while(i<ndat):
    ZLP_100_x6[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum7, read the intensity
ZLP_100_y7 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum7_m0d852eV_7p4173eV.txt")
ndat=int(len(ZLP_100_y7))
# Energy loss values
ZLP_100_x7 = np.zeros(ndat)
Eloss_min = -0.852 # eV
Eloss_max = +7.4173 # eV
i=0
while(i<ndat):
    ZLP_100_x7[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
# Spectrum8, read the intensity
ZLP_100_y8 = np.loadtxt("Data/Apr/ZLP_200keV_100ms_Spectrum8_m0d926eV_8p5091eV.txt")
ndat=int(len(ZLP_100_y8))
# Energy loss values
ZLP_100_x8 = np.zeros(ndat)
Eloss_min = -0.926 # eV
Eloss_max = +8.5091 # eV
i=0
while(i<ndat):
    ZLP_100_x8[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
# Spectrum9, read the intensity
ZLP_100_y9 = np.loadtxt("Data/Apr/ZLP_200keV_100ms_Spectrum9_m0d9262eV_8p1069eV.txt")
ndat=int(len(ZLP_100_y9))
# Energy loss values
ZLP_100_x9 = np.zeros(ndat)
Eloss_min = -0.9262 # eV
Eloss_max = +8.1069 # eV
i=0
while(i<ndat):
    ZLP_100_x9[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum10, read the intensity
ZLP_100_y10 = np.loadtxt("Data/Apr/ZLP_200keV_100ms_Spectrum10_m0d9262eV_8p1915eV.txt")
ndat=int(len(ZLP_100_y10))
# Energy loss values
ZLP_100_x10 = np.zeros(ndat)
Eloss_min = -0.9262 # eV
Eloss_max = +8.1915 # eV
i=0
while(i<ndat):
    ZLP_100_x10[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    

# Spectrum11, read the intensity
ZLP_100_y11 = np.loadtxt("Data/Apr/ZLP_200keV_100ms_Spectrum11_m0d926eV_8p325eV.txt")
ndat=int(len(ZLP_100_y11))
# Energy loss values
ZLP_100_x11 = np.zeros(ndat)
Eloss_min = -0.926 # eV
Eloss_max = +8.325 # eV
i=0
while(i<ndat):
    ZLP_100_x11[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
    
# Spectrum12, read the intensity
ZLP_100_y12 = np.loadtxt("Data/Apr/ZLP_200keV_100ms_Spectrum12_m0d926eV_8p1047eV.txt")
ndat=int(len(ZLP_100_y12))
# Energy loss values
ZLP_100_x12 = np.zeros(ndat)
Eloss_min = -0.926 # eV
Eloss_max = +8.1047 # eV
i=0
while(i<ndat):
    ZLP_100_x12[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1 

# Spectrum13, read the intensity
ZLP_100_y13 = np.loadtxt("Data/Apr/ZLP_200keV_100ms_Spectrum13_m0d926eV_8p0619eV.txt")
ndat=int(len(ZLP_100_y13))
# Energy loss values
ZLP_100_x13 = np.zeros(ndat)
Eloss_min = -0.926 # eV
Eloss_max = +8.0619 # eV
i=0
while(i<ndat):
    ZLP_100_x13[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1 

    # Spectrum14, read the intensity
ZLP_100_y14 = np.loadtxt("Data/Apr/ZLP_200keV_100ms_Spectrum14_m0d926eV_8p0619eV.txt")
ndat=int(len(ZLP_100_y14))
# Energy loss values
ZLP_100_x14 = np.zeros(ndat)
Eloss_min = -0.926 # eV
Eloss_max = +8.0619 # eV
i=0
while(i<ndat):
    ZLP_100_x14[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

    # Spectrum15, read the intensity
ZLP_100_y15 = np.loadtxt("Data/Apr/ZLP_200keV_100ms_Spectrum15_m0d926eV_8p1479eV.txt")
ndat=int(len(ZLP_100_y15))
# Energy loss values
ZLP_100_x15 = np.zeros(ndat)
Eloss_min = -0.926 # eV
Eloss_max = +8.1479 # eV
i=0
while(i<ndat):
    ZLP_100_x15[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
    
    
ZLP_100_x1, ZLP_100_y1 = window(ZLP_100_x1, ZLP_100_y1, -0.05, 0.05)
ZLP_100_x2, ZLP_100_y2 = window(ZLP_100_x2, ZLP_100_y2, -0.05, 0.05)
ZLP_100_x3, ZLP_100_y3 = window(ZLP_100_x3, ZLP_100_y3, -0.05, 0.05)
ZLP_100_x4, ZLP_100_y4 = window(ZLP_100_x4, ZLP_100_y4, -0.05, 0.05)
ZLP_100_x5, ZLP_100_y5 = window(ZLP_100_x5, ZLP_100_y5, -0.05, 0.05)
ZLP_100_x6, ZLP_100_y6 = window(ZLP_100_x6, ZLP_100_y6, -0.05, 0.05)
ZLP_100_x7, ZLP_100_y7 = window(ZLP_100_x7, ZLP_100_y7, -0.05, 0.05)
ZLP_100_x8, ZLP_100_y8 = window(ZLP_100_x8, ZLP_100_y8, -0.05, 0.05)
ZLP_100_x9, ZLP_100_y9 = window(ZLP_100_x9, ZLP_100_y9, -0.05, 0.05)
ZLP_100_x10, ZLP_100_y10 = window(ZLP_100_x10, ZLP_100_y10, -0.05, 0.05)
ZLP_100_x11, ZLP_100_y11 = window(ZLP_100_x11, ZLP_100_y11, -0.05, 0.05)
ZLP_100_x12, ZLP_100_y12 = window(ZLP_100_x12, ZLP_100_y12, -0.05, 0.05)
ZLP_100_x13, ZLP_100_y13 = window(ZLP_100_x13, ZLP_100_y13, -0.05, 0.05)
ZLP_100_x14, ZLP_100_y14 = window(ZLP_100_x14, ZLP_100_y14, -0.05, 0.05)
ZLP_100_x15, ZLP_100_y15 = window(ZLP_100_x15, ZLP_100_y15, -0.05, 0.05)


import matplotlib.pyplot as plt

plt.plot(ZLP_100_x1, ZLP_100_y1,color="blue",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#1")
plt.plot(ZLP_100_x2, ZLP_100_y2,color="red",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#2")
plt.plot(ZLP_100_x3, ZLP_100_y3,color="green",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#3")
plt.plot(ZLP_100_x4, ZLP_100_y4,color="navy",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#4")
plt.plot(ZLP_100_x5, ZLP_100_y5,color="black",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#5")
plt.plot(ZLP_100_x6, ZLP_100_y6,color="red",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#6")
plt.plot(ZLP_100_x7, ZLP_100_y7,color="orange",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#7")
plt.plot(ZLP_100_x8, ZLP_100_y8,color="blue",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#8")
plt.plot(ZLP_100_x9, ZLP_100_y9,color="red",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#9")
plt.plot(ZLP_100_x10, ZLP_100_y10,color="green",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#10")
plt.plot(ZLP_100_x11, ZLP_100_y11,color="navy",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#11")
plt.plot(ZLP_100_x12, ZLP_100_y12,color="black",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#12")
plt.plot(ZLP_100_x13, ZLP_100_y13,color="red",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#13")
plt.plot(ZLP_100_x14, ZLP_100_y14,color="orange",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#14")
plt.plot(ZLP_100_x15, ZLP_100_y15,color="orange",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#15")
plt.xlabel(r"Energy loss (eV)",fontsize=13)
plt.ylabel(r"Intensity (a.u.)",fontsize=13)
#plt.ylim(0,8e5)
#plt.xlim(-0.15,0.15)
plt.grid(True)
plt.title('200 keV, 100ms')
plt.legend(fontsize=12)
plt.show()


################### SECOND SUBSET OF DATA #######################

window_width = 100

#""""""""""""#######"""""DATA t = 10 ms""""""########""""""""""""

# Spectrum1, read the intensity

ZLP_10_y1 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum1_m0p3961eV_2p7361eV.txt")
ndat=int(len(ZLP_10_y1))
# Energy loss values
ZLP_10_x1 = np.zeros(ndat)
Eloss_min = -0.3961 # eV
Eloss_max = +2.7361 # eV
i=0
while(i<ndat):
    ZLP_10_x1[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum2, read the intensity
ZLP_10_y2 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum2_m0p3901eV_2p6711eV.txt")
ndat=int(len(ZLP_10_y2))
# Energy loss values
ZLP_10_x2 = np.zeros(ndat)
Eloss_min = -0.3901 # eV
Eloss_max = +2.6711 # eV
i=0
while(i<ndat):
    ZLP_10_x2[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1



# Spectrum3, read the intensity
ZLP_10_y3 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum3_m0p3825eV_2p9789eV.txt")
ndat=int(len(ZLP_10_y3))
# Energy loss values
ZLP_10_x3 = np.zeros(ndat)
Eloss_min = -0.3825 # eV
Eloss_max = +2.9789 # eV
i=0
while(i<ndat):
    ZLP_10_x3[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum4, read the intensity
ZLP_10_y4 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum4_m0p383eV_2p6339eV.txt")
ndat=int(len(ZLP_10_y4))
# Energy loss values
ZLP_10_x4 = np.zeros(ndat)
Eloss_min = -0.383 # eV
Eloss_max = +2.6339 # eV
i=0
while(i<ndat):
    ZLP_10_x4[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum5, read the intensity
ZLP_10_y5 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum5_m0p3749eV_2p5672eV.txt")
ndat=int(len(ZLP_10_y5))
# Energy loss values
ZLP_10_x5 = np.zeros(ndat)
Eloss_min = -0.3749 # eV
Eloss_max = +2.5672# eV
i=0
while(i<ndat):
    ZLP_10_x5[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
# Spectrum6, read the intensity
ZLP_10_y6 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum6_m0p4334eV_3p0336eV.txt")
ndat=int(len(ZLP_10_y6))
# Energy loss values
ZLP_10_x6 = np.zeros(ndat)
Eloss_min = -0.4334 # eV
Eloss_max = +3.0336# eV
i=0
while(i<ndat):
    ZLP_10_x6[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum7, read the intensity
ZLP_10_y7 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_Spectrum7_m0p10948eV_0p73658eV.txt")
ndat=int(len(ZLP_10_y7))
# Energy loss values
ZLP_10_x7 = np.zeros(ndat)
Eloss_min = -0.10948 # eV
Eloss_max = +0.73658 # eV
i=0
while(i<ndat):
    ZLP_10_x7[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum8, read the intensity
ZLP_10_y8 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_Spectrum8_m0p109eV_0p73979eV.txt")
ndat=int(len(ZLP_10_y8))
# Energy loss values
ZLP_10_x8 = np.zeros(ndat)
Eloss_min = -0.109 # eV
Eloss_max = +0.7397 # eV
i=0
while(i<ndat):
    ZLP_10_x8[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum9, read the intensity
ZLP_10_y9 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_Spectrum9_m0p11821eV_0p81652eV.txt")
ndat=int(len(ZLP_10_y9))
# Energy loss values
ZLP_10_x9 = np.zeros(ndat)
Eloss_min = -0.11821 # eV
Eloss_max = +0.81652 # eV
i=0
while(i<ndat):
    ZLP_10_x9[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
    
# Spectrum10, read the intensity
ZLP_10_y10 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_Spectrum10_m0p118eV_0p79739eV.txt")
ndat=int(len(ZLP_10_y10))
# Energy loss values
ZLP_10_x10 = np.zeros(ndat)
Eloss_min = -0.118 # eV
Eloss_max = +0.79739 # eV
i=0
while(i<ndat):
    ZLP_10_x10[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
    
# Spectrum11, read the intensity
ZLP_10_y11 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_Spectrum11_m0p118eV_0p81148eV.txt")
ndat=int(len(ZLP_10_y11))
# Energy loss values
ZLP_10_x11 = np.zeros(ndat)
Eloss_min = -0.118 # eV
Eloss_max = +0.81148 # eV
i=0
while(i<ndat):
    ZLP_10_x11[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
# Spectrum12, read the intensity
ZLP_10_y12 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_Spectrum12_m0p118eV_0p81148eV.txt")
ndat=int(len(ZLP_10_y12))
# Energy loss values
ZLP_10_x12 = np.zeros(ndat)
Eloss_min = -0.118 # eV
Eloss_max = +0.81148 # eV
i=0
while(i<ndat):
    ZLP_10_x12[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
# Spectrum13, read the intensity
ZLP_10_y13 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_Spectrum13_m0p14111eV_0p99667eV.txt")
ndat=int(len(ZLP_10_y13))
# Energy loss values
ZLP_10_x13 = np.zeros(ndat)
Eloss_min = -0.14111 # eV
Eloss_max = +0.99667 # eV
i=0
while(i<ndat):
    ZLP_10_x13[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
# Spectrum14, read the intensity
ZLP_10_y14 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_Spectrum14_m0p141eV_0p96117eV.txt")
ndat=int(len(ZLP_10_y14))
# Energy loss values
ZLP_10_x14 = np.zeros(ndat)
Eloss_min = -0.141 # eV
Eloss_max = +0.96117 # eV
i=0
while(i<ndat):
    ZLP_10_x14[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    


ZLP_10_x1, ZLP_10_y1 = window(ZLP_10_x1, ZLP_10_y1, -0.05, 0.05)
ZLP_10_x2, ZLP_10_y2 = window(ZLP_10_x2, ZLP_10_y2, -0.05, 0.05)
ZLP_10_x3, ZLP_10_y3 = window(ZLP_10_x3, ZLP_10_y3, -0.05, 0.05)
ZLP_10_x4, ZLP_10_y4 = window(ZLP_10_x4, ZLP_10_y4, -0.05, 0.05)
ZLP_10_x5, ZLP_10_y5 = window(ZLP_10_x5, ZLP_10_y5, -0.05, 0.05)
ZLP_10_x6, ZLP_10_y6 = window(ZLP_10_x6, ZLP_10_y6, -0.05, 0.05)
ZLP_10_x7, ZLP_10_y7 = window(ZLP_10_x7, ZLP_10_y7, -0.05, 0.05)
ZLP_10_x8, ZLP_10_y8 = window(ZLP_10_x8, ZLP_10_y8, -0.05, 0.05)
ZLP_10_x9, ZLP_10_y9 = window(ZLP_10_x9, ZLP_10_y9, -0.05, 0.05)
ZLP_10_x10, ZLP_10_y10 = window(ZLP_10_x10, ZLP_10_y10, -0.05, 0.05)
ZLP_10_x11, ZLP_10_y11 = window(ZLP_10_x11, ZLP_10_y11, -0.05, 0.05)
ZLP_10_x12, ZLP_10_y12 = window(ZLP_10_x12, ZLP_10_y12, -0.05, 0.05)
ZLP_10_x13, ZLP_10_y13 = window(ZLP_10_x13, ZLP_10_y13, -0.05, 0.05)
ZLP_10_x14, ZLP_10_y14 = window(ZLP_10_x14, ZLP_10_y14, -0.05, 0.05)
    
plt.plot(ZLP_10_x1, ZLP_10_y1 ,color="green",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#1")
plt.plot(ZLP_10_x2, ZLP_10_y2 ,color="red",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#2")
#plt.plot(ZLP_10_x3, ZLP_10_y3 ,color="blue",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#3")
plt.plot(ZLP_10_x4, ZLP_10_y4 ,color="pink",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#4")
plt.plot(ZLP_10_x5, ZLP_10_y5 ,color="lightblue",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#5")
plt.plot(ZLP_10_x6, ZLP_10_y6 ,color="orange",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#6")
plt.plot(ZLP_10_x7, ZLP_10_y7,color="orange",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#7")
plt.plot(ZLP_10_x8, ZLP_10_y8,color="blue",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#8")
plt.plot(ZLP_10_x9, ZLP_10_y9,color="red",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#9")
plt.plot(ZLP_10_x10, ZLP_10_y10,color="green",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#10")
plt.plot(ZLP_10_x11, ZLP_10_y11,color="navy",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#11")
plt.plot(ZLP_10_x12, ZLP_10_y12,color="black",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#12")
plt.plot(ZLP_10_x13, ZLP_10_y13,color="red",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#13")
plt.plot(ZLP_10_x14, ZLP_10_y14,color="orange",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#14")

plt.xlabel(r"Energy loss (eV)",fontsize=13)
plt.ylabel(r"Intensity (a.u.)",fontsize=13)
#plt.ylim(0,1e5)
plt.grid(True)
plt.title('200 keV, 10ms')
plt.legend()
plt.savefig("EELSData-ZLP-2.pdf")
plt.show()


#""""""""""""#######"""""DATA t = 2 ms""""""########""""""""""""


# Spectrum1, read the intensity
ndat=63
ZLP_2_y1 = np.loadtxt("Data/Feb/Spectrum1.txt")

# Energy loss values
ZLP_2_x1 = np.zeros(ndat)
Eloss_min = -0.102 # eV
Eloss_max = +0.0988 # eV

i=0
while(i<ndat):
    ZLP_2_x1[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum2, read the intensity
ndat=59
ZLP_2_y2 = np.loadtxt("Data/Feb/Spectrum2.txt")

# Energy loss values
ZLP_2_x2 = np.zeros(ndat)
Eloss_min = -0.102 # eV
Eloss_max = +0.0986 # eV

i=0
while(i<ndat):
    ZLP_2_x2[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum3, read the intensity
ndat=63
ZLP_2_y3 = np.loadtxt("Data/Feb/Spectrum3.txt")

# Energy loss values
ZLP_2_x3 = np.zeros(ndat)
Eloss_min = -0.0984 # eV
Eloss_max = +0.1015 # eV

i=0
while(i<ndat):
    ZLP_2_x3[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum4, read the intensity
ndat=52
ZLP_2_y4 = np.loadtxt("Data/Feb/Spectrum4.txt")

# Energy loss values
ZLP_2_x4 = np.zeros(ndat)
Eloss_min = -0.102 # eV
Eloss_max = +0.102 # eV

i=0
while(i<ndat):
    ZLP_2_x4[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum5, read the intensity
ndat=40
ZLP_2_y5 = np.loadtxt("Data/Feb/Spectrum5.txt")
# Energy loss values
ZLP_2_x5 = np.zeros(ndat)
Eloss_min = -0.0984 # eV
Eloss_max = +0.0984 # eV
i=0
while(i<ndat):
    ZLP_2_x5[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1   
    
# Spectrum6, read the intensity
ZLP_2_y6 = np.loadtxt("Data/Apr/ZLP_200keV_2ms_Spectrum6_m0p642eV_5p502eV.txt")
# Energy loss values
ndat = int(len(ZLP_2_y6))
ZLP_2_x6 = np.zeros(ndat)
Eloss_min = -0.642# eV
Eloss_max = +5.502 # eV
i=0
while(i<ndat):
    ZLP_2_x6[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
    
# Spectrum7, read the intensity
ZLP_2_y7 = np.loadtxt("Data/Apr/ZLP_200keV_2ms_Spectrum7_m0p642eV_5p4451eV.txt")
# Energy loss values
ndat = int(len(ZLP_2_y7))
ZLP_2_x7 = np.zeros(ndat)
Eloss_min = -0.642# eV
Eloss_max = +5.4451 # eV
i=0
while(i<ndat):
    ZLP_2_x7[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
    
    
# Spectrum8, read the intensity
ZLP_2_y8 = np.loadtxt("Data/Apr/ZLP_200keV_2ms_Spectrum8_m0p642eV_5p4734eV.txt")
# Energy loss values
ndat = int(len(ZLP_2_y8))
ZLP_2_x8 = np.zeros(ndat)
Eloss_min = -0.642# eV
Eloss_max = +5.4734 # eV
i=0
while(i<ndat):
    ZLP_2_x8[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
    
# Spectrum9, read the intensity
ZLP_2_y9 = np.loadtxt("Data/Apr/ZLP_200keV_2ms_Spectrum9_m0p642eV_5p5308eV.txt")
# Energy loss values
ndat = int(len(ZLP_2_y9))
ZLP_2_x9 = np.zeros(ndat)
Eloss_min = -0.642# eV
Eloss_max = +5.5308 # eV
i=0
while(i<ndat):
    ZLP_2_x9[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    

# Spectrum10, read the intensity
ZLP_2_y10 = np.loadtxt("Data/Apr/ZLP_200keV_2ms_Spectrum10_m0p6537eV_5p661eV.txt")
# Energy loss values
ndat = int(len(ZLP_2_y10))
ZLP_2_x10 = np.zeros(ndat)
Eloss_min = -0.6537# eV
Eloss_max = +5.661 # eV
i=0
while(i<ndat):
    ZLP_2_x10[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
    
# Spectrum11, read the intensity
ZLP_2_y11 = np.loadtxt("Data/Apr/ZLP_200keV_2ms_Spectrum11_m0p7012eV_6p1045eV.txt")
# Energy loss values
ndat = int(len(ZLP_2_y11))
ZLP_2_x11 = np.zeros(ndat)
Eloss_min = -0.7012# eV
Eloss_max = +6.1045 # eV
i=0
while(i<ndat):
    ZLP_2_x11[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1  
    
ZLP_2_x1, ZLP_2_y1 = window(ZLP_2_x1, ZLP_2_y1, -0.05, 0.05)
ZLP_2_x2, ZLP_2_y2 = window(ZLP_2_x2, ZLP_2_y2, -0.05, 0.05)
ZLP_2_x3, ZLP_2_y3 = window(ZLP_2_x3, ZLP_2_y3, -0.05, 0.05)
ZLP_2_x4, ZLP_2_y4 = window(ZLP_2_x4, ZLP_2_y4, -0.05, 0.05)
ZLP_2_x5, ZLP_2_y5 = window(ZLP_2_x5, ZLP_2_y5, -0.05, 0.05)
ZLP_2_x6, ZLP_2_y6 = window(ZLP_2_x6, ZLP_2_y6, -0.05, 0.05)
ZLP_2_x7, ZLP_2_y7 = window(ZLP_2_x7, ZLP_2_y7, -0.05, 0.05)
ZLP_2_x8, ZLP_2_y8 = window(ZLP_2_x8, ZLP_2_y8, -0.05, 0.05)
ZLP_2_x9, ZLP_2_y9 = window(ZLP_2_x9, ZLP_2_y9, -0.05, 0.05)
ZLP_2_x10, ZLP_2_y10 = window(ZLP_2_x10, ZLP_2_y10, -0.05, 0.05)
ZLP_2_x11, ZLP_2_y11 = window(ZLP_2_x11, ZLP_2_y11, -0.05, 0.05)

plt.plot(ZLP_2_x1, ZLP_2_y1 ,color="green",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#1")
plt.plot(ZLP_2_x2, ZLP_2_y2 ,color="red",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#2")
plt.plot(ZLP_2_x3, ZLP_2_y3 ,color="blue",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#3")
plt.plot(ZLP_2_x4, ZLP_2_y4 ,color="pink",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#4")
plt.plot(ZLP_2_x5, ZLP_2_y5 ,color="lightblue",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#5")
plt.plot(ZLP_2_x6, ZLP_2_y6 ,color="orange",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#6")
plt.plot(ZLP_2_x7, ZLP_2_y7,color="orange",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#7")
plt.plot(ZLP_2_x8, ZLP_2_y8,color="blue",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#8")
plt.plot(ZLP_2_x9, ZLP_2_y9,color="red",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#9")
plt.plot(ZLP_2_x10, ZLP_2_y10,color="green",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#10")
plt.plot(ZLP_2_x11, ZLP_2_y11,color="navy",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#11")
plt.xlabel(r"Energy loss (eV)",fontsize=13)
plt.ylabel(r"Intensity (a.u.)",fontsize=13)
#plt.ylim(0,1e5)
plt.grid(True)
plt.title('200 keV, 2ms')
plt.legend()
plt.savefig("EELSData-ZLP-3.pdf")
plt.show()
print("\n ************************ Data files have been prepared ***************************** \n")


N_train = 50
N_data = N_train
N_val = 20

N_test = N_val


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

import random

def sequence(datafiles):
    x = list(np.linspace(0,datafiles-1, datafiles))
    return x

def prepare_x_data(time):
    
        if time == 10:
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13 = ZLP_10_x1, ZLP_10_x2, ZLP_10_x4, ZLP_10_x5, ZLP_10_x6, ZLP_10_x7, ZLP_10_x8, ZLP_00_x9, ZLP_10_x10, ZLP_10_x11, ZLP_10_x12, ZLP_10_x13, ZLP_10_x14
            datafiles = 13
        if time == 100:
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 = ZLP_100_x1, ZLP_100_x2, ZLP_100_x3, ZLP_100_x4, ZLP_100_x5, ZLP_100_x6, ZLP_100_x7, ZLP_100_x8, ZLP_100_x9, ZLP_100_x10, ZLP_100_x11, ZLP_100_x12, ZLP_100_x13, ZLP_100_x14, ZLP_100_x15
            datafiles = 15
        if time == 2:
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = ZLP_2_x1, ZLP_2_x2, ZLP_2_x3, ZLP_2_x4, ZLP_2_x5, ZLP_2_x6, ZLP_2_x7, ZLP_2_x8, ZLP_2_x9, ZLP_2_x10, ZLP_2_x11 
            datafiles = 5
            
        array1 = sequence(datafiles)
        mix_x = array1
        
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
            if i == 9: 
                mix_x[n] = x10
            if i == 10:
                mix_x[n] = x11
            if i == 11:
                mix_x[n] = x12
            if i == 12:
                mix_x[n] = x13
            if i == 13:
                mix_x[n] = x14
            if i == 14:
                mix_x[n] = x15    

        mix_x = np.concatenate(mix_x)
    
        return mix_x
    
def prepare_y_data(time):
        
        if time == 10:
            y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13 = ZLP_10_y1, ZLP_10_y2, ZLP_10_y4, ZLP_10_y5, ZLP_10_y6, ZLP_10_y7, ZLP_10_y8, ZLP_10_y9, ZLP_10_y10, ZLP_10_y11, ZLP_10_y12, ZLP_10_y13, ZLP_10_y14
            normalization = max(y1)
            datafiles = 13
        if time == 100:
            y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15 = ZLP_100_y1, ZLP_100_y2, ZLP_100_y3, ZLP_100_y4, ZLP_100_y5, ZLP_100_y6, ZLP_100_y7, ZLP_100_y8, ZLP_100_y9, ZLP_100_y10, ZLP_100_y11, ZLP_100_y12, ZLP_100_y13, ZLP_100_y14, ZLP_100_y15
            normalization = max(ZLP_10_y1)
            datafiles = 15
        if time == 2:
            y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11 = ZLP_2_y1, ZLP_2_y2, ZLP_2_y3, ZLP_2_y4, ZLP_2_y5, ZLP_2_y6, ZLP_2_y7, ZLP_2_y8, ZLP_2_y9, ZLP_2_y10, ZLP_2_y11 
            datafiles = 11
            normalization = max(ZLP_10_y1)
            datafiles = 5

        y1 = np.divide(y1, normalization)
        y2 = np.divide(y2, normalization)
        y3 = np.divide(y3, normalization)
        y4 = np.divide(y4, normalization)
        y5 = np.divide(y5, normalization)
        y6 = np.divide(y6, normalization)
        y7 = np.divide(y7, normalization)
        y8 = np.divide(y8, normalization)
        y9 = np.divide(y9, normalization)
        y10 = np.divide(y10, normalization)
        y11 = np.divide(y11, normalization)
        if datafiles >= 12:
            y12 = np.divide(y12, normalization)
            y13 = np.divide(y13, normalization)
            if datafiles >= 14:
                y14 = np.divide(y14, normalization)
                y15 = np.divide(y15, normalization)
        
        array2 = sequence(datafiles)
        mix_y = array2

        for n, i in enumerate(array2):
            if i == 0:
                mix_y[n] = y1
            if i == 1:
                mix_y[n] = y2
            if i == 2:
                mix_y[n] = y3
            if i == 3:
                mix_y[n] = y4
            if i == 4:
                mix_y[n] = y5   
            if i == 5:
                mix_y[n] = y6
            if i == 6:
                mix_y[n] = y7  
            if i == 7:
                mix_y[n] = y8
            if i == 8:
                mix_y[n] = y9
            if i == 9: 
                mix_y[n] = y10
            if i == 10:
                mix_y[n] = y11
            if i == 11:
                mix_y[n] = y12
            if i == 12:
                mix_y[n] = y13
            if i == 13:
                mix_y[n] = y14
            if i == 14:
                mix_y[n] = y15    
        
        mix_y = np.concatenate(mix_y)

        return mix_y
    
from sklearn.model_selection import train_test_split

def prepare_mix_data(time):

    x_train = prepare_x_data(time)
    y_train = prepare_y_data(time)  
    
    df = np.stack((x_train, y_train)).T
    df = np.matrix(df)
    
    #splitting data randomly to train and test using the sklearn library
    df_train, df_test = train_test_split(df, test_size=0.1)
    x_train = df_train[:,0]
    y_train = df_train[:,1]
    x_val = df_test[:,0]
    y_val = df_test[:,1]
    
    
    x_train = x_train * 20
    x_val = x_val * 20
    return x_train, y_train, x_val, y_val

