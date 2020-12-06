###############################################
import numpy as np
from numpy import loadtxt
import pandas as pd
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


    

#""""""""""""#######"""""DATA t = 100 ms""""""########""""""""""""

# Spectrum1, read the intensity

ZLP_100_y1 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum1_m0d9567eV_8p6eV.txt")
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
ZLP_100_y2 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum2_m0d947eV_8p4681eV.txt")
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
ZLP_100_y3 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum3_m0d94eV_8p3603eV.txt")
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
ZLP_100_y4 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum4_m0d961eV_8p7343eV.txt")
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
ZLP_100_y5 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum5_m0d951eV_8p5967eV.txt")
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
ZLP_100_y6 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum6_m0d852eV_7p4173eV.txt")
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
ZLP_100_y7 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum7_m0d852eV_7p4173eV.txt")
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
ZLP_100_y8 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum8_m0d926eV_8p5091eV.txt")
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
ZLP_100_y9 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum9_m0d9262eV_8p1069eV.txt")
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
ZLP_100_y10 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum10_m0d9262eV_8p1915eV.txt")
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
ZLP_100_y11 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum11_m0d926eV_8p325eV.txt")
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
ZLP_100_y12 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum12_m0d926eV_8p1047eV.txt")
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
ZLP_100_y13 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum13_m0d926eV_8p0619eV.txt")
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
ZLP_100_y14 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum14_m0d926eV_8p0619eV.txt")
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
ZLP_100_y15 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_Spectrum15_m0d926eV_8p1479eV.txt")
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

file100_200 = pd.DataFrame()

for i, j in enumerate([ZLP_100_x1, ZLP_100_x2, ZLP_100_x3, ZLP_100_x4, ZLP_100_x5, ZLP_100_x6, ZLP_100_x7, ZLP_100_x8, ZLP_100_x9, ZLP_100_x10, ZLP_100_x11, ZLP_100_x12, ZLP_100_x13, ZLP_100_x14, ZLP_100_x15]):
    
    file100_200['x%(i)s'%{"i":i}] = j
for i, j in enumerate([ZLP_100_y1, ZLP_100_y2, ZLP_100_y3, ZLP_100_y4, ZLP_100_y5, ZLP_100_y6, ZLP_100_y7, ZLP_100_y8, ZLP_100_y9, ZLP_100_y10, ZLP_100_y11, ZLP_100_y12, ZLP_100_y13, ZLP_100_y14, ZLP_100_y15]):            
    file100_200['y%(i)s'%{"i":i}] = j
                
file100_200['time'] = 100
file100_200['energy'] = 200


          
plt.figure(figsize=(4,7))
plt.fill_between(ZLP_100_x1, 0, 800000, color='lightblue', alpha=.1)
plt.plot(ZLP_100_x1, ZLP_100_y1,color="tan",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="200 keV")
plt.plot(ZLP_100_x2, ZLP_100_y2,color="salmon",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x3, ZLP_100_y3,color="tomato",ls="dotted",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x4, ZLP_100_y4,color="darksalmon",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x5, ZLP_100_y5,color="brown",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x6, ZLP_100_y6,color="moccasin",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x7, ZLP_100_y7,color="orange",ls="dotted",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x8, ZLP_100_y8,color="khaki",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x9, ZLP_100_y9,color="red",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x10, ZLP_100_y10,color="darkorange",ls="dotted",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x11, ZLP_100_y11,color="crimson",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x12, ZLP_100_y12,color="palevioletred",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x13, ZLP_100_y13,color="violet",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x14, ZLP_100_y14,color="gold",ls="dotted",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x15, ZLP_100_y15,color="chocolate",ls="dotted",linewidth=2.0,marker="D",markersize=0.0)


plt.xlim([-.1, .1])
plt.ylim([0, 950000])

#fig.set_size_inches(12, 5)
plt.xlabel(r"Energy loss (eV)",fontsize=13)
plt.ylabel(r"Intensity (a.u.)",fontsize=13)
#plt.ylim(0,8e5)
#plt.xlim(-0.15,0.15)
plt.grid(True)
#plt.title('100ms')
plt.legend(fontsize=8, bbox_to_anchor=(1, 1))
plt.show()

