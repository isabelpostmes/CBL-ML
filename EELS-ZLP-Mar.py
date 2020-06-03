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
import matplotlib.pyplot as plt
fig = plt.gcf()
###############################################

mode = 'Mixture'

window_min = -0.2
window_max = 0.5

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
    
# Spectrum101, read the intensity
ZLP_100_y101 = np.loadtxt("Data/Apr/ZLP_200keV_100ms_0d3mm_Spectrum1_m1eV_3p9951eV.txt")
ndat=int(len(ZLP_100_y101))
# Energy loss values
ZLP_100_x101 = np.zeros(ndat)
Eloss_min = -1 # eV
Eloss_max = +3.9951 # eV
i=0
while(i<ndat):
    ZLP_100_x101[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum102, read the intensity
ZLP_100_y102 = np.loadtxt("Data/Apr/ZLP_200keV_100ms_0d3mm_Spectrum2_m1eV_3p9951eV.txt")
ndat=int(len(ZLP_100_y102))
# Energy loss values
ZLP_100_x102 = np.zeros(ndat)
Eloss_min = -1 # eV
Eloss_max = +3.9951 # eV
i=0
while(i<ndat):
    ZLP_100_x102[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum103, read the intensity
ZLP_100_y103 = np.loadtxt("Data/Apr/ZLP_200keV_100ms_0d3mm_Spectrum3_m1eV_3p9709eV.txt")
ndat=int(len(ZLP_100_y103))
# Energy loss values
ZLP_100_x103 = np.zeros(ndat)
Eloss_min = -1 # eV
Eloss_max = +3.9709 # eV
i=0
while(i<ndat):
    ZLP_100_x103[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
    
# Spectrum104, read the intensity
ZLP_100_y104 = np.loadtxt("Data/Apr/ZLP_200keV_100ms_0d3mm_Spectrum4_m1eV_3p9113eV.txt")
ndat=int(len(ZLP_100_y104))
# Energy loss values
ZLP_100_x104 = np.zeros(ndat)
Eloss_min = -1 # eV
Eloss_max = +3.9113 # eV
i=0
while(i<ndat):
    ZLP_100_x104[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1        

# Spectrum201, read the intensity
ZLP_100_y201 = np.loadtxt("Data/Apr/ZLP_60keV_100ms_Spectrum1_m0d5362eV_5p4322eV.txt")
ndat=int(len(ZLP_100_y201))
# Energy loss values
ZLP_100_x201 = np.zeros(ndat)
Eloss_min = -0.5362 # eV
Eloss_max = +5.4322 # eV
i=0
while(i<ndat):
    ZLP_100_x201[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1  
    
    
# Spectrum202, read the intensity
ZLP_100_y202 = np.loadtxt("Data/Apr/ZLP_60keV_100ms_Spectrum2_m0d536eV_5p303eV.txt")
ndat=int(len(ZLP_100_y202))
# Energy loss values
ZLP_100_x202 = np.zeros(ndat)
Eloss_min = -0.536 # eV
Eloss_max = +5.303 # eV
i=0
while(i<ndat):
    ZLP_100_x202[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1      
      
# Spectrum203, read the intensity
ZLP_100_y203 = np.loadtxt("Data/Apr/ZLP_60keV_100ms_Spectrum3_m0d536eV_5p4625eV.txt")
ndat=int(len(ZLP_100_y203))
# Energy loss values
ZLP_100_x203 = np.zeros(ndat)
Eloss_min = -0.536 # eV
Eloss_max = +5.4625 # eV
i=0
while(i<ndat):
    ZLP_100_x203[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1 
    
    
# Spectrum204, read the intensity
ZLP_100_y204 = np.loadtxt("Data/Apr/ZLP_60keV_100ms_Spectrum4_m0d536eV_5p3977eV.txt")
ndat=int(len(ZLP_100_y204))
# Energy loss values
ZLP_100_x204 = np.zeros(ndat)
Eloss_min = -0.536 # eV
Eloss_max = +5.3977 # eV
i=0
while(i<ndat):
    ZLP_100_x204[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1  
    
# Spectrum205, read the intensity
ZLP_100_y205 = np.loadtxt("Data/Apr/ZLP_60keV_100ms_Spectrum5_m0d536eV_5p5966eV.txt")
ndat=int(len(ZLP_100_y205))
# Energy loss values
ZLP_100_x205 = np.zeros(ndat)
Eloss_min = -0.536 # eV
Eloss_max = +5.5966 # eV
i=0
while(i<ndat):
    ZLP_100_x205[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1      
                  
        
    # Spectrum206, read the intensity
ZLP_100_y206 = np.loadtxt("Data/Apr/ZLP_60keV_100ms_Spectrum6_m0d536eV_5p5288eV.txt")
ndat=int(len(ZLP_100_y206))
# Energy loss values
ZLP_100_x206 = np.zeros(ndat)
Eloss_min = -0.536 # eV
Eloss_max = +5.5288 # eV
i=0
while(i<ndat):
    ZLP_100_x206[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1  
    
    # Spectrum207, read the intensity
ZLP_100_y207 = np.loadtxt("Data/Apr/ZLP_60keV_100ms_Spectrum7_m0d536eV_5p3977eV.txt")
ndat=int(len(ZLP_100_y207))
# Energy loss values
ZLP_100_x207 = np.zeros(ndat)
Eloss_min = -0.536 # eV
Eloss_max = +5.3977 # eV
i=0
while(i<ndat):
    ZLP_100_x207[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1  
    
ZLP_100_x1, ZLP_100_y1 = window(ZLP_100_x1, ZLP_100_y1, window_min, window_max)
ZLP_100_x2, ZLP_100_y2 = window(ZLP_100_x2, ZLP_100_y2, window_min, window_max)
ZLP_100_x3, ZLP_100_y3 = window(ZLP_100_x3, ZLP_100_y3, window_min, window_max)
ZLP_100_x4, ZLP_100_y4 = window(ZLP_100_x4, ZLP_100_y4, window_min, window_max)
ZLP_100_x5, ZLP_100_y5 = window(ZLP_100_x5, ZLP_100_y5, window_min, window_max)
ZLP_100_x6, ZLP_100_y6 = window(ZLP_100_x6, ZLP_100_y6, window_min, window_max)
ZLP_100_x7, ZLP_100_y7 = window(ZLP_100_x7, ZLP_100_y7, window_min, window_max)
ZLP_100_x8, ZLP_100_y8 = window(ZLP_100_x8, ZLP_100_y8, window_min, window_max)
ZLP_100_x9, ZLP_100_y9 = window(ZLP_100_x9, ZLP_100_y9, window_min, window_max)
ZLP_100_x10, ZLP_100_y10 = window(ZLP_100_x10, ZLP_100_y10, window_min, window_max)
ZLP_100_x11, ZLP_100_y11 = window(ZLP_100_x11, ZLP_100_y11, window_min, window_max)
ZLP_100_x12, ZLP_100_y12 = window(ZLP_100_x12, ZLP_100_y12, window_min, window_max)
ZLP_100_x13, ZLP_100_y13 = window(ZLP_100_x13, ZLP_100_y13, window_min, window_max)
ZLP_100_x14, ZLP_100_y14 = window(ZLP_100_x14, ZLP_100_y14, window_min, window_max)
ZLP_100_x15, ZLP_100_y15 = window(ZLP_100_x15, ZLP_100_y15, window_min, window_max)

ZLP_100_x201, ZLP_100_y201 = window(ZLP_100_x201, ZLP_100_y201, window_min, window_max)
ZLP_100_x202, ZLP_100_y202 = window(ZLP_100_x202, ZLP_100_y202, window_min, window_max)
ZLP_100_x203, ZLP_100_y203 = window(ZLP_100_x203, ZLP_100_y203, window_min, window_max)
ZLP_100_x204, ZLP_100_y204 = window(ZLP_100_x204, ZLP_100_y204, window_min, window_max)
ZLP_100_x205, ZLP_100_y205 = window(ZLP_100_x205, ZLP_100_y205, window_min, window_max)
ZLP_100_x206, ZLP_100_y206 = window(ZLP_100_x206, ZLP_100_y206, window_min, window_max)
ZLP_100_x207, ZLP_100_y207 = window(ZLP_100_x207, ZLP_100_y207, window_min, window_max)

import matplotlib.pyplot as plt

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

plt.plot(ZLP_100_x201, ZLP_100_y201,color="forestgreen",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="60 keV")
plt.plot(ZLP_100_x202, ZLP_100_y202,color="limegreen",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x203, ZLP_100_y203,color="green",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x204, ZLP_100_y204,color="turquoise",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x205, ZLP_100_y205,color="seagreen",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x206, ZLP_100_y206,color="lightblue",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_100_x207, ZLP_100_y207,color="teal",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)

#fig.set_size_inches(12, 5)
plt.xlabel(r"Energy loss (eV)",fontsize=13)
plt.ylabel(r"Intensity (a.u.)",fontsize=13)
#plt.ylim(0,8e5)
#plt.xlim(-0.15,0.15)
plt.grid(True)
plt.title('100ms')
plt.legend(fontsize=8, bbox_to_anchor=(1, 1))
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
    
# Spectrum15, read the intensity
ZLP_10_y15 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_Spectrum16_m0p18768eV_1p29638eV.txt")
ndat=int(len(ZLP_10_y15))
# Energy loss values
ZLP_10_x15 = np.zeros(ndat)
Eloss_min = -0.18768 # eV
Eloss_max = +1.29638# eV
i=0
while(i<ndat):
    ZLP_10_x15[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
# Spectrum6, read the intensity
ZLP_10_y16 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_Spectrum17_m0p188eV_1p28719eV.txt")
ndat=int(len(ZLP_10_y16))
# Energy loss values
ZLP_10_x16 = np.zeros(ndat)
Eloss_min = -0.188 # eV
Eloss_max = +1.28719# eV
i=0
while(i<ndat):
    ZLP_10_x16[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum7, read the intensity
ZLP_10_y17 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_Spectrum18_m0p188eV_1p28156eV.txt")
ndat=int(len(ZLP_10_y17))
# Energy loss values
ZLP_10_x17 = np.zeros(ndat)
Eloss_min = -0.188 # eV
Eloss_max = +1.28156 # eV
i=0
while(i<ndat):
    ZLP_10_x17[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    

# Spectrum18, read the intensity
ZLP_10_y18 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_Spectrum19_m0p188eV_1p29286eV.txt")
ndat=int(len(ZLP_10_y18))
# Energy loss values
ZLP_10_x18 = np.zeros(ndat)
Eloss_min = -0.188 # eV
Eloss_max = +1.29286 # eV
i=0
while(i<ndat):
    ZLP_10_x18[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum19, read the intensity
ZLP_10_y19 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_Spectrum20_m0p188eV_1p29858eV.txt")
ndat=int(len(ZLP_10_y19))
# Energy loss values
ZLP_10_x19 = np.zeros(ndat)
Eloss_min = -0.188 # eV
Eloss_max = +1.29858 # eV
i=0
while(i<ndat):
    ZLP_10_x19[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
### Aperture files
    
# Spectrum101, read the intensity
ZLP_10_y101 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_0d3mm_Spectrum1_m0d7456eV_2p8136eV.txt")
ndat=int(len(ZLP_10_y101))
# Energy loss values
ZLP_10_x101 = np.zeros(ndat)
Eloss_min = -0.7456 # eV
Eloss_max = +2.8136 # eV
i=0
while(i<ndat):
    ZLP_10_x101[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum102, read the intensity
ZLP_10_y102 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_0d3mm_Spectrum2_m0d7456eV_2p8744eV.txt")
ndat=int(len(ZLP_10_y102))
# Energy loss values
ZLP_10_x102 = np.zeros(ndat)
Eloss_min = -0.7456 # eV
Eloss_max = +2.8744 # eV
i=0
while(i<ndat):
    ZLP_10_x102[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum103, read the intensity
ZLP_10_y103 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_0d3mm_Spectrum3_m0d6756eV_2p5573eV.txt")
ndat=int(len(ZLP_10_y103))
# Energy loss values
ZLP_10_x103 = np.zeros(ndat)
Eloss_min = -0.6756 # eV
Eloss_max = +2.5573 # eV
i=0
while(i<ndat):
    ZLP_10_x103[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
    
# Spectrum104, read the intensity
ZLP_10_y104 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_0d3mm_Spectrum4_m0d676eV_2p5512eV.txt")
ndat=int(len(ZLP_10_y104))
# Energy loss values
ZLP_10_x104 = np.zeros(ndat)
Eloss_min = -0.676 # eV
Eloss_max = +2.5512 # eV
i=0
while(i<ndat):
    ZLP_10_x104[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1   
    
# Spectrum105, read the intensity
ZLP_10_y105 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_0d3mm_Spectrum5_m0d676eV_2p5512eV.txt")
ndat=int(len(ZLP_10_y105))
# Energy loss values
ZLP_10_x105 = np.zeros(ndat)
Eloss_min = -0.676 # eV
Eloss_max = +2.5512 # eV
i=0
while(i<ndat):
    ZLP_10_x105[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
    
# Spectrum106, read the intensity
ZLP_10_y106 = np.loadtxt("Data/Apr/ZLP_200keV_10ms_0d3mm_Spectrum6_m0d676eV_2p5739eV.txt")
ndat=int(len(ZLP_10_y106))
# Energy loss values
ZLP_10_x106 = np.zeros(ndat)
Eloss_min = -0.676 # eV
Eloss_max = +2.5739 # eV
i=0
while(i<ndat):
    ZLP_10_x106[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1        

    # Spectrum201, read the intensity
ZLP_10_y201 = np.loadtxt("Data/Apr/ZLP_60keV_10ms_Spectrum1_m0d4eV_4p2282eV.txt")
ndat=int(len(ZLP_10_y201))
# Energy loss values
ZLP_10_x201 = np.zeros(ndat)
Eloss_min = -0.4 # eV
Eloss_max = +4.2282 # eV
i=0
while(i<ndat):
    ZLP_10_x201[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1  
    
    
# Spectrum202, read the intensity
ZLP_10_y202 = np.loadtxt("Data/Apr/ZLP_60keV_10ms_Spectrum2_m0d4eV_4p126eV.txt")
ndat=int(len(ZLP_10_y202))
# Energy loss values
ZLP_10_x202 = np.zeros(ndat)
Eloss_min = -0.4 # eV
Eloss_max = +4.126 # eV
i=0
while(i<ndat):
    ZLP_10_x202[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1      
      
# Spectrum203, read the intensity
ZLP_10_y203 = np.loadtxt("Data/Apr/ZLP_60keV_10ms_Spectrum3_m0d4eV_4p2811eV.txt")
ndat=int(len(ZLP_10_y203))
# Energy loss values
ZLP_10_x203 = np.zeros(ndat)
Eloss_min = -0.4 # eV
Eloss_max = +4.2811 # eV
i=0
while(i<ndat):
    ZLP_10_x203[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1 
    
    
# Spectrum204, read the intensity
ZLP_10_y204 = np.loadtxt("Data/Apr/ZLP_60keV_10ms_Spectrum4_m0d3994eV_4p2224eV.txt")
ndat=int(len(ZLP_10_y204))
# Energy loss values
ZLP_10_x204 = np.zeros(ndat)
Eloss_min = -0.3994 # eV
Eloss_max = +4.2224 # eV
i=0
while(i<ndat):
    ZLP_10_x204[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1  
    
# Spectrum205, read the intensity
ZLP_10_y205 = np.loadtxt("Data/Apr/ZLP_60keV_10ms_Spectrum5_m0d3994eV_4p3244eV.txt")
ndat=int(len(ZLP_10_y205))
# Energy loss values
ZLP_10_x205 = np.zeros(ndat)
Eloss_min = -0.3994 # eV
Eloss_max = +4.3244 # eV
i=0
while(i<ndat):
    ZLP_10_x205[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1      
                  
        
    # Spectrum206, read the intensity
ZLP_10_y206 = np.loadtxt("Data/Apr/ZLP_60keV_10ms_Spectrum6_m0d4326eV_4p7792eV.txt")
ndat=int(len(ZLP_10_y206))
# Energy loss values
ZLP_10_x206 = np.zeros(ndat)
Eloss_min = -0.4326 # eV
Eloss_max = +4.7792 # eV
i=0
while(i<ndat):
    ZLP_10_x206[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1  
    


ZLP_10_x1, ZLP_10_y1 = window(ZLP_10_x1, ZLP_10_y1, window_min, window_max)
ZLP_10_x2, ZLP_10_y2 = window(ZLP_10_x2, ZLP_10_y2, window_min, window_max)
ZLP_10_x3, ZLP_10_y3 = window(ZLP_10_x3, ZLP_10_y3, window_min, window_max)
ZLP_10_x4, ZLP_10_y4 = window(ZLP_10_x4, ZLP_10_y4, window_min, window_max)
ZLP_10_x5, ZLP_10_y5 = window(ZLP_10_x5, ZLP_10_y5, window_min, window_max)
ZLP_10_x6, ZLP_10_y6 = window(ZLP_10_x6, ZLP_10_y6, window_min, window_max)
ZLP_10_x7, ZLP_10_y7 = window(ZLP_10_x7, ZLP_10_y7, window_min, window_max)
ZLP_10_x8, ZLP_10_y8 = window(ZLP_10_x8, ZLP_10_y8, window_min, window_max)
ZLP_10_x9, ZLP_10_y9 = window(ZLP_10_x9, ZLP_10_y9, window_min, window_max)
ZLP_10_x10, ZLP_10_y10 = window(ZLP_10_x10, ZLP_10_y10, window_min, window_max)
ZLP_10_x11, ZLP_10_y11 = window(ZLP_10_x11, ZLP_10_y11, window_min, window_max)
ZLP_10_x12, ZLP_10_y12 = window(ZLP_10_x12, ZLP_10_y12, window_min, window_max)
ZLP_10_x13, ZLP_10_y13 = window(ZLP_10_x13, ZLP_10_y13, window_min, window_max)
ZLP_10_x14, ZLP_10_y14 = window(ZLP_10_x14, ZLP_10_y14, window_min, window_max)
ZLP_10_x15, ZLP_10_y15 = window(ZLP_10_x15, ZLP_10_y15, window_min, window_max)
ZLP_10_x16, ZLP_10_y16 = window(ZLP_10_x16, ZLP_10_y16, window_min, window_max)
ZLP_10_x17, ZLP_10_y17 = window(ZLP_10_x17, ZLP_10_y17, window_min, window_max)
ZLP_10_x18, ZLP_10_y18 = window(ZLP_10_x18, ZLP_10_y18, window_min, window_max)
ZLP_10_x19, ZLP_10_y19 = window(ZLP_10_x19, ZLP_10_y19, window_min, window_max)

## With aperture
    
ZLP_10_x201, ZLP_10_y201 = window(ZLP_10_x201, ZLP_10_y201, window_min, window_max)
ZLP_10_x202, ZLP_10_y202 = window(ZLP_10_x202, ZLP_10_y202, window_min, window_max)
ZLP_10_x203, ZLP_10_y203 = window(ZLP_10_x203, ZLP_10_y203, window_min, window_max)
ZLP_10_x204, ZLP_10_y204 = window(ZLP_10_x204, ZLP_10_y204, window_min, window_max)
ZLP_10_x205, ZLP_10_y205 = window(ZLP_10_x205, ZLP_10_y205, window_min, window_max)
ZLP_10_x206, ZLP_10_y206 = window(ZLP_10_x206, ZLP_10_y206, window_min, window_max)




    
#plt.plot(ZLP_10_x1, ZLP_10_y1 ,color="forestgreen",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#1")
#plt.plot(ZLP_10_x2, ZLP_10_y2 ,color="limegreen",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#2")
#plt.plot(ZLP_10_x3, ZLP_10_y3 ,color="blue",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#3")
#plt.plot(ZLP_10_x4, ZLP_10_y4 ,color="seagreen",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#4")
#plt.plot(ZLP_10_x5, ZLP_10_y5 ,color="turquoise",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#5")
#plt.plot(ZLP_10_x6, ZLP_10_y6 ,color="green",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#6")
#plt.plot(ZLP_10_x7, ZLP_10_y7,color="indianred",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#7")
plt.plot(ZLP_10_x8, ZLP_10_y8,color="maroon",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x9, ZLP_10_y9,color="red",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="200 keV")
plt.plot(ZLP_10_x10, ZLP_10_y10,color="chocolate",ls="dotted",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x11, ZLP_10_y11,color="gold",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x12, ZLP_10_y12,color="peru",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x13, ZLP_10_y13,color="orange",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x14, ZLP_10_y14,color="khaki",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x15, ZLP_10_y15 ,color="salmon",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x16, ZLP_10_y16 ,color="khaki",ls="dotted",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x17, ZLP_10_y17,color="darkorange",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x18, ZLP_10_y18,color="crimson",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x19, ZLP_10_y19,color="palevioletred",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)

plt.plot(ZLP_10_x201, ZLP_10_y201,color="forestgreen",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="60 keV")
plt.plot(ZLP_10_x202, ZLP_10_y202,color="limegreen",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x203, ZLP_10_y203,color="green",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x204, ZLP_10_y204,color="turquoise",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x205, ZLP_10_y205,color="seagreen",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x206, ZLP_10_y206,color="lightblue",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
#plt.plot(ZLP_10_x207, ZLP_10_y207,color="teal",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)



fig.set_size_inches(12, 5)
plt.xlabel(r"Energy loss (eV)",fontsize=13)
plt.ylabel(r"Intensity (a.u.)",fontsize=13)
#plt.ylim(0,1e5)
plt.grid(True)
plt.title('10ms')
plt.legend(fontsize=8, bbox_to_anchor=(1, 1))
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
#    
#ZLP_2_x1, ZLP_2_y1 = window(ZLP_2_x1, ZLP_2_y1, window_min, window_max)
#ZLP_2_x2, ZLP_2_y2 = window(ZLP_2_x2, ZLP_2_y2, window_min, window_max)
#ZLP_2_x3, ZLP_2_y3 = window(ZLP_2_x3, ZLP_2_y3, window_min, window_max)
#ZLP_2_x4, ZLP_2_y4 = window(ZLP_2_x4, ZLP_2_y4, window_min, window_max)
#ZLP_2_x5, ZLP_2_y5 = window(ZLP_2_x5, ZLP_2_y5, window_min, window_max)
##ZLP_2_x6, ZLP_2_y6 = window(ZLP_2_x6, ZLP_2_y6, window_min, window_max)
#ZLP_2_x7, ZLP_2_y7 = window(ZLP_2_x7, ZLP_2_y7, window_min, window_max)
#ZLP_2_x8, ZLP_2_y8 = window(ZLP_2_x8, ZLP_2_y8, window_min, window_max)
#ZLP_2_x9, ZLP_2_y9 = window(ZLP_2_x9, ZLP_2_y9, window_min, window_max)
#ZLP_2_x10, ZLP_2_y10 = window(ZLP_2_x10, ZLP_2_y10, window_min, window_max)
#ZLP_2_x11, ZLP_2_y11 = window(ZLP_2_x11, ZLP_2_y11, window_min, window_max)


#plt.plot(ZLP_2_x1, ZLP_2_y1 ,color="forestgreen",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#1")
#plt.plot(ZLP_2_x2, ZLP_2_y2 ,color="skyblue",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#2")
#plt.plot(ZLP_2_x3, ZLP_2_y3 ,color="limegreen",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#3")
#plt.plot(ZLP_2_x4, ZLP_2_y4 ,color="turquoise",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#4")
#plt.plot(ZLP_2_x5, ZLP_2_y5 ,color="lightblue",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#5")
#plt.plot(ZLP_2_x6, ZLP_2_y6 ,color="orange",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#6")
#plt.plot(ZLP_2_x7, ZLP_2_y7,color="gold",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#7")
#plt.plot(ZLP_2_x8, ZLP_2_y8,color="darkorange",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#8")
#plt.plot(ZLP_2_x9, ZLP_2_y9,color="red",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#9")
#plt.plot(ZLP_2_x10, ZLP_2_y10,color="crimson",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#10")
#plt.plot(ZLP_2_x11, ZLP_2_y11,color="palevioletred",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#11")
#fig.set_size_inches(12, 5)
#plt.xlabel(r"Energy loss (eV)",fontsize=13)
#plt.ylabel(r"Intensity (a.u.)",fontsize=13)
##plt.ylim(0,1e5)
##plt.grid(True)
#plt.title('200 keV, 2ms')
#plt.legend(fontsize=8, bbox_to_anchor=(1, 1))
#plt.savefig("EELSData-ZLP-3.pdf")
#plt.show()
#print("\n ************************ Data files have been prepared ***************************** \n")


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

def prepare_x_data(time, energy):
    
        if time == 10:
            if energy == 200:
                x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = ZLP_10_x8, ZLP_10_x9, ZLP_10_x10, ZLP_10_x11, ZLP_10_x12, ZLP_10_x13, ZLP_10_x14, ZLP_10_x15, ZLP_10_x16, ZLP_10_x17, ZLP_10_x18, ZLP_10_x19
                datafiles = 12
            if energy == 60:
                x1, x2, x3, x4, x5, x6= ZLP_10_x201, ZLP_10_x202, ZLP_10_x203, ZLP_10_x204, ZLP_10_x205, ZLP_10_x206
                datafiles = 6
        if time == 100:
            if energy == 200:   
                x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 = ZLP_100_x1, ZLP_100_x2, ZLP_100_x3, ZLP_100_x4, ZLP_100_x5, ZLP_100_x6, ZLP_100_x7, ZLP_100_x8, ZLP_100_x9, ZLP_100_x10, ZLP_100_x11, ZLP_100_x12, ZLP_100_x13, ZLP_100_x14, ZLP_100_x15
                datafiles = 15
            if energy == 60:
                x1, x2, x3, x4, x5, x6, x7 = ZLP_100_x201, ZLP_100_x202, ZLP_100_x203, ZLP_100_x204, ZLP_100_x205, ZLP_100_x206, ZLP_100_x207
                datafiles = 7
        if time == 2:
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = ZLP_2_x1, ZLP_2_x2, ZLP_2_x3, ZLP_2_x4, ZLP_2_x5, ZLP_2_x6, ZLP_2_x7, ZLP_2_x8, ZLP_2_x9, ZLP_2_x10, ZLP_2_x11 
            datafiles = 11
            
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
            if i == 15:
                mix_x[n] = x16
            if i == 16:
                mix_x[n] = x17  
            if i == 17:
                mix_x[n] = x18
            if i == 18:
                mix_x[n] = x19 

        mix_x = np.concatenate(mix_x)
    
        return mix_x
    
def prepare_y_data(time, energy):
        
        if time == 10:
            if energy == 200:
                y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12 = ZLP_10_y8, ZLP_10_y9, ZLP_10_y10, ZLP_10_y11, ZLP_10_y12, ZLP_10_y13, ZLP_10_y14, ZLP_10_y15, ZLP_10_y16, ZLP_10_y17, ZLP_10_y18, ZLP_10_y19
                datafiles = 12
                normalization = max(y1)
            if energy == 60:
                y1, y2, y3, y4, y5, y6 = ZLP_10_y201, ZLP_10_y202, ZLP_10_y203, ZLP_10_y204, ZLP_10_y205, ZLP_10_y206
                datafiles = 6
                normalization = max(y1)
        if time == 100:
            if energy == 200:   
                y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15 = ZLP_100_y1, ZLP_100_y2, ZLP_100_y3, ZLP_100_y4, ZLP_100_y5, ZLP_100_y6, ZLP_100_y7, ZLP_100_y8, ZLP_100_y9, ZLP_100_y10, ZLP_100_y11, ZLP_100_y12, ZLP_100_y13, ZLP_100_y14, ZLP_100_y15
                datafiles = 15
                normalization = max(y1)
            if energy == 60:
                y1, y2, y3, y4, y5, y6, y7 = ZLP_100_y201, ZLP_100_y202, ZLP_100_y203, ZLP_100_y204, ZLP_100_y205, ZLP_100_y206, ZLP_100_y207
                datafiles = 7
                normalization = max(y1)
        if time == 2:
            y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11 = ZLP_2_y1, ZLP_2_y2, ZLP_2_y3, ZLP_2_y4, ZLP_2_y5, ZLP_2_y6, ZLP_2_y7, ZLP_2_y8, ZLP_2_y9, ZLP_2_y10, ZLP_2_y11 
            datafiles = 11
            normalization = max(ZLP_10_y8)
        
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
            if i == 9: 
                mix_y[n] = np.divide(y10, normalization)
            if i == 10:
                mix_y[n] = np.divide(y11, normalization)
            if i == 11:
                mix_y[n] = np.divide(y12, normalization)
            if i == 12:
                mix_y[n] = np.divide(y13, normalization)
            if i == 13:
                mix_y[n] = np.divide(y14, normalization)
            if i == 14:
                mix_y[n] = np.divide(y15, normalization)    
            if i == 15:
                mix_y[n] = np.divide(y16, normalization)
            if i == 16:
                mix_y[n] = np.divide(y17, normalization)  
            if i == 17:
                mix_y[n] = np.divide(y18, normalization)
            if i == 18:
                mix_y[n] = np.divide(y19, normalization)
     
        
        mix_y = np.concatenate(mix_y)

        return mix_y
    
from sklearn.model_selection import train_test_split

def prepare_mix_data(time, energy):

    x_train = prepare_x_data(time, energy)
    y_train = prepare_y_data(time, energy)  
    
    df = np.stack((x_train, y_train)).T
    df = np.matrix(df)
    
    x_train = df[:,0]
    y_train = df[:,1]

    return x_train, y_train

