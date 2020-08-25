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
    
# Spectrum101, read the intensity
ZLP_100_y101 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_0d3mm_Spectrum1_m1eV_3p9951eV.txt")
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
ZLP_100_y102 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_0d3mm_Spectrum2_m1eV_3p9951eV.txt")
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
ZLP_100_y103 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_0d3mm_Spectrum3_m1eV_3p9709eV.txt")
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
ZLP_100_y104 = np.loadtxt("Data/Vacuum/ZLP_200keV_100ms_0d3mm_Spectrum4_m1eV_3p9113eV.txt")
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
ZLP_100_y201 = np.loadtxt("Data/Vacuum/ZLP_60keV_100ms_Spectrum1_m0d5362eV_5p4322eV.txt")
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
ZLP_100_y202 = np.loadtxt("Data/Vacuum/ZLP_60keV_100ms_Spectrum2_m0d536eV_5p303eV.txt")
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
ZLP_100_y203 = np.loadtxt("Data/Vacuum/ZLP_60keV_100ms_Spectrum3_m0d536eV_5p4625eV.txt")
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
ZLP_100_y204 = np.loadtxt("Data/Vacuum/ZLP_60keV_100ms_Spectrum4_m0d536eV_5p3977eV.txt")
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
ZLP_100_y205 = np.loadtxt("Data/Vacuum/ZLP_60keV_100ms_Spectrum5_m0d536eV_5p5966eV.txt")
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
ZLP_100_y206 = np.loadtxt("Data/Vacuum/ZLP_60keV_100ms_Spectrum6_m0d536eV_5p5288eV.txt")
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
ZLP_100_y207 = np.loadtxt("Data/Vacuum/ZLP_60keV_100ms_Spectrum7_m0d536eV_5p3977eV.txt")
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
    

file100_200 = pd.DataFrame()
file100_60 = pd.DataFrame()

for i, j in enumerate([ZLP_100_x1, ZLP_100_x2, ZLP_100_x3, ZLP_100_x4, ZLP_100_x5, ZLP_100_x6, ZLP_100_x7, ZLP_100_x8, ZLP_100_x9, ZLP_100_x10, ZLP_100_x11, ZLP_100_x12, ZLP_100_x13, ZLP_100_x14, ZLP_100_x15]):
    
    file100_200['x%(i)s'%{"i":i}] = j
for i, j in enumerate([ZLP_100_y1, ZLP_100_y2, ZLP_100_y3, ZLP_100_y4, ZLP_100_y5, ZLP_100_y6, ZLP_100_y7, ZLP_100_y8, ZLP_100_y9, ZLP_100_y10, ZLP_100_y11, ZLP_100_y12, ZLP_100_y13, ZLP_100_y14, ZLP_100_y15]):            
    file100_200['y%(i)s'%{"i":i}] = j
                
file100_200['time'] = 100
file100_200['energy'] = 200

for i, j in enumerate([ZLP_100_x201, ZLP_100_x202, ZLP_100_x203, ZLP_100_x204, ZLP_100_x205, ZLP_100_x206, ZLP_100_x207]):
    file100_60['x%(i)s'%{"i":i}] = j
for i, j in enumerate([ZLP_100_y201, ZLP_100_y202, ZLP_100_y203, ZLP_100_y204, ZLP_100_y205, ZLP_100_y206, ZLP_100_y207]):       
    file100_60['y%(i)s'%{"i":i}] = j
                
file100_60['time'] = 100
file100_60['energy'] = 60
                
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

plt.xlim([-.2, .5])

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

ZLP_10_y1 = np.loadtxt("Data/Vacuum/sp1-ZLP-200keV-10ms_m0d7478eV-to-5d1884[1].txt")
ndat=int(len(ZLP_10_y1))
# Energy loss values
ZLP_10_x1 = np.zeros(ndat)
Eloss_min = -0.7478# eV
Eloss_max = +5.1884 # eV
i=0
while(i<ndat):
    ZLP_10_x1[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum2, read the intensity
ZLP_10_y2 = np.loadtxt("Data/Vacuum/sp2-ZLP-200keV-10ms_m0d7109eV-to-4d8889[1].txt")
ndat=int(len(ZLP_10_y2))
# Energy loss values
ZLP_10_x2 = np.zeros(ndat)
Eloss_min = -0.7109 # eV
Eloss_max = +4.8889 # eV
i=0
while(i<ndat):
    ZLP_10_x2[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1



# Spectrum3, read the intensity
ZLP_10_y3 = np.loadtxt("Data/Vacuum/sp3-ZLP-200keV-10ms_m0d6763eV-to-4d59eV[1].txt")
ndat=int(len(ZLP_10_y3))
# Energy loss values
ZLP_10_x3 = np.zeros(ndat)
Eloss_min = -0.6763# eV
Eloss_max = +4.59 # eV
i=0
while(i<ndat):
    ZLP_10_x3[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum4, read the intensity
ZLP_10_y4 = np.loadtxt("Data/Vacuum/sp4-ZLP-200keV-10ms_m0d696eV-to-4d7448eV[1].txt")
ndat=int(len(ZLP_10_y4))
# Energy loss values
ZLP_10_x4 = np.zeros(ndat)
Eloss_min = -0.696 # eV
Eloss_max = +4.7448 # eV
i=0
while(i<ndat):
    ZLP_10_x4[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

# Spectrum5, read the intensity
ZLP_10_y5 = np.loadtxt("Data/Vacuum/sp5-ZLP-200keV-10ms_m0d6747eV-to-4d5591eV[1].txt")
ndat=int(len(ZLP_10_y5))
# Energy loss values
ZLP_10_x5 = np.zeros(ndat)
Eloss_min = -0.6747 # eV
Eloss_max = +4.5591# eV
i=0
while(i<ndat):
    ZLP_10_x5[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    

    
### Aperture files
    
# Spectrum101, read the intensity
ZLP_10_y101 = np.loadtxt("Data/Vacuum/ZLP_200keV_10ms_0d3mm_Spectrum1_m0d7456eV_2p8136eV.txt")
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
ZLP_10_y102 = np.loadtxt("Data/Vacuum/ZLP_200keV_10ms_0d3mm_Spectrum2_m0d7456eV_2p8744eV.txt")
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
ZLP_10_y103 = np.loadtxt("Data/Vacuum/ZLP_200keV_10ms_0d3mm_Spectrum3_m0d6756eV_2p5573eV.txt")
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
ZLP_10_y104 = np.loadtxt("Data/Vacuum/ZLP_200keV_10ms_0d3mm_Spectrum4_m0d676eV_2p5512eV.txt")
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
ZLP_10_y105 = np.loadtxt("Data/Vacuum/ZLP_200keV_10ms_0d3mm_Spectrum5_m0d676eV_2p5512eV.txt")
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
ZLP_10_y106 = np.loadtxt("Data/Vacuum/ZLP_200keV_10ms_0d3mm_Spectrum6_m0d676eV_2p5739eV.txt")
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
ZLP_10_y201 = np.loadtxt("Data/Vacuum/ZLP_60keV_10ms_Spectrum1_m0d4eV_4p2282eV.txt")
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
ZLP_10_y202 = np.loadtxt("Data/Vacuum/ZLP_60keV_10ms_Spectrum2_m0d4eV_4p126eV.txt")
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
ZLP_10_y203 = np.loadtxt("Data/Vacuum/ZLP_60keV_10ms_Spectrum3_m0d4eV_4p2811eV.txt")
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
ZLP_10_y204 = np.loadtxt("Data/Vacuum/ZLP_60keV_10ms_Spectrum4_m0d3994eV_4p2224eV.txt")
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
ZLP_10_y205 = np.loadtxt("Data/Vacuum/ZLP_60keV_10ms_Spectrum5_m0d3994eV_4p3244eV.txt")
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
ZLP_10_y206 = np.loadtxt("Data/Vacuum/ZLP_60keV_10ms_Spectrum6_m0d4326eV_4p7792eV.txt")
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


    
plt.xlim([-.1, .1])
bla=10
plt.plot(ZLP_10_x1, ZLP_10_y1/bla,color="gold",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x2, ZLP_10_y2/bla,color="peru",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x3, ZLP_10_y3/bla,color="orange",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x4, ZLP_10_y4/bla,color="khaki",ls="solid",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x5, ZLP_10_y5/bla ,color="salmon",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)

plt.plot(ZLP_10_x201, ZLP_10_y201,color="forestgreen",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="60 keV")
plt.plot(ZLP_10_x202, ZLP_10_y202,color="limegreen",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x203, ZLP_10_y203,color="green",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x204, ZLP_10_y204,color="turquoise",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x205, ZLP_10_y205,color="seagreen",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)
plt.plot(ZLP_10_x206, ZLP_10_y206,color="lightblue",ls="dashed",linewidth=2.0,marker="D",markersize=0.0)



file10_200 = pd.DataFrame()
file10_60 = pd.DataFrame()

for i, j in enumerate([ZLP_10_x1, ZLP_10_x2, ZLP_10_x3, ZLP_10_x4, ZLP_10_x5]):
    file10_200['x%(i)s'%{"i":i}] = j
    
for i, j in enumerate([ZLP_10_y1, ZLP_10_y2, ZLP_10_y3, ZLP_10_y4, ZLP_10_y5]):            
    file10_200['y%(i)s'%{"i":i}] = j
                
file10_200['time'] = 10
file10_200['energy'] = 200

for i, j in enumerate([ZLP_10_x201, ZLP_10_x202, ZLP_10_x203, ZLP_10_x204, ZLP_10_x205, ZLP_10_x206]):
    file10_60['x%(i)s'%{"i":i}] = j
for i, j in enumerate([ZLP_10_y201, ZLP_10_y202, ZLP_10_y203, ZLP_10_y204, ZLP_10_y205, ZLP_10_y206]):            
    file10_60['y%(i)s'%{"i":i}] = j
                
file10_60['time'] = 10
file10_60['energy'] = 60

fig.set_size_inches(12, 5)
plt.xlabel(r"Energy loss (eV)",fontsize=13)
plt.ylabel(r"Intensity (a.u.)",fontsize=13)
#plt.ylim(0,1e5)
plt.grid(True)
plt.title('10ms')
plt.legend(fontsize=8, bbox_to_anchor=(1, 1))
plt.savefig("EELSData-ZLP-2.pdf")
plt.show()




