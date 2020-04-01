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
    

#""""""""""""#######"""""DATA PART 1""""""########""""""""""""

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

ZLP_100_x1, ZLP_100_y1 = window(ZLP_100_x1, ZLP_100_y1, -0.05, 0.05)
ZLP_100_x2, ZLP_100_y2 = window(ZLP_100_x2, ZLP_100_y2, -0.05, 0.05)
ZLP_100_x3, ZLP_100_y3 = window(ZLP_100_x3, ZLP_100_y3, -0.05, 0.05)
ZLP_100_x4, ZLP_100_y4 = window(ZLP_100_x4, ZLP_100_y4, -0.05, 0.05)
ZLP_100_x5, ZLP_100_y5 = window(ZLP_100_x5, ZLP_100_y5, -0.05, 0.05)
ZLP_100_x6, ZLP_100_y6 = window(ZLP_100_x6, ZLP_100_y6, -0.05, 0.05)
ZLP_100_x7, ZLP_100_y7 = window(ZLP_100_x7, ZLP_100_y7, -0.05, 0.05)

import matplotlib.pyplot as plt

plt.plot(ZLP_100_x1, ZLP_100_y1,color="blue",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#1")
plt.plot(ZLP_100_x2, ZLP_100_y2,color="red",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#2")
plt.plot(ZLP_100_x3, ZLP_100_y3,color="green",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#3")
plt.plot(ZLP_100_x4, ZLP_100_y4,color="navy",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#4")
plt.plot(ZLP_100_x5, ZLP_100_y5,color="black",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#5")
plt.plot(ZLP_100_x6, ZLP_100_y6,color="red",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#6")
plt.plot(ZLP_100_x7, ZLP_100_y7,color="orange",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#7")
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

#""""""""""""#######"""""DATA PART 1""""""########""""""""""""

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
Eloss_max = +3.0336 # eV
i=0
while(i<ndat):
    ZLP_10_x6[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1

    

ZLP_10_x1, ZLP_10_y1 = window(ZLP_10_x1, ZLP_10_y1, -0.05, 0.05)
ZLP_10_x2, ZLP_10_y2 = window(ZLP_10_x2, ZLP_10_y2, -0.05, 0.05)
ZLP_10_x3, ZLP_10_y3 = window(ZLP_10_x3, ZLP_10_y3, -0.05, 0.05)
ZLP_10_x4, ZLP_10_y4 = window(ZLP_10_x4, ZLP_10_y4, -0.05, 0.05)
ZLP_10_x5, ZLP_10_y5 = window(ZLP_10_x5, ZLP_10_y5, -0.05, 0.05)
ZLP_10_x6, ZLP_10_y6 = window(ZLP_10_x6, ZLP_10_y6, -0.05, 0.05)


    
plt.plot(ZLP_10_x1, ZLP_10_y1 ,color="green",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#1")
plt.plot(ZLP_10_x2, ZLP_10_y2 ,color="red",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#2")
plt.plot(ZLP_10_x3, ZLP_10_y3 ,color="blue",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#3")
plt.plot(ZLP_10_x4, ZLP_10_y4 ,color="pink",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="#4")
plt.plot(ZLP_10_x5, ZLP_10_y5 ,color="lightblue",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="#5")
plt.plot(ZLP_10_x6, ZLP_10_y6 ,color="orange",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="#6")

plt.xlabel(r"Energy loss (eV)",fontsize=13)
plt.ylabel(r"Intensity (a.u.)",fontsize=13)
#plt.ylim(0,1e5)
plt.grid(True)
plt.title('200 keV, 10ms')
plt.legend()
plt.savefig("EELSData-ZLP-2.pdf")
plt.show()
            
print("\n ************************ Data files have been prepared ***************************** \n")


N_train = 100
N_data = N_train
N_val = 20


N_test = N_val


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

import random

seednumber = np.random.randint(100)

print("Seed number for this set is:", seednumber)

def sequence():
    x = [random.randint(0,6) for i in range(N_data)]
    return x

def prepare_x_data(time):
        if time == 10:
            x1, x2, x3, x4, x5, x6, x7 = ZLP_10_x1, ZLP_10_x2, ZLP_10_x4, ZLP_10_x5, ZLP_10_x6, ZLP_10_x1, ZLP_10_x2
        if time == 100:
            x1, x2, x3, x4, x5, x6, x7 = ZLP_100_x1, ZLP_100_x2, ZLP_100_x3, ZLP_100_x4, ZLP_100_x5, ZLP_100_x6, ZLP_100_x7
            
        random.seed(seednumber)
        array1 = sequence()
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

        mix_x = np.concatenate(mix_x)
    
        return mix_x
    
def prepare_y_data(time):
        
        if time == 10:
            y1, y2, y3, y4, y5, y6, y7 = ZLP_10_y1, ZLP_10_y2, ZLP_10_y4, ZLP_10_y5, ZLP_10_y6, ZLP_10_y1, ZLP_10_y2
            normalization = max(y1)
        if time == 100:
            y1, y2, y3, y4, y5, y6, y7 = ZLP_100_y1, ZLP_100_y2, ZLP_100_y3, ZLP_100_y4, ZLP_100_y5, ZLP_100_y6, ZLP_100_y7
            normalization = max(ZLP_10_y1)

        y1 = np.divide(y1, normalization)
        y2 = np.divide(y2, normalization)
        y3 = np.divide(y3, normalization)
        y4 = np.divide(y4, normalization)
        y5 = np.divide(y5, normalization)
        y6 = np.divide(y6, normalization)
        y7 = np.divide(y7, normalization)
        
        
        random.seed(seednumber)
        array2 = sequence()
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
        
        mix_y = np.concatenate(mix_y)

        return mix_y
    
from sklearn.model_selection import train_test_split

def prepare_mix_data(time):

    x_train = prepare_x_data(time)
    y_train = prepare_y_data(time)  
    
    df = np.stack((x_train, y_train)).T
    df = np.matrix(df)
    
    #splitting data randomly to train and test using the sklearn library
    df_train, df_test = train_test_split(df, test_size=0.2)
    x_train = df_train[:,0]
    y_train = df_train[:,1]
    x_val = df_test[:,0]
    y_val = df_test[:,1]
    
    x_train = x_train * 20
    x_val = x_val * 20
    return x_train, y_train, x_val, y_val


import matplotlib.pyplot as plt

if mode == 'Mixture':
    x_train, y_train, x_val, y_val = prepare_mix_data(100)
    
    
N_train_tot = len(x_train)

x_train.reshape(N_train_tot, 1)
y_train.reshape(N_train_tot, 1)


plt.plot(x_train, y_train, 'o')
plt.title('Training df_train')
plt.show()