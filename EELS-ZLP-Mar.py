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

##################################################
# Read data from file

# Spectrum1, read the intensity

EELSData_intensity_zlp_1 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum1_m0d9567eV_8p6eV.txt")
ndat=int(len(EELSData_intensity_zlp_1))
# Energy loss values
EELSData_Eloss_1 = np.zeros(ndat)
Eloss_min = -0.9567 # eV
Eloss_max = +8.6 # eV

i=0
while(i<ndat):
    EELSData_Eloss_1[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum2, read the intensity
EELSData_intensity_zlp_2 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum2_m0d947eV_8p4681eV.txt")
ndat=int(len(EELSData_intensity_zlp_2))
# Energy loss values
EELSData_Eloss_2 = np.zeros(ndat)
Eloss_min = -0.947 # eV
Eloss_max = +8.468 # eV

i=0
while(i<ndat):
    EELSData_Eloss_2[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum3, read the intensity
EELSData_intensity_zlp_3 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum3_m0d94eV_8p3603eV.txt")
ndat=int(len(EELSData_intensity_zlp_3))
# Energy loss values
EELSData_Eloss_3 = np.zeros(ndat)
Eloss_min = -0.94 # eV
Eloss_max = +8.3603 # eV

i=0
while(i<ndat):
    EELSData_Eloss_3[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum4, read the intensity
EELSData_intensity_zlp_4 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum4_m0d961eV_8p7343eV.txt")
ndat=int(len(EELSData_intensity_zlp_4))
# Energy loss values
EELSData_Eloss_4 = np.zeros(ndat)
Eloss_min = -0.961 # eV
Eloss_max = +8.7343 # eV

i=0
while(i<ndat):
    EELSData_Eloss_4[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum5, read the intensity
EELSData_intensity_zlp_5 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum5_m0d951eV_8p5967eV.txt")
ndat=int(len(EELSData_intensity_zlp_5))
# Energy loss values
EELSData_Eloss_5 = np.zeros(ndat)
Eloss_min = -0.951 # eV
Eloss_max = +8.5967 # eV

i=0
while(i<ndat):
    EELSData_Eloss_5[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
# Spectrum6, read the intensity
EELSData_intensity_zlp_6 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum6_m0d852eV_7p4173eV.txt")
ndat=int(len(EELSData_intensity_zlp_6))
# Energy loss values
EELSData_Eloss_6 = np.zeros(ndat)
Eloss_min = -0.852 # eV
Eloss_max = +7.4173 # eV

i=0
while(i<ndat):
    EELSData_Eloss_6[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
# Spectrum7, read the intensity
EELSData_intensity_zlp_7 = np.loadtxt("Data/Mar/ZLP_200keV_100ms_Spectrum7_m0d852eV_7p4173eV.txt")
ndat=int(len(EELSData_intensity_zlp_7))
# Energy loss values
EELSData_Eloss_7 = np.zeros(ndat)
Eloss_min = -0.852 # eV
Eloss_max = +7.4173 # eV

i=0
while(i<ndat):
    EELSData_Eloss_7[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
    
import matplotlib.pyplot as plt

plt.plot(EELSData_Eloss_1, EELSData_intensity_zlp_1,color="blue",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #1")
plt.plot(EELSData_Eloss_2, EELSData_intensity_zlp_2,color="red",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #2")
plt.plot(EELSData_Eloss_3, EELSData_intensity_zlp_3,color="green",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #3")
plt.plot(EELSData_Eloss_4, EELSData_intensity_zlp_4,color="navy",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #4")
plt.plot(EELSData_Eloss_5, EELSData_intensity_zlp_5,color="black",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #5")
plt.plot(EELSData_Eloss_6, EELSData_intensity_zlp_6,color="red",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #6")
plt.plot(EELSData_Eloss_7, EELSData_intensity_zlp_7,color="orange",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #7")
plt.xlabel(r"Energy loss (eV)",fontsize=13)
plt.ylabel(r"Intensity (a.u.)",fontsize=13)
plt.ylim(0,8e5)
#plt.xlim(-0.15,0.15)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig("EELSData-ZLP-1.pdf")
plt.show()


################### SECOND SUBSET OF DATA #######################

ZLP1 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum1_m0p5492eV_3p8104eV.txt")
ZLP2 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum2_m0p5492eV_3p8104eV.txt")
ZLP3 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum3_m0p5492eV_3p8104eV.txt")
ZLP4 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum4_m0p5492eV_3p8104eV.txt")
ZLP5 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum5_m0p5492eV_3p8104eV.txt")
ZLP6 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum6_m0p5492eV_3p8104eV.txt")
ZLP7 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum7_m0p5492eV_3p8104eV.txt")
ZLP8 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum8_m0p5492eV_3p8104eV.txt")
ZLP9 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum9_m0p5492eV_3p8104eV.txt")
ZLP10 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum10_m0p5492eV_3p8104eV.txt")
ZLP11 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum11_m0p5492eV_3p8104eV.txt")
ZLP12 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum12_m0p5492eV_3p8104eV.txt")
ZLP13 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum13_m0p5492eV_3p8104eV.txt")
ZLP14 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum14_m0p5492eV_3p8104eV.txt")
ZLP15 = np.loadtxt("Data/Mar/Part2/ZLP_200keV_10ms_Spectrum15_m0p5492eV_3p8104eV.txt")

ndat = int(len(ZLP1))
xdata = np.zeros(ndat)
Eloss_min = -0.5492 # eV
Eloss_max = +3.8104 # eV

i=0
while(i<ndat):
    xdata[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1
plt.plot(xdata, ZLP1 ,color="green",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #1")
plt.plot(xdata, ZLP2 ,color="lightgreen",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #2")
plt.plot(xdata, ZLP3 ,color="blue",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #3")
plt.plot(xdata, ZLP4 ,color="navy",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #4")
plt.plot(xdata, ZLP5 ,color="black",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #5")
plt.plot(xdata, ZLP6 ,color="orange",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #6")
plt.plot(xdata, ZLP7 ,color="yellow",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #7")
plt.plot(xdata, ZLP8 ,color="darkgreen",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #8")
plt.plot(xdata, ZLP9 ,color="purple",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #9")
plt.plot(xdata, ZLP10 ,color="pink",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #10")
plt.plot(xdata, ZLP11 ,color="red",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #11")
plt.plot(xdata, ZLP12 ,color="brown",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #12")
plt.plot(xdata, ZLP13 ,color="grey",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #13")
plt.plot(xdata, ZLP14 ,color="orange",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #14")
plt.plot(xdata, ZLP15 ,color="lightblue",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #15")

plt.xlabel(r"Energy loss (eV)",fontsize=13)
plt.ylabel(r"Intensity (a.u.)",fontsize=13)
plt.ylim(0,1e5)
plt.xlim(-0.05,0.1)
plt.grid(True)
#plt.legend(fontsize=12)
plt.savefig("EELSData-ZLP-2.pdf")
            
print("\n ************************ Data files have been prepared ***************************** \n")


N_train = 1000
N_data = N_train
N_val = 200
N_test = N_val


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def prepare_single_data():
    '''Training set of many copies of one single spectrum'''
    y_train = EELSData_intensity_zlp_2
    x_train = EELSData_Eloss_2

    vector = np.ones(N_train)
    x_train =  np.swapaxes( (np.reshape(vector,[N_train,1, 1]) * x_train), 1, 2)
    
    y_train = np.swapaxes( (np.reshape(vector,[N_train,1, 1]) * y_train), 1, 2)
    ### Normalization
    y_max = np.max(y_train, axis=1)
    y_train_norm = np.divide(y_train, y_max[:, None, :])
    y_train_norm = np.squeeze(y_train_norm, axis=2)

    y_train = y_train_norm.flatten()
    x_train = x_train.flatten()
    
    return x_train, y_train


def prepare_semisingle_data():
    '''Training set of many copies of one single spectrum'''
    y_train = EELSData_intensity_zlp_1
    x_train = EELSData_Eloss_1

    vector = np.ones(N_train)
    x_train =  np.swapaxes( (np.reshape(vector,[N_train,1, 1]) * x_train), 1, 2)
    y_train = np.swapaxes( (np.reshape(vector,[N_train,1, 1]) * y_train), 1, 2)
    ### Normalization
    y_max = np.max(y_train, axis=1)
    y_train_norm = np.divide(y_train, y_max[:, None, :])
    y_train_norm = np.squeeze(y_train_norm, axis=2)
    y_train = y_train_norm

    y_train = y_train.flatten()
    x_train = x_train.flatten()
    
    '''Validation data is another spectrum'''
    x_val = EELSData_Eloss_3
    y_val = EELSData_intensity_zlp_3
    
    valvector = np.ones(N_val)
    x_val =  np.swapaxes( (np.reshape(valvector,[N_val,1, 1]) * x_val), 1, 2)
    y_val = np.swapaxes( (np.reshape(valvector,[N_val,1, 1]) * y_val), 1, 2)
    ### Normalization
    y_val_max = np.max(y_val, axis=1)
    y_val_norm = np.divide(y_val, y_val_max[:, None, :])
    y_val_norm = np.squeeze(y_val_norm, axis=2)
    y_val = y_val_norm

    y_val = y_val.flatten()
    x_val = x_val.flatten()

    x_train, y_train, x_val, y_val = np.around(x_train, 4), np.around(y_train, 4), np.around(x_val, 4), np.around(y_val, 4)
    return x_train, y_train, x_val, y_val


import random

seednumber = np.random.randint(100)
print("Seed number for this set is:", seednumber)

def sequence():
    x = [random.randint(0,6) for i in range(N_data)]
    return x

def prepare_x_data():
        
        xdata1, x2, x3, x4, x5, x6, x7 = EELSData_Eloss_1, EELSData_Eloss_2, EELSData_Eloss_3, EELSData_Eloss_4, EELSData_Eloss_5, EELSData_Eloss_6, EELSData_Eloss_7
        random.seed(seednumber)
        array1 = sequence()
        mix_x = array1
        
        for n, i in enumerate(array1):
            if i == 0:
                mix_x[n] = xdata1
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
    
def prepare_y_data():
        
        y1, y2, y3, y4, y5, y6, y7 = EELSData_intensity_zlp_1, EELSData_intensity_zlp_2, EELSData_intensity_zlp_3, EELSData_intensity_zlp_4, EELSData_intensity_zlp_5, EELSData_intensity_zlp_5, EELSData_intensity_zlp_5
        normalization = max(y1)
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

def prepare_mix_data():

    x_train = prepare_x_data()
    y_train = prepare_y_data()  
    
    df = np.stack((x_train, y_train)).T
    np.matrix(df)
    
    #splitting data randomly to train and test using the sklearn library
    df_train, df_test = train_test_split(df, test_size=0.2)
    x_train = df_train[:,0]
    y_train = df_train[:,1]
    x_val = df_test[:,0]
    y_val = df_test[:,1]
    
    return x_train, y_train, x_val, y_val


import matplotlib.pyplot as plt

if mode == 'Mixture':
    x_train, y_train, x_val, y_val = prepare_mix_data()
    
if mode == 'Single':
    x_train, y_train = prepare_single_data()
    
N_train_tot = len(x_train)

x_train.reshape(N_train_tot, 1)
y_train.reshape(N_train_tot, 1)

