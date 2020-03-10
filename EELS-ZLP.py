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

##################################################
# Read data from file
#

# Spectrum1, read the intensity
ndat=63
EELSData_intensity_zlp_1 = np.loadtxt("Data/Spectrum1.txt")

# Energy loss values
EELSData_Eloss_1 = np.zeros(ndat)
Eloss_min = -0.102 # eV
Eloss_max = +0.0988 # eV

i=0
while(i<ndat):
    EELSData_Eloss_1[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum2, read the intensity
ndat=59
EELSData_intensity_zlp_2 = np.loadtxt("Data/Spectrum2.txt")

# Energy loss values
EELSData_Eloss_2 = np.zeros(ndat)
Eloss_min = -0.102 # eV
Eloss_max = +0.0986 # eV

i=0
while(i<ndat):
    EELSData_Eloss_2[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum3, read the intensity
ndat=63
EELSData_intensity_zlp_3 = np.loadtxt("Data/Spectrum3.txt")

# Energy loss values
EELSData_Eloss_3 = np.zeros(ndat)
Eloss_min = -0.0984 # eV
Eloss_max = +0.1015 # eV

i=0
while(i<ndat):
    EELSData_Eloss_3[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum4, read the intensity
ndat=52
EELSData_intensity_zlp_4 = np.loadtxt("Data/Spectrum4.txt")

# Energy loss values
EELSData_Eloss_4 = np.zeros(ndat)
Eloss_min = -0.102 # eV
Eloss_max = +0.102 # eV

i=0
while(i<ndat):
    EELSData_Eloss_4[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1


# Spectrum5, read the intensity
ndat=40
EELSData_intensity_zlp_5 = np.loadtxt("Data/Spectrum5.txt")

# Energy loss values
EELSData_Eloss_5 = np.zeros(ndat)
Eloss_min = -0.0984 # eV
Eloss_max = +0.0984 # eV

i=0
while(i<ndat):
    EELSData_Eloss_5[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
            
print("\n ************************ Data files have been prepared ***************************** \n")


N_train = 10000
N_val = 2000
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
    x_val = x_train[N_train-N_test:]
    x_train = x_train[:(N_train-N_test)]
    
    y_train = np.swapaxes( (np.reshape(vector,[N_train,1, 1]) * y_train), 1, 2)
    ### Normalization
    y_max = np.max(y_train, axis=1)
    y_train_norm = np.divide(y_train, y_max[:, None, :])
    y_train_norm = np.squeeze(y_train_norm, axis=2)
    y_val = y_train_norm[N_train-N_test:]
    y_train = y_train_norm[:(N_train-N_test)]

    y_train = y_train.flatten()
    x_train = x_train.flatten()
    y_val = y_val.flatten()
    x_val = x_val.flatten()
    
    x_train, y_train, x_val, y_val = np.around(x_train, 4), np.around(y_train, 4), np.around(x_val, 4), np.around(y_val, 4)
    return x_train, y_train, x_val, y_val

x_train, y_train_norm, x_val, y_val = prepare_single_data()

def prepare_semisingle_data():
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

def sequence():
    N_data = 10000
    x = [random.randint(0,4) for i in range(N_data)]
    return x

def prepare_x_data(maxim):
        N_data = 10000
        xdata1, x2, x3, x4, x5 = EELSData_Eloss_1, EELSData_Eloss_2, EELSData_Eloss_3, EELSData_Eloss_4, EELSData_Eloss_5
        random.seed(9001)
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

        mix_x = np.concatenate(mix_x)
        x_train, x_val = mix_x[:maxim], mix_x[maxim:]
        return x_train, x_val
    
def prepare_y_data(maxim):
        y1, y2, y3, y4, y5 = EELSData_intensity_zlp_1, EELSData_intensity_zlp_2, EELSData_intensity_zlp_3, EELSData_intensity_zlp_4, EELSData_intensity_zlp_5
        y1 = np.divide(y1, max(y1))
        y2 = np.divide(y2, max(y2))
        y3 = np.divide(y3, max(y3))
        y4 = np.divide(y4, max(y4))
        y5 = np.divide(y5, max(y5))
        
        random.seed(9001)
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
        
        mix_y = np.concatenate(mix_y)
        y_train, y_val = mix_y[:maxim], mix_y[maxim:]
        
        
        return y_train, y_val
    
def prepare_mix_data():
    N_train = 10000
    N_data = 10000
    maxim = N_train*40
    N_val = 2000

    x_train, x_val = prepare_x_data(400000)
    y_train, y_val = prepare_y_data(400000)    
    
    return x_train, y_train, x_val, y_val

print('\n ****************** Training and validation sets have been prepared **************** \n')
print(' prepare_single_data \n prepare_semisingle_data \n prepare_mixed_data')