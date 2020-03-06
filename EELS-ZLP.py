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
EELSData_intensity_zlp_1 = np.loadtxt("data/Spectrum1.txt")

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
EELSData_intensity_zlp_2 = np.loadtxt("data/Spectrum2.txt")

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
EELSData_intensity_zlp_3 = np.loadtxt("data/Spectrum3.txt")

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
EELSData_intensity_zlp_4 = np.loadtxt("data/Spectrum4.txt")

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
EELSData_intensity_zlp_5 = np.loadtxt("data/Spectrum5.txt")

# Energy loss values
EELSData_Eloss_5 = np.zeros(ndat)
Eloss_min = -0.0984 # eV
Eloss_max = +0.0984 # eV

i=0
while(i<ndat):
    EELSData_Eloss_5[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
            

import matplotlib.pyplot as plt
plt.plot(EELSData_Eloss_1, EELSData_intensity_zlp_1,color="blue",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #1")
plt.plot(EELSData_Eloss_2, EELSData_intensity_zlp_2,color="red",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #2")
plt.plot(EELSData_Eloss_3, EELSData_intensity_zlp_3,color="green",ls="solid",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #3")
plt.plot(EELSData_Eloss_4, EELSData_intensity_zlp_4,color="navy",ls="dotted",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #4")
plt.plot(EELSData_Eloss_5, EELSData_intensity_zlp_5,color="black",ls="dashed",linewidth=2.0,marker="D",markersize=0.0,label="ZLP spectrum #5")
plt.xlabel(r"Energy loss (eV)",fontsize=13)
plt.ylabel(r"Intensity (a.u.)",fontsize=13)
plt.ylim(0,2e4)
plt.xlim(-0.103,0.103)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig("EELSData-ZLP.pdf")


print("\n ***************************************************** \n")
