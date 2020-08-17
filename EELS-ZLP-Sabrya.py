###############################################
import numpy as np
import random
from numpy import loadtxt
import math
import scipy
import sklearn
from scipy import optimize
from scipy.optimize import leastsq

import matplotlib.pyplot as plt
from scipy import integrate

import pandas as pd
fig = plt.gcf()
###############################################

    
# Spectrum1, read the intensity
ZLP_y14 = np.loadtxt("Data/Sabrya/14.txt")
ndat=int(len(ZLP_y14))
# Energy loss values
ZLP_x14 = np.zeros(ndat)
Eloss_min = -4.054 # eV
Eloss_max = +45.471 # eV
i=0
while(i<ndat):
    ZLP_x14[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat) 
    i = i + 1

ZLP_y15 = np.loadtxt("Data/Sabrya/15.txt")
ZLP_y16 = np.loadtxt("Data/Sabrya/16.txt")
ZLP_y19 = np.loadtxt("Data/Sabrya/19.txt")
ZLP_y20 = np.loadtxt("Data/Sabrya/20.txt")
ZLP_y21 = np.loadtxt("Data/Sabrya/21.txt")

ZLP_x15, ZLP_x16, ZLP_x19, ZLP_x20, ZLP_x21 = ZLP_x14,  ZLP_x14,  ZLP_x14,  ZLP_x14,  ZLP_x14

file14 = pd.DataFrame({"x":ZLP_x14, "y":ZLP_y14})
file15 = pd.DataFrame({"x":ZLP_x15, "y":ZLP_y15})
file16 = pd.DataFrame({"x":ZLP_x16, "y":ZLP_y16})
file19 = pd.DataFrame({"x":ZLP_x19, "y":ZLP_y19})
file20 = pd.DataFrame({"x":ZLP_x20, "y":ZLP_y20})
file21 = pd.DataFrame({"x":ZLP_x21, "y":ZLP_y21})

ZLP_y17 = np.loadtxt("Data/Sabrya/17.txt")
ZLP_y22 = np.loadtxt("Data/Sabrya/22.txt")
ZLP_y23 = np.loadtxt("Data/Sabrya/23.txt")

ZLP_x17, ZLP_x22, ZLP_x23 = ZLP_x14,  ZLP_x14,  ZLP_x14


file17 = pd.DataFrame({"x":ZLP_x17, "y":ZLP_y17})
file22 = pd.DataFrame({"x":ZLP_x22, "y":ZLP_y22})
file23 = pd.DataFrame({"x":ZLP_x23, "y":ZLP_y23})



fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_size_inches((15,5)) 

from matplotlib import cm
cm_subsection = np.linspace(0,1,10) 
colors = [ cm.viridis(x) for x in cm_subsection ]

for i, file in enumerate([file14, file15, file16, file17, file19, file20, file21, file22, file23]):
    zeropoint = file[file['y'] == file['y'].max()]['x']
    file['x_shifted'] = file['x'] - float(zeropoint)
    
    x = file['x_shifted']
    y = file['y']
    y_int = integrate.cumtrapz(y, x, initial=0)
    normalization = y_int[-1]
    file['y_norm'] = file['y'] / float(normalization)
    
    ax1.plot(file['x_shifted'] , file['y_norm'], '-', label=str(file), color=colors[i])
ax1.set_xlim([-.5, 4])
ax1.set_xlabel(r"Energy loss (eV)",fontsize=13)
ax1.set_ylabel(r"Intensity [-]",fontsize=13)   
ax1.set_title('Sample data specimen 2')

 




x14, x15, x16, x17,  x19, x20, x21, x22, x23= ZLP_x14, ZLP_x15, ZLP_x16, ZLP_x17, ZLP_x19, ZLP_x20, ZLP_x21, ZLP_x22, ZLP_x23

y14, y15, y16, y17, y19, y20, y21, y22, y23= ZLP_y14, ZLP_y15, ZLP_y16, ZLP_y17, ZLP_y19, ZLP_y20, ZLP_y21, ZLP_y22, ZLP_y23





