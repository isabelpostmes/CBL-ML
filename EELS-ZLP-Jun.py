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
ZLP_y1 = np.loadtxt("Data/WS2/Specimen_1/Position-1_EELS-Spectrum_m2eV-to-18d48eV.txt")
ndat=int(len(ZLP_y1))
# Energy loss values
ZLP_x1 = np.zeros(ndat)
Eloss_min = -2 # eV
Eloss_max = +18.48 # eV
i=0
while(i<ndat):
    ZLP_x1[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat) 
    i = i + 1
    
    
# peak 2
ZLP_y2 = np.loadtxt("Data/WS2/Specimen_1/Position-2_EELS-Spectrum_m2eV-to-18d48eV.txt")
ndat=int(len(ZLP_y2))
# Energy loss values
ZLP_x2 = np.zeros(ndat)
Eloss_min = -2 # eV
Eloss_max = +18.48 # eV
i=0
while(i<ndat):
    ZLP_x2[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
    
    # ZLPuum peak
ZLP_y3 = np.loadtxt("Data/WS2/Specimen_1/Position-3_EELS-Spectrum_m2eV-to-18d48eV.txt")
ndat=int(len(ZLP_y3))
# Energy loss values
ZLP_x3 = np.zeros(ndat)
Eloss_min = -2 # eV
Eloss_max = +18.48 # eV
i=0
while(i<ndat):
    ZLP_x3[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
    
    
    # ZLPuum peak
ZLP_y4 = np.loadtxt("Data/WS2/Specimen_1/Position-4_EELS-Spectrum_m2eV-to-18d48eV.txt")
ndat=int(len(ZLP_y4))
# Energy loss values
ZLP_x4 = np.zeros(ndat)
Eloss_min = -2 # eV
Eloss_max = +18.48 # eV
i=0
while(i<ndat):
    ZLP_x4[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1     
    
    # ZLPuum peak
ZLP_y5 = np.loadtxt("Data/WS2/Specimen_1/Position-5_EELS-Spectrum_m2eV-to-18d48eV.txt")
ndat=int(len(ZLP_y5))
# Energy loss values
ZLP_x5 = np.zeros(ndat)
Eloss_min = -2 # eV
Eloss_max = +18.48 # eV
i=0
while(i<ndat):
    ZLP_x5[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    
    
    
# ZLPuum peak
ZLP_y6 = np.loadtxt("Data/WS2/Specimen_1/Position-6_EELS-Spectrum_m2eV-to-18d48eV.txt")
ndat=int(len(ZLP_y6))
# Energy loss values
ZLP_x6 = np.zeros(ndat)
Eloss_min = -2 # eV
Eloss_max = +18.48 # eV
i=0
while(i<ndat):
    ZLP_x6[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat)
    i = i + 1    

file1 = pd.DataFrame({"x":ZLP_x1, "y":ZLP_y1})
file2 = pd.DataFrame({"x":ZLP_x2, "y":ZLP_y2})
file3 = pd.DataFrame({"x":ZLP_x3, "y":ZLP_y3})
file4 = pd.DataFrame({"x":ZLP_x4, "y":ZLP_y4})
file5 = pd.DataFrame({"x":ZLP_x5, "y":ZLP_y5})
file6 = pd.DataFrame({"x":ZLP_x6, "y":ZLP_y6})

 
fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_size_inches((15,5))
from matplotlib import cm
cm_subsection = np.linspace(0,1,10) 
colors = [ cm.viridis(x) for x in cm_subsection ]

for i, file in enumerate([file1, file2, file3, file4, file5, file6]):
    zeropoint = file[file['y'] == file['y'].max()]['x']
    file['x_shifted'] = file['x'] - float(zeropoint)
    ### Integration
    
    windowfile = file[file['x_shifted']<14]
    x = windowfile['x_shifted']
    y = windowfile['y']
    y_int = integrate.cumtrapz(y, x, initial=0)
    normalization = y.max()

    windowfile['y_norm'] = windowfile['y'] / float(normalization)
    
    ax1.plot(windowfile['x_shifted'] , windowfile['y_norm'], '-', label=str(file), color=colors[i])
ax1.set_xlim([-.5, 1])
ax1.set_xlabel(r"Energy loss (eV)",fontsize=13)
ax1.set_ylabel(r"Intensity [-]",fontsize=13)
ax1.set_title('Sample data specimen 1')


for i, file in enumerate([file1, file2, file3, file4, file5, file6]):
    
    
    ax2.plot(windowfile['x_shifted'] , np.log(windowfile['y_norm']), '-', label=str(file), color=colors[i])
    
ax2.set_xlim([-.5, 1])
ax2.set_xlabel(r"Energy loss (eV)",fontsize=13)
ax2.set_ylabel(r"log intensity [-]",fontsize=13)   
ax2.set_title('Log sample data specimen 1')
fig.tight_layout()
plt.show()


x1, x2, x3, x4, x5, x6 = ZLP_x1, ZLP_x2, ZLP_x3, ZLP_x4, ZLP_x5, ZLP_x6

y1, y2, y3, y4, y5, y6 = ZLP_y1, ZLP_y2, ZLP_y3, ZLP_y4, ZLP_y5, ZLP_y6


####### VACUUM #########


# Spectrum1, read the intensity
Vac_y1 = np.loadtxt("Data/WS2/Specimen_1/Position-7-Vacuum_EELS-Spectrum_m2eV-to-18d48eV.txt")
ndat=int(len(Vac_y1))
# Energy loss values
Vac_x1 = np.zeros(ndat)
Eloss_min = -2 # eV
Eloss_max = +18.48 # eV
i=0
while(i<ndat):
    Vac_x1[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat) 
    i = i + 1
    
    

Vac_y7 = np.loadtxt("Data/WS2/Specimen_1/Position-7-Vacuum_EELS-Spectrum_m2eV-to-18d48eV.txt")
Vac_y8 = np.loadtxt("Data/WS2/Specimen_1/Position-8-Vacuum_EELS-Spectrum_m2eV-to-18d48eV.txt")
Vac_y9 = np.loadtxt("Data/WS2/Specimen_1/Position-9-Vacuum_EELS-Spectrum_m2eV-to-18d48eV.txt")
Vac_y10 = np.loadtxt("Data/WS2/Specimen_1/Position-10-Vacuum_EELS-Spectrum_m2eV-to-18d48eV.txt")
Vac_y11 = np.loadtxt("Data/WS2/Specimen_1/Position-11-Vacuum_EELS-Spectrum_m2eV-to-18d48eV.txt")
Vac_y12 = np.loadtxt("Data/WS2/Specimen_1/Position-12-Vacuum_EELS-Spectrum_m2eV-to-18d48eV.txt")
Vac_y13 = np.loadtxt("Data/WS2/Specimen_1/Position-13-Vacuum_EELS-Spectrum_m2eV-to-18d48eV.txt")
Vac_y14 = np.loadtxt("Data/WS2/Specimen_1/Position-14-Vacuum_EELS-Spectrum_m2eV-to-18d48eV.txt")
Vac_y15 = np.loadtxt("Data/WS2/Specimen_1/Position-15-Vacuum_EELS-Spectrum_m2eV-to-18d48eV.txt")
Vac_y16 = np.loadtxt("Data/WS2/Specimen_1/Position-16-Vacuum_EELS-Spectrum_m2eV-to-18d48eV.txt")
Vac_y17 = np.loadtxt("Data/WS2/Specimen_1/Position-17-Vacuum_EELS-Spectrum_m2eV-to-18d48eV.txt")
Vac_y18 = np.loadtxt("Data/WS2/Specimen_1/Position-18-Vacuum_EELS-Spectrum_m2eV-to-18d48eV.txt")

Vac_x7, Vac_x8, Vac_x9, Vac_x10, Vac_x11, Vac_x12, Vac_x13, Vac_x14, Vac_x15, Vac_x16, Vac_x17, Vac_x18 =\
Vac_x1, Vac_x1, Vac_x1, Vac_x1, Vac_x1, Vac_x1, Vac_x1, Vac_x1, Vac_x1, Vac_x1, Vac_x1, Vac_x1

file7 = pd.DataFrame({"x":Vac_x7, "y":Vac_y7})
file8 = pd.DataFrame({"x":Vac_x8, "y":Vac_y8})
file9 = pd.DataFrame({"x":Vac_x9, "y":Vac_y9})
file10 = pd.DataFrame({"x":Vac_x10, "y":Vac_y10})
file11 = pd.DataFrame({"x":Vac_x11, "y":Vac_y11})
file12 = pd.DataFrame({"x":Vac_x12, "y":Vac_y12})
file13 = pd.DataFrame({"x":Vac_x13, "y":Vac_y13})
file14 = pd.DataFrame({"x":Vac_x14, "y":Vac_y14})
file15 = pd.DataFrame({"x":Vac_x15, "y":Vac_y15})
file16 = pd.DataFrame({"x":Vac_x16, "y":Vac_y16})
file17 = pd.DataFrame({"x":Vac_x17, "y":Vac_y17})
file18 = pd.DataFrame({"x":Vac_x18, "y":Vac_y18})


fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_size_inches((15,5))


cm_subsection = np.linspace(0,1,15) 
colors = [ cm.viridis(x) for x in cm_subsection ]
for i, file in enumerate([file7, file8, file9, file10, file11, file12, file13, file14, file15, file16, file17]):
    zeropoint = file[file['y'] == file['y'].max()]['x']
    file['x_shifted'] = file['x'] - float(zeropoint)
    
    ### Integration
    x = file['x_shifted']
    y = file['y']
    y_int = integrate.cumtrapz(y, x, initial=0)
    normalization = y_int[-1]
    
    file['y_norm'] = file['y'] / float(normalization)
    
    
    ax1.plot(file['x_shifted'] , file['y_norm'], '-', label=str(file), color=colors[i])
ax1.set_xlim([-.5, 1])
ax1.set_xlabel(r"Energy loss (eV)",fontsize=13)
ax1.set_ylabel(r"Intensity [-]",fontsize=13) 
ax1.set_title('Vacuum ZLP specimen 1')

for i, file in enumerate([file7, file8, file9, file10, file11, file12, file13, file14, file15, file16, file17]):
    
    ax2.plot(file['x_shifted'] , np.log(file['y_norm']), '-',  color=colors[i], label=i)
ax2.set_xlim([-.5, 1])
ax2.set_xlabel(r"Energy loss (eV)",fontsize=13)
ax2.set_ylabel(r"Intensity [-]",fontsize=13)  
ax2.set_title('Vacuum log ZLP specimen 1')

fig.tight_layout()
plt.show()



