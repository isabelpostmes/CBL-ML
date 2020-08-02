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
ZLP_y4 = np.loadtxt("Data/WS2/Specimen_3/Position-4-Specimen_EELS-Spectrum_m0d93eV-to-9d07eV.txt")
ndat=int(len(ZLP_y4))
# Energy loss values
ZLP_x4 = np.zeros(ndat)
Eloss_min = -0.7811 # eV
Eloss_max = +18.2989 # eV
i=0
while(i<ndat):
    ZLP_x4[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat) 
    i = i + 1

ZLP_y5 = np.loadtxt("Data/WS2/Specimen_3/Position-5-Specimen_EELS-Spectrum_m0d93eV-to-9d07eV.txt")
ZLP_y6 = np.loadtxt("Data/WS2/Specimen_3/Position-6-Specimen_EELS-Spectrum_m0d93eV-to-9d07eV.txt")
ZLP_y7 = np.loadtxt("Data/WS2/Specimen_3/Position-7-Specimen_EELS-Spectrum_m0d93eV-to-9d07eV.txt")
ZLP_y8 = np.loadtxt("Data/WS2/Specimen_3/Position-8-Specimen_EELS-Spectrum_m0d93eV-to-9d07eV.txt")
ZLP_y9 = np.loadtxt("Data/WS2/Specimen_3/Position-9-Specimen_EELS-Spectrum_m0d93eV-to-9d07eV.txt")
ZLP_y10 = np.loadtxt("Data/WS2/Specimen_3/Position-10-Specimen_EELS-Spectrum_m0d93eV-to-9d07eV.txt")
ZLP_y11 = np.loadtxt("Data/WS2/Specimen_3/Position-11-Specimen_EELS-Spectrum_m0d93eV-to-9d07eV.txt")
ZLP_y12 = np.loadtxt("Data/WS2/Specimen_3/Position-12-Specimen_EELS-Spectrum_m0d93eV-to-9d07eV.txt")
ZLP_y13 = np.loadtxt("Data/WS2/Specimen_3/Position-13-Specimen_EELS-Spectrum_m0d93eV-to-9d07eV.txt")


ZLP_x5, ZLP_x6, ZLP_x7, ZLP_x8, ZLP_x9, ZLP_x10, ZLP_x11, ZLP_x12, ZLP_x13 = ZLP_x4,  ZLP_x4,  ZLP_x4,  ZLP_x4,  ZLP_x4,  ZLP_x4,  ZLP_x4,  ZLP_x4,  ZLP_x4,  


file4 = pd.DataFrame({"x":ZLP_x4, "y":ZLP_y4})
file5 = pd.DataFrame({"x":ZLP_x5, "y":ZLP_y5})
file6 = pd.DataFrame({"x":ZLP_x6, "y":ZLP_y6})
file7 = pd.DataFrame({"x":ZLP_x7, "y":ZLP_y7})
file8 = pd.DataFrame({"x":ZLP_x8, "y":ZLP_y8})
file9 = pd.DataFrame({"x":ZLP_x9, "y":ZLP_y9})
file10 = pd.DataFrame({"x":ZLP_x10, "y":ZLP_y10})
file11 = pd.DataFrame({"x":ZLP_x11, "y":ZLP_y11})
file12 = pd.DataFrame({"x":ZLP_x12, "y":ZLP_y12})
file13 = pd.DataFrame({"x":ZLP_x13, "y":ZLP_y13})


fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_size_inches((15,5)) 

from matplotlib import cm
cm_subsection = np.linspace(0,1,10) 
colors = [ cm.viridis(x) for x in cm_subsection ]

for i, file in enumerate([file4, file5, file6, file7, file8, file9, file10, file11, file12, file13]):
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

 


for i, file in enumerate([file4, file5, file6, file7, file8, file9, file10, file11, file12, file13]):
    zeropoint = file[file['y'] == file['y'].max()]['x']
    file['x_shifted'] = file['x'] - float(zeropoint)
    x = file['x_shifted']
    y = file['y']
    y_int = integrate.cumtrapz(y, x, initial=0)
    normalization = y_int[-1]
    file['y_norm'] = file['y'] / float(normalization)
    
    
    ax2.plot(file['x_shifted'] , np.log(file['y_norm']), '-', label=str(file), color=colors[i])
ax2.set_xlim([-.5, 4])
ax2.set_xlabel(r"Energy loss (eV)",fontsize=13)
ax2.set_ylabel(r"Intensity [-]",fontsize=13) 
ax2.set_title('Log sample data specimen 2')
fig.tight_layout()
plt.show()


x4, x5, x6, x7, x8, x9, x10, x11, x12, x13= ZLP_x4, ZLP_x5, ZLP_x6, ZLP_x7, ZLP_x8, ZLP_x9, ZLP_x10, ZLP_x11, ZLP_x12, ZLP_x13

y4, y5, y6, y7, y8, y9, y10, y11, y12, y13= ZLP_y4, ZLP_y5, ZLP_y6, ZLP_y7, ZLP_y8, ZLP_y9, ZLP_y10, ZLP_y11, ZLP_y12, ZLP_y13

####### VACUUM #########


# Spectrum1, read the intensity
Vac_y1 = np.loadtxt("Data/WS2/Specimen_3/Position-1-Vacuum_EELS-Spectrum_m0d93eV-to-9d07eV.txt")
ndat=int(len(Vac_y1))
# Energy loss values
Vac_x1 = np.zeros(ndat)
Eloss_min = -0.7811 # eV
Eloss_max = +18.2989 # eV
i=0
while(i<ndat):
    Vac_x1[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat) 
    i = i + 1
    
Vac_y2 = np.loadtxt("Data/WS2/Specimen_3/Position-2-Vacuum_EELS-Spectrum_m0d93eV-to-9d07eV.txt")
Vac_y3 = np.loadtxt("Data/WS2/Specimen_3/Position-3-Vacuum_EELS-Spectrum_m0d93eV-to-9d07eV.txt")




Vac_x1, Vac_x2, Vac_x3 = Vac_x1,Vac_x1,Vac_x1

file1 = pd.DataFrame({"x":Vac_x1, "y":Vac_y1})
file2 = pd.DataFrame({"x":Vac_x2, "y":Vac_y2})
file3 = pd.DataFrame({"x":Vac_x3, "y":Vac_y3})

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_size_inches((15,5))

cm_subsection = np.linspace(0,1,4) 
colors = [ cm.viridis(x) for x in cm_subsection ]
for i, file in enumerate([file1, file2, file3]):
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
ax1.set_title('Vacuum data specimen 2')

for i, file in enumerate([file1, file2, file3]):
    zeropoint = file[file['y'] == file['y'].max()]['x']
    file['x_shifted'] = file['x'] - float(zeropoint)
    
    x = file['x_shifted']
    y = file['y']
    y_int = integrate.cumtrapz(y, x, initial=0)
    normalization = y_int[-1]
    file['y_norm'] = file['y'] / float(normalization)
    
    
    ax2.plot(file['x_shifted'] , np.log(file['y_norm']), '-',  color=colors[i], label=i)
ax2.set_xlim([-.5, 4])
ax2.set_xlabel(r"Energy loss (eV)",fontsize=13)
ax2.set_ylabel(r"Intensity [-]",fontsize=13)   
ax2.set_title('Log vacuum data specimen 2')
fig.tight_layout()
plt.show()



