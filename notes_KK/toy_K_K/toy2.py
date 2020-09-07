#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:55:00 2020

@author: isabel
"""
import matplotlib.pyplot as plt
import numpy as np
import math





# Model for ZLP
def IZLP(DeltaE):
    sigma = 0.3 # eV (variance)
    A = 1e10 # normalisation
    izlp = A*math.exp( -DeltaE**2/sigma**2 )
    return izlp

def IZLP_der(DeltaE):
    sigma = 0.3 # eV (variance)
    A = 1e10 # normalisation
    izlp = -2*DeltaE/sigma**2 * IZLP(DeltaE)
    return izlp



# Model for signal
def Isample(DeltaE):
    b = 0.5 # exponent
    Ebg = 1.5 # bandgap energy (eV)
    Bnorm = 0.1
    isample = 0
    if(DeltaE > Ebg):
        isample = Bnorm * ( DeltaE - Ebg)**b
    return isample

def Isample_der(DeltaE):
    b = 0.5 # exponent
    Ebg = 1.5 # bandgap energy (eV)
    Bnorm = 0.1
    isample = 0
    if(DeltaE > Ebg):
        isample = b/(DeltaE-Ebg) * Isample(DeltaE)
    return isample


# Min and max values of the EEL spectra
deltaE_min = -4 # eV, adjust for each spectrum
deltaE_max = 5 # eV, adjust for each spectrum
l=2000
deltaE = np.zeros(l)
EELtot = np.zeros(l)
EELZLP = np.zeros(l)
EELsample = np.zeros(l)
EELtot_der = np.zeros(l)
EELZLP_der = np.zeros(l)
EELsample_der = np.zeros(l)
for i in range(l):
    deltaE[i] = deltaE_min + (deltaE_max-deltaE_min)*(1.0*i)/(l-1.0)
    EELtot[i] = Isample(deltaE[i]) +  IZLP(deltaE[i])
    EELZLP[i] = IZLP(deltaE[i])
    EELsample[i] = Isample(deltaE[i])
    EELtot_der[i] = Isample_der(deltaE[i]) +  IZLP_der(deltaE[i])
    EELZLP_der[i] = IZLP_der(deltaE[i])
    EELsample_der[i] = Isample_der(deltaE[i])


plt.plot(deltaE,EELtot,linewidth=2.5,color="black",label="\rm total")
#plt.plot(deltaE,EELZLP,linewidth=2.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
#plt.plot(deltaE,EELsample,linewidth=2.5,color="red",ls="dashdot",label=r"${\rm sample}$")
plt.savefig("EELS_toy.pdf")
