#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 23:39:39 2020

@author: juan, isabel
"""
import numpy as np
import math
import matplotlib.pyplot as plt
#import lhapdf
from  matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text',usetex=False)
from numpy import fft


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

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5,3.5))

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


plt.plot(deltaE,EELtot,linewidth=2.5,color="black",label=r"${\rm total}$")
plt.plot(deltaE,EELZLP,linewidth=2.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(deltaE,EELsample,linewidth=2.5,color="red",ls="dashdot",label=r"${\rm sample}$")


# Now produce the plot        
plt.xlabel(r"${\rm Energy~loss~(eV)}$",fontsize=17)
plt.ylabel(r"${\rm Intensity~(a.u.)}$",fontsize=17)
plt.xlim(1.45,2.0)
plt.ylim(0,0.09)
#plt.grid(True)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.yticks([])

plt.legend(frameon=True,loc=[0.46,0.67],prop={'size':12.5})

yline_xx=np.array([1.52,1.52])
yline_yy=np.array([-20,1e6])
plt.plot(yline_xx, yline_yy,color='grey',lw=1.9,ls="dotted")

yline_xx=np.array([1.687,1.687])
yline_yy=np.array([-20,1e6])
plt.plot(yline_xx, yline_yy,color='grey',lw=1.9,ls="dotted")

plt.text(1.525,0.0035,r"$\Delta E_I$",fontsize=12)
plt.text(1.635,0.0035,r"$\Delta E_{II}$",fontsize=12)
plt.text(1.475,0.077,r"$I$",fontsize=20)
plt.text(1.6,0.077,r"$II$",fontsize=20)
plt.text(1.9,0.077,r"$III$",fontsize=20)


# Create an inset
inset = fig.add_axes([0.55, 0.25, 0.38, 0.30])  # Fraction of figure size (3, 2.4)
inset.axes.get_xaxis().set_visible(True)
inset.axes.get_yaxis().set_visible(True)
#inset.set_ylabel(r"$d I/d\Delta E$",fontsize=12)
props = dict(boxstyle='round', facecolor='white', alpha=1,lw=0.5)
inset.text(1.60,-1.35,r"$d I/d\Delta E$",fontsize=12,bbox=props)
inset.set_xlim(1.45, 1.7)
inset.set_ylim(-2, 1)
inset.tick_params(which='both',direction='in',labelsize=8,right=True)
inset.tick_params(which='major',length=3)
inset.tick_params(which='minor',length=3)
#inset.set_yticks([])

print(EELtot_der)

inset.plot(deltaE,EELtot_der,linewidth=1.5,color="black",label=r"${\rm total}$")
inset.plot(deltaE,EELZLP_der,linewidth=1.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
inset.plot(deltaE,EELsample_der,linewidth=1.5,color="red",ls="dashdot",label=r"${\rm sample}$")
inset.grid(True)

#plt.tight_layout()
#plt.savefig("EELS_toy.pdf")

### ISABEL TAKES OVER
shape_J = 100




#step 1: modulate intensity

#TODO: change variables to correct values
beta = 30E-3
m_0 = 1
v = 0.5 #needs to be smaller than c
c = 1 #natural units?
gamma = (1-v**2/c**2)**-0.5



theta_E = deltaE/(gamma*m_0*v**2)
log_term = np.log(1+(beta/theta_E)**2)

EELsample_ac = EELsample/log_term


#step 2: extrapolation
r = 3 #Drude model, can also use estimation from exp. data
A = EELsample_ac[-1]
n_times_extra = 3
sem_inf = l*(n_times_extra+1)

ddeltaE = (deltaE[-1]-deltaE[0])/deltaE.size

EELsample_extrp = np.zeros(sem_inf)
deltaE_extrp = np.linspace(deltaE[0], sem_inf*ddeltaE+ddeltaE, sem_inf)

EELsample_extrp[:l] = EELsample_ac
deltaE_extrp[:l] = deltaE

EELsample_extrp[l:] = A*np.power(deltaE_extrp[l:],r)


#step 3: normalisation and retreiving Im[1/eps(E)]

Re_eps0 = 0 #value of Re[1/eps(0)]
int_EELsample_over_deltaE = np.sum(EELsample_extrp/deltaE_extrp)*ddeltaE
K = 2*int_EELsample_over_deltaE/(math.pi*(1-Re_eps0))


Im_eps = EELsample_extrp/K #Im[-1/eps(E)]


#step 4: retreiving Re[1/eps(E)]

method = 3  #1: integration at datapoints, except deltaE = deltaE'
            #2: integration between datapoints deltaE_i = (deltaE_i + deltaE_i+1)/2
            #3: FT

deltaE_extrp_Re = deltaE_extrp
Re_eps = np.zeros(sem_inf) 

#integrate around each energy (to avoid singularities)
if method == 1:
    deltaE_extrp_Re = deltaE_extrp
    Re_eps = np.zeros(sem_inf) 
    for i in range(deltaE_extrp.size):
        deltaE_i = deltaE_extrp[i]
        select = (deltaE_extrp != deltaE_i)
        Re_eps[i] = 1 - 2/math.pi * np.sum(Im_eps[select]*deltaE_extrp[select]/(np.power(deltaE_extrp[select],2)-deltaE_i**2))
elif method ==2:
    deltaE_extrp_Re = np.zeros(sem_inf-1) 
    Re_eps = np.zeros(sem_inf-1) 
    for i in range(deltaE_extrp.size-1):
        deltaE_i = (deltaE_extrp[i]+deltaE_extrp[i+1])/2
        deltaE_extrp_Re = deltaE_i
        Re_eps[i] = 1 - 2/math.pi * np.sum(Im_eps*deltaE_extrp/(np.power(deltaE_extrp,2)-deltaE_i**2))
else: 
    if method != 3:
        print("you have selected a wrong method, please select 1,2, or 3. FT method used.")
    FT_Im_eps = fft.ifft(-1j*Im_eps)
    plt.figure()
    plt.plot(deltaE_extrp[:2*l], np.real(FT_Im_eps[:2*l]), label = r"$F(Im)_1$")
    plt.plot(deltaE_extrp[:2*l], np.imag(FT_Im_eps[:2*l]), label = r"$F(Im)_2$")
    plt.plot(deltaE_extrp[:2*l], np.absolute(FT_Im_eps[:2*l]), label = r"$|F(Im)|$")
    
    
    sgn = np.ones(Im_eps.shape)
    half = math.floor(Im_eps.size/2)
    sgn[:half] *= -1
    FT_Re_eps = sgn * FT_Im_eps
    Re_eps = fft.fft(FT_Re_eps)



#step 6: retreiving ε
if method ==2:
    #Re_eps and Im_eps shifted with respect to eachother: what makes sense?
    eps1 = Re_eps / (Re_eps**2 + Im_eps[:-1]**2)
    eps2 = Im_eps[:-1] / (Re_eps**2 + Im_eps[:-1]**2)
    eps = eps1 + 1j*eps2
else: #method == 1 || method == 3:
    eps1 = Re_eps / (Re_eps**2 + Im_eps**2)
    eps2 = Im_eps / (Re_eps**2 + Im_eps**2)
    eps = eps1 + 1j*eps2



plt.figure()
plt.plot(deltaE_extrp[:2*l], eps1[:2*l], label = r"$\varepsilon_1$")
plt.plot(deltaE_extrp[:2*l], eps2[:2*l], label = r"$\varepsilon_2$")
plt.plot(deltaE_extrp[:2*l], np.absolute(eps[:2*l]), label = r"$|\varepsilon|$")
plt.title(r"dielectric function")
plt.legend()
plt.xlabel(r"$\Delta E$")
plt.ylabel(r"$\varepsilon$")
