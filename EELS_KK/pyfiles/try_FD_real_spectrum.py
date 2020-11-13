#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 18:31:28 2020
try with real spectrum
@author: isabel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
#import start_cleaning_lau
from functions_revised import *

def exp_decay(x, r):
    """ y_offset * np.power(x - x[0], -r) """
    return y_offset * np.power(1 + x - x[0], -r)

def CFT(x, y):
    x_0 = np.min(x)
    N_0 = np.argmin(np.absolute(x))
    N = len(x)
    x_max = np.max(x)
    delta_x = (x_max-x_0)/N
    k = np.linspace(0, N-1, N)
    cont_factor = np.exp(2j*np.pi*N_0*k/N)*delta_x #np.exp(-1j*(x_0)*k*delta_omg)*delta_x
    F_k = cont_factor * np.fft.fft(y)
    return F_k

def iCFT(x, Y_k):
    x_0 = np.min(x)
    N_0 = np.argmin(np.absolute(x))
    x_max = np.max(x)
    N = len(x)
    delta_x = (x_max-x_0)/N
    k = np.linspace(0, N-1, N)
    cont_factor = np.exp(-2j*np.pi*N_0*k/N)
    f_n = np.fft.ifft(cont_factor*Y_k)/delta_x # 2*np.pi ##np.exp(-2j*np.pi*x_0*k)
    return f_n



x = df_sample.iloc[2].x_shifted
y = df_sample.iloc[2].y_norm

x_lim = 1.3

x_exp = x[x<2][ x[x<2]> x_lim]
y_exp = y[x<2][ x[x<2]> x_lim]

y_offset = y_exp[0]

y_fit_exp, cov = scipy.optimize.curve_fit(exp_decay, x_exp, y_exp)


plt.figure()
plt.plot(x,y)
plt.plot(x_exp, exp_decay(x_exp, y_fit_exp[0]))
plt.xlim(0,5)
plt.ylim(0,0.02)


y_ZLP = np.concatenate((y[x<= x_lim], exp_decay(x[x>x_lim], y_fit_exp[0])))



r = 3 #Drude model, can also use estimation from exp. data
A = y[-1]
n_times_extra = 2
l = len(x)
sem_inf = l*(n_times_extra+1)

ddeltaE = (x[-1]-x[0])/l

y_extrp = np.zeros(sem_inf)
y_ZLP_extrp = np.zeros(sem_inf)
x_extrp = np.linspace(x[0]- l*ddeltaE, sem_inf*ddeltaE+x[0]- l*ddeltaE, sem_inf)

y_ZLP_extrp[l:l*2] = y_ZLP

y_extrp[l:l*2] = y
x_extrp[l:l*2] = x

y_extrp[2*l:] = A*np.power(1+x_extrp[2*l:]-x_extrp[2*l],-r)

x = x_extrp
y = y_extrp
y_ZLP = y_ZLP_extrp
y_EEL = y - y_ZLP



plt.figure()
plt.plot(x,y)
plt.plot(x, y_ZLP)
plt.plot(x, y_EEL)
plt.xlim(0,5)
plt.ylim(0,0.02)


plt.figure()
#tot = 1200
tot = 5*l
plt.plot(x[:tot],y[:tot])
plt.plot(x[:tot], y_EEL[:tot])
#plt.xlim(-10,100) 
plt.ylim(0,0.02)

try_only_ZLP = False
if try_only_ZLP:
    y = y_ZLP

z_nu = CFT(x,y_ZLP)
i_nu = CFT(x,y)
abs_i_nu = np.absolute(i_nu)
max_i_nu = np.max(abs_i_nu)
i_nu_copy = np.copy(i_nu)
#i_nu[abs_i_nu<max_i_nu*0.00000000000001] = 0
N_ZLP = 1#scipy.integrate.cumtrapz(y_ZLP, x, initial=0)[-1]#1 #arbitrary units??? np.sum(EELZLP)

s_nu = N_ZLP*np.log(i_nu/z_nu)
j1_nu = z_nu*s_nu/N_ZLP
#s_nu_2 = s_nu
#s_nu_2[np.isnan(s_nu)] = 0#1E10 #disregard NaN values, but setting them to 0 doesnt seem fair, as they should be inf

plt.figure()
plt.title("s_nu and j1_nu")
plt.plot(s_nu)
plt.plot(j1_nu)
#s_nu[150:1850] = 0
S_E = np.real(iCFT(x,s_nu))
s_nu_nc = s_nu
s_nu_nc[500:-500] = 0
S_E_nc = np.real(iCFT( x,s_nu_nc))
J1_E = np.real(iCFT(x,j1_nu))


plt.figure()
plt.title("z_nu")
plt.plot(z_nu)
plt.figure()
plt.title("i_nu")
plt.plot(i_nu)

plt.figure()
plt.title("i_nu/z_nu")
plt.plot(i_nu/z_nu)
plt.figure()
plt.title("s_nu")
plt.plot(s_nu)

plt.figure()
plt.title("zoomed spectrum and calculated values")
plt.plot(x,y,linewidth=2.5,color="black",label=r"${\rm total}$")
plt.plot(x,y_ZLP,linewidth=2.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(x,y_EEL,linewidth=2.5,color="red",ls="dashdot",label=r"${\rm sample}$")
#plt.plot(x,S_E[:len(x)],linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S(E)}$")
plt.plot(x,S_E_nc[:len(x)],linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S_{nc}(E)}$")
plt.plot(x,J1_E,linewidth=2.5,color="green",ls="dashdot",label=r"${\rm J^1(E)}$")

plt.legend()
# Now produce the plot        
plt.xlabel(r"${\rm Energy~loss~(eV)}$",fontsize=17)
plt.ylabel(r"${\rm Intensity~(a.u.)}$",fontsize=17)
plt.xlim(1.45,15.0)
plt.ylim(0,0.01)

#l = 160
plt.figure()
plt.title("total spectrum and calculated values")
#plt.plot(x,S_E,linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S(E)}$")
plt.plot(x,S_E_nc[:len(x)],linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S_{nc}(E)}$")
plt.plot(x,y_ZLP,linewidth=2.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(x,y_EEL,linewidth=2.5,color="red",label=r"${\rm sample}$")
plt.plot(x,J1_E,linewidth=2.5,color="green",ls="dotted",label=r"${\rm J^1(E)}$")

plt.ylim(0,0.3)


plt.figure()
plt.title("zoomed spectrum and calculated values")
#plt.plot(x,S_E,linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S(E)}$")
plt.plot(x,y_ZLP,linewidth=1,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(x,y_EEL,linewidth=0.5,color="red",label=r"${\rm sample}$")
plt.plot(x,J1_E,linewidth=0.5,color="green",label=r"${\rm J^1(E)}$")
plt.plot(x,S_E_nc[:len(x)],linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S_{nc}(E)}$")
plt.xlim(0,55)
plt.ylim(-0.001,0.01)
plt.legend()




plt.figure()
plt.xlim(-5, 70)
#plt.xlim(40,60)
#plt.ylim(-0.0005,0.0015)
plt.title("sample spectrum and J1")
plt.plot(x,y_EEL,linewidth=0.5,color="green",label=r"${\rm sample}$")
plt.plot(x,J1_E,linewidth=0.5,color="blue", label=r"${\rm J1(E)}$")
plt.plot(x,S_E_nc[:len(x)],linewidth=1.5,color="pink",label=r"${\rm S_{nc}(E)}$")
plt.legend()


#%%smoothed signal

x = df_sample.iloc[2].x_shifted
y = df_sample.iloc[2].y_smooth

x_lim = 1.3

x_exp = x[x<2][ x[x<2]> x_lim]
y_exp = y[x<2][ x[x<2]> x_lim]

y_offset = y_exp[0]

y_fit_exp, cov = scipy.optimize.curve_fit(exp_decay, x_exp, y_exp)


plt.figure()
plt.plot(x,y)
plt.plot(x_exp, exp_decay(x_exp, y_fit_exp[0]))
plt.xlim(0,5)
plt.ylim(0,0.02)
  

y_ZLP = np.concatenate((y[x<= x_lim], exp_decay(x[x>x_lim], y_fit_exp[0])))



r = 3 #Drude model, can also use estimation from exp. data
A = y[-1]
n_times_extra = 2
l = len(x)
sem_inf = l*(n_times_extra+1)

ddeltaE = (x[-1]-x[0])/l

y_extrp = np.zeros(sem_inf)
y_ZLP_extrp = np.zeros(sem_inf)
x_extrp = np.linspace(x[0]- l*ddeltaE, sem_inf*ddeltaE+x[0]- l*ddeltaE, sem_inf)

y_ZLP_extrp[l:l*2] = y_ZLP

y_extrp[l:l*2] = y
x_extrp[l:l*2] = x

y_extrp[2*l:] = A*np.power(1+x_extrp[2*l:]-x_extrp[2*l],-r)

x = x_extrp
y = y_extrp
y_ZLP = y_ZLP_extrp
y_EEL = y - y_ZLP



plt.figure()
plt.plot(x,y)
plt.plot(x, y_ZLP)
plt.plot(x, y_EEL)
plt.xlim(0,5)
plt.ylim(0,0.02)


plt.figure()
#tot = 1200
tot = 5*l
plt.plot(x[:tot],y[:tot])
plt.plot(x[:tot], y_EEL[:tot])
#plt.xlim(-10,100) 
plt.ylim(0,0.02)

try_only_ZLP = False
if try_only_ZLP:
    y = y_ZLP

z_nu = CFT(x,y_ZLP)
i_nu = CFT(x,y)
abs_i_nu = np.absolute(i_nu)
max_i_nu = np.max(abs_i_nu)
i_nu_copy = np.copy(i_nu)
#i_nu[abs_i_nu<max_i_nu*0.00000000000001] = 0
N_ZLP = 1#scipy.integrate.cumtrapz(y_ZLP, x, initial=0)[-1]#1 #arbitrary units??? np.sum(EELZLP)

s_nu = N_ZLP*np.log(i_nu/z_nu)
j1_nu = z_nu*s_nu/N_ZLP
#s_nu_2 = s_nu
#s_nu_2[np.isnan(s_nu)] = 0#1E10 #disregard NaN values, but setting them to 0 doesnt seem fair, as they should be inf
"""
plt.figure()
plt.title("s_nu and j1_nu")
plt.plot(s_nu)
plt.plot(j1_nu)
S_E = np.real(iCFT(x,s_nu))
s_nu_nc = s_nu
s_nu_nc[500:-500] = 0
S_E_nc = np.real(iCFT( x,s_nu_nc))
J1_E = np.real(iCFT(x,j1_nu))


plt.figure()
plt.title("z_nu")
plt.plot(z_nu)
plt.figure()
plt.title("i_nu")
plt.plot(i_nu)

plt.figure()
plt.title("i_nu/z_nu")
plt.plot(i_nu/z_nu)
plt.figure()
plt.title("s_nu")
plt.plot(s_nu)

plt.figure()
plt.title("zoomed spectrum and calculated values")
plt.plot(x,y,linewidth=2.5,color="black",label=r"${\rm total}$")
plt.plot(x,y_ZLP,linewidth=2.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(x,y_EEL,linewidth=2.5,color="red",ls="dashdot",label=r"${\rm sample}$")
plt.plot(x,S_E[:len(x)],linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S(E)}$")
plt.plot(x,S_E_nc[:len(x)],linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S_{nc}(E)}$")
plt.plot(x,J1_E,linewidth=2.5,color="green",ls="dashdot",label=r"${\rm J^1(E)}$")

plt.legend()
# Now produce the plot        
plt.xlabel(r"${\rm Energy~loss~(eV)}$",fontsize=17)
plt.ylabel(r"${\rm Intensity~(a.u.)}$",fontsize=17)
plt.xlim(1.45,15.0)
plt.ylim(0,0.01)

#l = 160
plt.figure()
plt.title("total spectrum and calculated values")
plt.plot(x,S_E,linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S(E)}$")
plt.plot(x,S_E_nc[:len(x)],linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S_{nc}(E)}$")
plt.plot(x,y_ZLP,linewidth=2.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(x,y_EEL,linewidth=2.5,color="red",label=r"${\rm sample}$")
plt.plot(x,J1_E,linewidth=2.5,color="green",ls="dotted",label=r"${\rm J^1(E)}$")

plt.ylim(0,0.3)


plt.figure()
plt.title("zoomed spectrum and calculated values")
plt.plot(x,S_E,linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S(E)}$")
plt.plot(x,y_ZLP,linewidth=1,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(x,y_EEL,linewidth=0.5,color="red",label=r"${\rm sample}$")
plt.plot(x,J1_E,linewidth=0.5,color="green",label=r"${\rm J^1(E)}$")
plt.plot(x,S_E_nc[:len(x)],linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S_{nc}(E)}$")
plt.xlim(0,55)
plt.ylim(-0.001,0.01)
plt.legend()




plt.figure()
plt.xlim(-5, 70)
#plt.xlim(40,60)
#plt.ylim(-0.0005,0.0015)
plt.title("sample spectrum and J1")
plt.plot(x,y_EEL,linewidth=1,color="green",label=r"${\rm sample}$")
plt.plot(x,J1_E,linewidth=1,color="blue", label=r"${\rm J1(E)}$")
plt.plot(x,S_E_nc[:len(x)],linewidth=1.5,color="pink",label=r"${\rm S_{nc}(E)}$")
plt.legend()
"""

#%%

xs = total_replicas['x14'].values
Nx = np.sum(total_replicas['x14']==total_replicas['x14'][0])
nx = int(len(total_replicas)/Nx)

x_14 = df_sample.iloc[5].x_shifted
y_14 = df_sample.iloc[5].y

#nx_2 = len(x_14)

#x_14 = xs[:nx]
#y_14 = total_replicas['data y14'].values[:nx]
y_ZLPs = total_replicas.match14.values


r = 3 #Drude model, can also use estimation from exp. data
n_times_extra = 2
l = len(x_14)
sem_inf = l*(n_times_extra+1)
ddeltaE = (x_14[-1]-x_14[0])/l


A = y_14[-1]
y_extrp = np.zeros(sem_inf)
x_extrp = np.linspace(x_14[0], (sem_inf-1)*ddeltaE+x_14[0], sem_inf)

y_extrp[:l] = y_14
x_extrp[:l] = x_14
y_extrp[l:] = A*np.power(1+x_extrp[l:]-x_extrp[l],-r)
x_14 = x_extrp
y_14 = y_extrp

i_nu = CFT(x_14,y_14)

S_14 = np.zeros((Nx, sem_inf))
S_nc_14 = np.zeros((Nx, sem_inf))
J1_14 = np.zeros((Nx, sem_inf))
ZLP_14 = np.zeros((Nx, sem_inf))


for i in range(Nx):
    y_ZLP = y_ZLPs[i*nx:(i+1)*nx]
    y_ZLP_extrp = np.zeros(sem_inf)
    y_ZLP_extrp[:nx] = y_ZLP
    y_ZLP_extrp[nx:] = y_ZLP[-1]*np.power(1+x_extrp[nx:]-x_extrp[nx],-r)
    y_ZLP = y_ZLP_extrp
    ZLP_14[i,:] = y_ZLP

    
    N_ZLP = 1#np.sum(y_ZLP)*ddeltaE
    
    z_nu = CFT(x_14,y_ZLP)
    s_nu = N_ZLP*np.log(i_nu/z_nu)
    j1_nu = z_nu*s_nu/N_ZLP
    
    S_E = np.real(iCFT(x_14,s_nu))
    s_nu_nc = s_nu
    s_nu_nc[500:-500] = 0
    S_E_nc = np.real(iCFT( x_14,s_nu_nc))
    J1_E = np.real(iCFT(x_14,j1_nu))
    
    S_14[i,:] = S_E
    S_nc_14[i,:] = S_E_nc
    J1_14[i,:] = J1_E

#%%
plt.figure()
plt.plot(y_14-y_ZLP, label = "$J_{EEL}(E)$")
plt.plot(J1_14[0,:], label = "$J_1(E)$")
plt.title("Scattering and single scattering spectrum 14")
plt.legend()

plt.figure()
plt.fill_between(x_14,y_14-np.average(ZLP_14,axis= 0)- np.std(ZLP_14,axis= 0),y_14- np.average(ZLP_14,axis= 0) +np.std(ZLP_14,axis= 0), color = [150/255, 150/255, 255/255])
plt.plot(x_14,y_14-np.average(ZLP_14,axis= 0), label = "$J_{EEL}(E)$")
plt.fill_between(x_14,np.average(J1_14,axis= 0)- np.std(J1_14,axis= 0), np.average(J1_14,axis= 0) +np.std(J1_14,axis= 0), color = [255/255, 220/255, 166/255])
plt.xlim(0,60)
plt.plot(x_14,np.average(J1_14,axis= 0),lw = 0.8, label = "avg. $J_1(E)$")
plt.title("Scattering and single scattering spectrum 14")
plt.legend()
#plt.ylim(-100, 9000)

plt.figure()
plt.fill_between(x_14,y_14-np.average(ZLP_14,axis= 0)- np.std(ZLP_14,axis= 0),y_14- np.average(ZLP_14,axis= 0) +np.std(ZLP_14,axis= 0), color = [150/255, 150/255, 255/255])
plt.plot(x_14,y_14-np.average(ZLP_14, axis = 0), label = "$J_{EEL}(E)$")
plt.fill_between(x_14,np.average(J1_14,axis= 0)- np.std(J1_14,axis= 0), np.average(J1_14,axis= 0) +np.std(J1_14,axis= 0), color = [255/255, 220/255, 166/255], hatch='+')
plt.xlim(1,7)
plt.plot(x_14,np.average(J1_14,axis= 0),lw = 0.8, label = "avg. $J_1(E)$")
plt.title("Scattering and single scattering spectrum 14")
plt.legend()
#plt.ylim(-100, 9000)


#%%

xs = total_replicas['x14'].values
Nx = np.sum(total_replicas['x14']==total_replicas['x14'][0])
nx = int(len(total_replicas)/Nx)

x_14 = df_sample.iloc[5].x_shifted
#y_14 = df_sample.iloc[5].y_smooth
y_14 = smooth(df_sample.iloc[5].y, 50)

#nx_2 = len(x_14)

#x_14 = xs[:nx]
#y_14 = total_replicas['data y14'].values[:nx]
y_ZLPs = total_replicas.match14.values


r = 3 #Drude model, can also use estimation from exp. data
n_times_extra = 2
l = len(x_14)
sem_inf = l*(n_times_extra+1)
ddeltaE = (x_14[-1]-x_14[0])/l


A = y_14[-1]
y_extrp = np.zeros(sem_inf)
x_extrp = np.linspace(x_14[0], (sem_inf-1)*ddeltaE+x_14[0], sem_inf)

y_extrp[:l] = y_14
x_extrp[:l] = x_14
y_extrp[l:] = A*np.power(1+x_extrp[l:]-x_extrp[l],-r)
x_14 = x_extrp
y_14 = y_extrp

i_nu = CFT(x_14,y_14)

S_14 = np.zeros((Nx, sem_inf))
S_nc_14 = np.zeros((Nx, sem_inf))
J1_14 = np.zeros((Nx, sem_inf))
ZLP_14 = np.zeros((Nx, sem_inf))

for i in range(Nx):
    y_ZLP = smooth(y_ZLPs[i*nx:(i+1)*nx],50)
    y_ZLP_extrp = np.zeros(sem_inf)
    y_ZLP_extrp[:nx] = y_ZLP
    y_ZLP_extrp[nx:] = y_ZLP[-1]*np.power(1+x_extrp[nx:]-x_extrp[nx],-r)
    y_ZLP = y_ZLP_extrp
    ZLP_14[i,:] = y_ZLP
    
    N_ZLP = 1#np.sum(y_ZLP)*ddeltaE
    
    z_nu = CFT(x_14,y_ZLP)
    s_nu = N_ZLP*np.log(i_nu/z_nu)
    j1_nu = z_nu*s_nu/N_ZLP
    
    S_E = np.real(iCFT(x_14,s_nu))
    s_nu_nc = s_nu
    s_nu_nc[500:-500] = 0
    S_E_nc = np.real(iCFT( x_14,s_nu_nc))
    J1_E = np.real(iCFT(x_14,j1_nu))
    
    S_14[i,:] = S_E
    S_nc_14[i,:] = S_E_nc
    J1_14[i,:] = J1_E

#%%
plt.figure()
plt.plot(y_14-y_ZLP, label = "$J_{EEL}(E)$")
plt.plot(J1_14[0,:], label = "$J_1(E)$")
plt.title("Smotthed scattering and single scattering spectrum 14")
plt.legend()

plt.figure()
plt.fill_between(x_14,y_14-np.average(ZLP_14,axis= 0)- np.std(ZLP_14,axis= 0),y_14- np.average(ZLP_14,axis= 0) +np.std(ZLP_14,axis= 0), color = [150/255, 150/255, 255/255])
plt.plot(x_14,y_14-np.average(ZLP_14, axis = 0), label = "$J_{EEL}(E)$")
plt.fill_between(x_14,np.average(J1_14,axis= 0)- np.std(J1_14,axis= 0), np.average(J1_14,axis= 0) +np.std(J1_14,axis= 0), color = [255/255, 220/255, 166/255])
plt.xlim(0,60)
plt.plot(x_14,np.average(J1_14,axis= 0),lw = 0.8, label = "avg. $J_1(E)$")
plt.title("Smoothed scattering and single scattering spectrum 14")
plt.legend()

plt.figure()
plt.fill_between(x_14,y_14-np.average(ZLP_14,axis= 0)- np.std(ZLP_14,axis= 0),y_14- np.average(ZLP_14,axis= 0) +np.std(ZLP_14,axis= 0), color = [150/255, 150/255, 255/255])
plt.plot(x_14,y_14-np.average(ZLP_14, axis = 0), label = "$J_{EEL}(E)$")
plt.fill_between(x_14,np.average(J1_14,axis= 0)- np.std(J1_14,axis= 0), np.average(J1_14,axis= 0) +np.std(J1_14,axis= 0), color = [255/255, 220/255, 166/255])
plt.xlim(1,7)
plt.plot(x_14,np.average(J1_14,axis= 0),lw = 0.8, label = "avg. $J_1(E)$")
plt.title("Smoothed scattering and single scattering spectrum 14")
plt.legend()