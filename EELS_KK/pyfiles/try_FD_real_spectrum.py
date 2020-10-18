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


def exp_decay(x, r):
    """ y_offset * np.power(x - x[0], -r) """
    return y_offset * np.power(1 + x - x[0], -r)


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
n_times_extra = 10
l = len(x)
sem_inf = l*(n_times_extra+1)

ddeltaE = (x[-1]-x[0])/l

y_extrp = np.zeros(sem_inf)
y_ZLP_extrp = np.zeros(sem_inf)
x_extrp = np.linspace(x[0], sem_inf*ddeltaE+x[0], sem_inf)

y_ZLP_extrp[:l] = y_ZLP

y_extrp[:l] = y
x_extrp[:l] = x

y_extrp[l:] = A*np.power(1+x_extrp[l:]-x_extrp[l],-r)

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

try_only_ZLP = True
if try_only_ZLP:
    y = y_ZLP

z_nu = scipy.fft.fft(y_ZLP)
i_nu = scipy.fft.fft((y))
abs_i_nu = np.absolute(i_nu)
max_i_nu = np.max(abs_i_nu)
i_nu_copy = np.copy(i_nu)
#i_nu[abs_i_nu<max_i_nu*0.00000000000001] = 0
N_ZLP = scipy.integrate.cumtrapz(y_ZLP, x, initial=0)[-1]#1 #arbitrary units??? np.sum(EELZLP)

s_nu = N_ZLP*np.log(i_nu/z_nu)
j1_nu = z_nu*s_nu/N_ZLP
#s_nu_2 = s_nu
#s_nu_2[np.isnan(s_nu)] = 0#1E10 #disregard NaN values, but setting them to 0 doesnt seem fair, as they should be inf

plt.figure()
plt.plot(s_nu)
plt.plot(j1_nu)
#s_nu[150:1850] = 0
S_E = np.real(scipy.fft.ifft(s_nu))
J1_E = np.real(scipy.fft.ifft(j1_nu))


plt.figure()
plt.plot(z_nu)
plt.figure()
plt.plot(i_nu)

plt.figure()
plt.plot(i_nu/z_nu)
plt.figure()
plt.plot(s_nu)

plt.figure()
plt.plot(x,y,linewidth=2.5,color="black",label=r"${\rm total}$")
plt.plot(x,y_ZLP,linewidth=2.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(x,y_EEL,linewidth=2.5,color="red",ls="dashdot",label=r"${\rm sample}$")
plt.plot(x,S_E[:len(x)],linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S(E)}$")
plt.plot(x,J1_E,linewidth=2.5,color="green",ls="dashdot",label=r"${\rm J^1(E)}$")

plt.legend()
# Now produce the plot        
plt.xlabel(r"${\rm Energy~loss~(eV)}$",fontsize=17)
plt.ylabel(r"${\rm Intensity~(a.u.)}$",fontsize=17)
plt.xlim(1.45,5.0)
plt.ylim(0,0.01)

#l = 160
plt.figure()
plt.plot(x,S_E,linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S(E)}$")
plt.plot(x,y_ZLP,linewidth=2.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(x,y_EEL,linewidth=2.5,color="red",label=r"${\rm sample}$")
plt.plot(x,J1_E,linewidth=2.5,color="green",ls="dotted",label=r"${\rm J^1(E)}$")

plt.ylim(0,0.3)

plt.figure()
plt.plot(x,y_EEL,linewidth=0.5,color="red",label=r"${\rm sample}$")
plt.plot(x,J1_E,linewidth=0.5,color="green", label=r"${\rm S(E)}$")


"""
x = df_sample.iloc[1].x_shifted
y = df_sample.iloc[1].y_smooth

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
y_EEL = y - y_ZLP

plt.figure()
plt.plot(x,y)
plt.plot(x, y_ZLP)
plt.plot(x, y_EEL)
plt.xlim(0,5)
plt.ylim(0,0.02)


plt.figure()
plt.plot(x,y)
plt.plot(x, y_EEL)
#plt.xlim(0,5) 



z_nu = scipy.fft.fft(y_ZLP)
i_nu = scipy.fft.fft((y))
abs_i_nu = np.absolute(i_nu)
max_i_nu = np.max(abs_i_nu)
i_nu_copy = np.copy(i_nu)
#i_nu[abs_i_nu<max_i_nu*0.00000000000001] = 0
N_ZLP = scipy.integrate.cumtrapz(y_ZLP, x, initial=0)#1 #arbitrary units??? np.sum(EELZLP)

s_nu = N_ZLP*np.log(i_nu/z_nu)

s_nu_2 = s_nu
s_nu_2[np.isnan(s_nu)] = 0#1E10 #disregard NaN values, but setting them to 0 doesnt seem fair, as they should be inf

plt.figure()
plt.plot(s_nu)
#s_nu[150:1850] = 0
S_E = np.real(scipy.fft.ifft(s_nu_2))

plt.figure()
plt.plot(z_nu)
plt.figure()
plt.plot(i_nu)

plt.figure()
plt.plot(i_nu/z_nu)
plt.figure()
plt.plot(s_nu)

plt.figure()
plt.plot(x,y,linewidth=2.5,color="black",label=r"${\rm total}$")
plt.plot(x,y_ZLP,linewidth=2.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(x,y_EEL,linewidth=2.5,color="red",ls="dashdot",label=r"${\rm sample}$")
plt.plot(x,S_E[:len(x)],linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S(E)}$")
plt.legend()
# Now produce the plot        
plt.xlabel(r"${\rm Energy~loss~(eV)}$",fontsize=17)
plt.ylabel(r"${\rm Intensity~(a.u.)}$",fontsize=17)
plt.xlim(1.45,5.0)
plt.ylim(0,0.01)

l = 160
plt.figure()
plt.plot(x,S_E,linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S(E)}$")
plt.plot(x,y_ZLP,linewidth=2.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(x,y_EEL,linewidth=2.5,color="red",label=r"${\rm sample}$")
plt.ylim(0,0.3)
"""