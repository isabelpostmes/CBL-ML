#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:17:21 2020

@author: isabel
"""
import math
import numpy as np
import matplotlib.pyplot as plt


def  kramer_kronig_egt(x, y, N_ZLP = 1, plot = False):
    #TODO: change variables to correct values
    #N_ZLP = 1 #1 als prob
    E = 200 #keV =e0??
    beta = 30 #mrad
    ri = 3 #refractive index
    nloops = 2 #number of iterations
    delta = 0.3 #eV stability parameter (0.1eV-0.5eV)
    m_0 = 511.06
    a_0 = 5.29 #nm, borhradius
    
    l = len(x)
    
    #extraplolate
    semi_inf = 2**math.floor(math.log2(l)+1)*4 #waar komt dit getal vandaan?? egerton: nn
    EELsample = np.zeros(semi_inf) #egerton: ssd
    EELsample[1:l+1] = y
    I = EELsample #egerton: d
    ddeltaE = (x[-1]-x[0])/l #energy/channel, egerton: epc
    deltaE = np.linspace(0, semi_inf-1, semi_inf)*ddeltaE + x[0] #egerton: e
    
    
    gamma = 1 + E/m_0
    T = E*(1+E/(2*m_0))/gamma**2 #kin energy? egerton: t=mv^2/2
    rk_0 = 2590*gamma*(2*T/m_0)**0.5 #TODO: waar komt 2590 vandaan????
    tgt = E * (2*m_0 + E)/ (m_0 + E)
    tgt = 2*gamma*T
    
    
    
    v = (2*E/m_0 *(1+E/(2*m_0))/gamma**2)**0.5
    theta_E = deltaE/tgt
    log_term = np.log(1+(beta/theta_E)**2)
    
    
    plt.figure()
    plt.title("eps over loops")
    for i in range(nloops):
        N_S = np.sum(I)*ddeltaE #integral over ssd
        I_ac = I/log_term
        I_ac_over_deltaE_int = np.sum(I_ac/deltaE)*ddeltaE
        
        K = I_ac_over_deltaE_int/ (math.pi/2) / (1-ri**-2) #normilized sum over I_ac/E, egerton: rk
        a_0pi2 = 332.5 #TODO ??????????? WHYY, factor 10???
        t_nm = K * a_0pi2 *T/ N_ZLP
        N_rat = N_S/N_ZLP #t/lambda blijkbaar, egerton: tol
        lambd = t_nm/N_rat
        
        Im_eps = I_ac/K #Im(-1/eps), egerton: imreps
        S_p = 2*np.imag(np.fft.fft(Im_eps)) / semi_inf #waarom /l ???? gek..., sine transform p(t)
        sgn = np.ones(semi_inf)
        half = math.floor(semi_inf/2)
        sgn[-half:] *= -1
        C_q = sgn*S_p
        Re_eps = np.fft.fft(C_q) #wrs daarom /l: geen inverse maar nog een keer FT
        
        #TODO WHY? correct function for reflected tail (?) #different from book, why?
        Re_eps_mid = np.real(Re_eps[half])
        cor = Re_eps_mid/2*np.power(half/np.linspace(semi_inf-1, half, half),2)
        Re_eps[:half] += 1 - cor
        Re_eps[half:] = 1 + cor #completely overwrite??
        
        eps_sq = np.power(Re_eps, 2) + np.power(Im_eps, 2)
        eps1 = Re_eps/eps_sq
        eps2 = Im_eps/eps_sq
        eps = eps1+ 1j*eps2
        
        plt.plot(deltaE[:2*l], eps1[:2*l], label = "eps1, i="+str(i))
        plt.plot(deltaE[:2*l], eps1[:2*l], label = "eps1, i="+str(i))
        
        
        #surface iter
        srf_eps_term = np.imag(-4/(1+eps)) - Im_eps #gerton: srfelf
        adep = tgt/(deltaE + delta) * (np.arctan(beta/theta_E)/theta_E - beta*1E-3/(beta**2-theta_E**2)) #waarom opeens beta mili maken? 
        S_s = 2000*K/rk_0/t_nm*adep*srf_eps_term #TODO 2000: 2*1000 ???, reken termen na
        plt.plot(deltaE[:2*l], S_s[:2*l]/np.max(np.abs(S_s)), label = "S_s, i=" + str(i))
        I = EELsample - S_s
    
    plt.legend()
    
    return eps, t_nm


MoS2_eloss = np.loadtxt('MoS2_data/MoS2_eloss.txt')
MoS2_eps = np.loadtxt('MoS2_data/MoS2_eps.txt')

deltaE =  MoS2_eloss[:,0]
eloss_x = MoS2_eloss[:,1]
eloss_z = MoS2_eloss[:,2]

#eps_MoS2_x, t_x = kramer_kronig_egt(deltaE, eloss_x, plot=True)#, method = 2)
#eps_MoS2_z, t_z = kramer_kronig_egt(deltaE, eloss_z, plot=True)#, method = 2)

l = len(deltaE)
ddeltaE = (deltaE[-1]-deltaE[0])/l #energy/channel, egerton: epc
deltaE = np.linspace(0, len(eps_MoS2_x)-1, len(eps_MoS2_x))*ddeltaE + deltaE[0] #egerton: e


plt.figure()
plt.title("dieelctric function MoS2, x direction")
plt.plot(deltaE, np.real(eps_MoS2_x), label = r'$\varepsilon_1$')
plt.plot(deltaE, np.imag(eps_MoS2_x), label = r'$\varepsilon_2$')
plt.legend()
#%%
plt.figure()
plt.title("dieelctric function MoS2, z direction")
plt.plot(deltaE, np.real(eps_MoS2_z), label = r'$\varepsilon_1$')
plt.plot(deltaE, np.imag(eps_MoS2_z), label = r'$\varepsilon_2$')
plt.legend()



plt.figure()
plt.title("dieelctric function MoS2, x direction")
plt.plot(deltaE[:l], np.real(eps_MoS2_x)[:l], label = r'$\varepsilon_1$')
plt.plot(deltaE[:l], np.imag(eps_MoS2_x)[:l], label = r'$\varepsilon_2$')
plt.legend()


plt.figure()
plt.title("dieelctric function MoS2, z direction")
plt.plot(deltaE[:l], np.real(eps_MoS2_z)[:l], label = r'$\varepsilon_1$')
plt.plot(deltaE[:l], np.imag(eps_MoS2_z)[:l], label = r'$\varepsilon_2$')
plt.legend()


plt.figure()
plt.title("dieelctric function MoS2, x direction, optocal values")
plt.plot(MoS2_eps[:,0][:l], MoS2_eps[:,1][:l], label = r'$\varepsilon_1$')
plt.plot(MoS2_eps[:,0][:l], MoS2_eps[:,2][:l], label = r'$\varepsilon_2$')
plt.legend()


plt.figure()
plt.title("dieelctric function MoS2, z direction, optocal values")
plt.plot(MoS2_eps[:,0][:l], MoS2_eps[:,3][:l], label = r'$\varepsilon_1$')
plt.plot(MoS2_eps[:,0][:l], MoS2_eps[:,4][:l], label = r'$\varepsilon_2$')
plt.legend()

#%%
xs = total_replicas['x14'].values
Nx = np.sum(total_replicas['x14']==total_replicas['x14'][0])
nx = int(len(total_replicas)/Nx)

x_14 = df_sample.iloc[5].x_shifted
y_14 = df_sample.iloc[5].y

y_ZLPs = total_replicas.match14.values

l = len(x_14)

ZLP_14 = np.zeros((Nx, l))


for i in range(Nx):
    y_ZLP = np.zeros(l)
    y_ZLP[:nx] = y_ZLPs[i*nx:(i+1)*nx]
    ZLP_14[i,:] = y_ZLP

deltaE = x_14
ZLP = np.average(ZLP_14, axis = 0)
EELsample = y_14-ZLP

ddeltaE = (np.max(deltaE)-np.min(deltaE))/l
N_ZLP = np.sum(ZLP) * ddeltaE

eps_14, t_14 = kramer_kronig_egt(deltaE, EELsample, N_ZLP = N_ZLP, plot=True)#, method = 2)

print("found thickness: ", round(t_14,3), "nm")

plt.figure()
plt.title("dieelctric function spectrum 14")
plt.plot(deltaE, np.real(eps_14)[:l], label = r'$\varepsilon_1$')
plt.plot(deltaE, np.imag(eps_14)[:l], label = r'$\varepsilon_2$')
plt.legend()



