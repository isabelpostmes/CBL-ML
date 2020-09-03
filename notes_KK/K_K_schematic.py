#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:41:26 2020

@author: isabel

Schematic Kramer-Kronig analysis


"""
import numpy as np
import math

shape_J = 100

J1_E = np.ones(shape_J)
E = np.linspace(1,shape_J, shape_J)


#step 1: modulate intensity

#TODO: change variables to correct values
beta = 1
m_0 = 1
v = 0.5 #needs to be smaller than c
c = 1 #natural units?
gamma = (1-v**2/c**2)**-0.5



theta_E = E/(gamma*m_0*v**2)
log_term = np.log(1+(beta/theta_E)**2)

J_ac = J1_E/log_term


#step 2: extrapolation
r = 3 #Drude model, can also use estimation from exp. data
A = J_ac[-1]
n_times_extra = 10
sem_inf = shape_J*(n_times_extra+1)

dE = (E[-1]-E[0])/E.size

J_extrp = np.zeros(sem_inf)
E_extrp = np.linspace(E[0], sem_inf*dE+dE, sem_inf)

J_extrp[:shape_J] = J_ac
E_extrp[:shape_J] = E

J_extrp[shape_J:] = A*np.power(E_extrp[shape_J:],r)


#step 3: normalisation

Re_eps0 = 0 #value of Re[1/eps(0)]
int_J_over_E = np.sum(J_extrp/E_extrp)*dE
K = 2*int_J_over_E/(math.pi*(1-Re_eps0))


#step 4: retreiving ε
Im_eps = J_extrp/K #Im[-1/eps(E)]

Re_eps = np.zeros(sem_inf) # Re[1/eps(0)]

#integrate around each energy (to avoid singularities)
for i in range(E_extrp.size):
    E = E_extrp[i]
    select = (E_extrp != E)
    Re_eps[i] = 1 - 2/math.pi * np.sum(Im_eps[select]*E_extrp[select]/(np.power(E_extrp[select],2)-E**2))








