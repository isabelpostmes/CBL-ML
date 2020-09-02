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
E = np.linspace(0,shape_J-1, shape_J)


#step 1: modulate intensity

#TODO: change variables to correct values
beta = 1
m_0 = 1
v = 0.5 #needs to be smaller than c
c = 1 #natural units?
gamma = (1-v**2/c**2)**-0.5



theta_E = E/(gamma*m_0*v**2)
log_term = np.log(1+(beta/theta_E)**2)

I_ac = J1_E/theta_E


#step 2: extrapolation
r = 3 #Drude model, can also use estimation from exp. data
A = I_ac[-1]
n_times_extra = 10
sem_inf = shape_J*(n_times_extra+1)

dE = (E[-1]-E[0])/E.size

I_extrp = np.zeros(sem_inf)
E_extrp = np.linspace(E[0], sem_inf*dE+dE, sem_inf)

I_extrp[:shape_J] = I_ac
E_extrp[:shape_J] = E

I_extrp[shape_J:] = A*np.power(E_extrp[shape_J:],r)


#step 3: normalisation


