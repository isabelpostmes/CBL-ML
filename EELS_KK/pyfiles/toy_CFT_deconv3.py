#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 01:07:53 2020

@author: isabel
"""
#ANOTHER TRY TO DO IT MYSELF

import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

def CFT_approx(x, y):
    x_0 = np.min(x)
    x_max = np.max(x)
    N = len(x)
    delta_x = (x_max-x_0)/N
    delta_omg = 2*np.pi/(N*delta_x)
    F_k = (1 + 1j) * np.zeros(N)
    n = np.linspace(0, N-1, N)
    for k in range(N):
        cont_factor = (2*np.pi)**-0.5*np.exp(-1j*x_0*k*delta_omg)*delta_x
        #cont_factor = np.exp(-1j*x_0*k*delta_omg)*delta_x
        exp_factor = -2j*np.pi*n*k/N
        F_k[k] = cont_factor * np.sum(np.exp(exp_factor)*y)
    return F_k

def iCFT_approx(x, Y_k):
    x_0 = np.min(x)
    x_max = np.max(x)
    N = len(x)
    delta_x = (x_max-x_0)/N
    delta_omg = 2*np.pi/(N*delta_x)
    f_n = (1 + 1j) * np.zeros(N)
    #n = np.linspace(0, N-1, N)
    k = np.linspace(0, N-1, N)
    for n in range(N):
        x_n = x[n]
        cont_factor = (2*np.pi)**-0.5*delta_omg
        #cont_factor = (2*np.pi)**-1*delta_omg
        exp_factor = 1j*k*delta_omg*x_0 + 2j*np.pi*n*k/N
        #exp_factor = 1j*k*delta_omg*x_n
        f_n[n] = cont_factor * np.sum(np.exp(exp_factor)*Y_k)
    return f_n


def convolute(xf, yf, xg, yg):
    Nf = len(xf)
    Ng = len(xf)
    Nh = Nf+Ng-1
    xf0 = np.min(xf)
    xf1 = np.max(xf)
    deltaxf = (xf1-xf0)/Nf
    xg0 = np.min(xg)
    xg1 = np.max(xg)
    deltaxg = (xg1-xg0)/Ng
    h_n = np.zeros(Nh)
    x_n = np.zeros(Nh)
    if not deltaxf == deltaxg:
        print("delta_x_f and delta_x_g not equal, unable to convolute")
        return x_n, h_n
    
    x_n = np.linspace(xf0+xg0, xf1+xg1, Nh)
    yg_expanded = np.zeros(Ng + 2*(Nf-1))
    yg_expanded[Nf-1:Ng+Nf-1] = yg
    
    for n in range(Nh):
        h_n[n] = np.sum(yf[::-1] * yg_expanded[n:Nf+n])*deltaxf
    
    return x_n, h_n

def ft(x, samples):
    """Approximate the Fourier Transform of a time-limited 
    signal by means of the discrete Fourier Transform.
    
    samples: signal values sampled at the positions t0 + n/Fs
    Fs: Sampling frequency of the signal
    t0: starting time of the sampling of the signal
    """
    x_0 = np.min(x)
    x_max = np.max(x)
    N = len(x)
    delta_x = (x_max-x_0)/N
    Fs = 1/delta_x
    t0 = -x_0
    f = np.linspace(-Fs/2, Fs/2, len(samples), endpoint=False)
    return np.fft.fftshift(np.fft.fft(samples)/Fs * np.exp(-2j*np.pi*f*t0))

def gauss(DeltaE, sigma, A, shift = 0):
    izlp = A*np.exp( -np.power((DeltaE-shift),2)/np.power(sigma,2) )
    return izlp

#try multiples:
N = 1000
x = np.linspace(-2,4,N)
y1 = np.zeros(N)
y1[200:300] = 1
x1, y1c = convolute(x,y1,x,y1)

Y1 = CFT_approx(x,y1)
Y1T = (6.2830)**0.5*np.power(Y1, 2)
Y1C = CFT_approx(x1,y1c)

Y12 = np.zeros(len(Y1T)*2-1)
for i in range(len(Y12)):
    if i%2 == 0:
        #print(i)
        Y12[i] = Y1T[int(i/2)]
    else:
        Y12[i] = 0#(Y1T[int((i-1)/2)] + Y1T[int((i+1)/2)])/2


plt.figure()
#plt.plot(Y1, label = "Y1")
#plt.plot(Y1T, label = "Y1*Y1")
plt.plot(Y1C, label = "FT(y1c)")
plt.plot(Y12, 'o', label = "Y1*Y1, 2")
plt.legend()
plt.xlim(0,50)

plt.figure()
plt.plot(np.imag(Y1), label = "Y1")
plt.plot(np.imag(Y1T), label = "Y1*Y1")
plt.plot(np.imag(Y1C), label = "FT(y1c)")
#plt.plot(Y12, 'o', label = "Y1*Y1, 2")
plt.legend()
plt.xlim(0,50)


N = 1000
x = np.linspace(-2,4,N)
y2 = gauss(x, 0.5,0.8,-1)
x2, y2c = convolute(x,y1,x,y1)

Y2 = CFT_approx(x,y2)
Y2T = (6.2830)**0.5*np.power(Y2, 2)
Y2C = CFT_approx(x2,y2c)

Y22 = np.zeros(len(Y2)*2-1)
for i in range(len(Y22)):
    if i%2 == 0:
        #print(i)
        Y22[i] = Y2T[int(i/2)]
    else:
        Y22[i] = 0#(Y2T[int((i-1)/2)] + Y2T[int((i+1)/2)])/2



plt.figure()
#plt.plot(Y2, label = "Y2")
plt.plot(Y2T, label = "Y2*Y2")
plt.plot(Y2C, label = "FT(y2c)")
plt.plot(Y22, 'o', label = "Y2*Y2, 2")
plt.legend()
plt.xlim(0,50)