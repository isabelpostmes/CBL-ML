#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:05:21 2020

@author: isabel
"""

import numpy as np
import pandas as pd
from copy import copy
import scipy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc, cm



def process_data(df_spectra, wl1 = 50, wl2 = 100):
    """
    INPUT:
        df_spectra: pandas DataFrame with columns ('x','y'), to be shifted and 
                    normalized
    
    OUTPUT:
        df_sn: pandas DataFrame with columns ('x','y','x_shifted','y_norm')
        
    This function shifts the values of 'x', such that the maximum of 'y' occurs 
    at 'x' = 0. Furthermore, it normilizes 'y' by deviding by the integrated 
    value of 'y'.
    """
    
    df_spectra[['y_smooth', 'dy_dx', 'dy_dx_smooth']] = np.empty
    df_spectra[['pos_der']] = np.nan
    
    for i in df_spectra.index:
        x_shifted = df_spectra.iloc[i].x_shifted
        y_norm = df_spectra.iloc[i].y_norm
        
        #y_smooth
        y_smooth = smooth(y_norm, wl1)
        df_spectra.at[i, 'y_smooth'] = y_smooth 
        
        #dy_dx
        dy_dx = np.divide(y_smooth[1:]-y_smooth[:-1], x_shifted[1:]-x_shifted[:-1])
        df_spectra.at[i,'dy_dx'] = np.append(np.nan, dy_dx)
        
        #dy_dx_smooth
        df_spectra.at[i,'dy_dx_smooth'] = np.append(np.nan,smooth(dy_dx, wl2))
        
        #crossing
        #first negative derivative after dE=0:
        
        crossing = (df_spectra.at[i,'dy_dx'] > 0)
        up = np.argmax(crossing[np.argmax(y_smooth)+1:]) + np.argmax(y_smooth) +1
        pos_der = df_spectra.at[i,'x_shifted'][up]
        df_spectra.at[i, 'pos_der'] = pos_der
    
    return df_spectra



def plot_deriv(df_vacuum, df_sample):
    nrows, ncols = 2,1
    gs = matplotlib.gridspec.GridSpec(nrows,ncols)
    plt.figure(figsize=(ncols*7,nrows*4.5))
    
    max_end_peak = df_sample.pos_der.max()
    
    
    
    n_spectra = len(df_vacuum) + len(df_sample)
    cm_subsection = np.linspace(0,1,n_spectra) 
    colors = [cm.viridis(x) for x in cm_subsection]
    
    hfont = rc('font',**{'family':'sans-serif','sans-serif':['Sans Serif']})
              
    
    ax = plt.subplot(gs[0])
    #ax.set_xlim([0,9])
    #ax.tick_params(which='major',direction='in',length=7)
    #ax.tick_params(which='minor',length=8)
    plt.axhline(y=0, color='black', linewidth=1, alpha=.8)
    #plt.axvline(x=0, color='darkgray', linestyle='--', linewidth = 1)
    #plt.axvline(x=dE1, color='darkgray', linestyle='--', linewidth = 1, label='$\Delta$E1' %{'s': dE1})
    
    ax.tick_params(which='major', length= 10, labelsize=18)
    ax.tick_params(which='minor', length= 10, labelsize=10)
    ax.set_xlim([0, max_end_peak*2])
    ax.set_ylim([-.002, .001])
    ax.set_ylabel('dI/dE',fontsize=18)
    ax.set_yticks([-0.002, -0.001, 0, 0.001])
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    
    label_todo = True
    for j in df_vacuum.index:
        if label_todo:
            ax.plot(df_vacuum.iloc[j].x_shifted,df_vacuum.iloc[j].dy_dx_smooth, color=colors[0], label='vacuum')
            label_todo = False
        ax.plot(df_vacuum.iloc[j].x_shifted,df_vacuum.iloc[j].dy_dx_smooth, color=colors[0])
        ax.tick_params(labelbottom=True)
        
    label_todo = True
    for j in df_sample.index:
        if label_todo:
            ax.plot(df_sample.iloc[j].x_shifted,df_sample.iloc[j].dy_dx_smooth, color=colors[-1], label='sample')
            label_todo = False
        ax.plot(df_sample.iloc[j].x_shifted,df_sample.iloc[j].dy_dx_smooth, color=colors[-1])
        
    
    ax.legend(loc=2, fontsize=16)



    ax = plt.subplot(gs[1])
    ax.axhline(y=1, linestyle='-', color='gray')
    ax.axvline(x=1.65, linestyle='--')
    ax.set_ylim([-1, 2])
    ax.set_xlim([0, max_end_peak*2])  
    ax.set_ylabel('R = dI/dE(sample) / dI/dE(vac)', fontsize=18)
    ax.set_xlabel('$\Delta$E (eV)', fontsize=218)
    ax.set_xlabel('Energy loss (eV)', fontsize=24)
    ax.tick_params(length= 10, labelsize=18)
    ax.tick_params(which='major', length= 10, labelsize=18)
    ax.tick_params(which='minor', length= 10, labelsize=10)
      
    
    
    #TODO: rethink: this won't work with different range spectra. Still do this?
    vac_avg = np.average(df_vacuum.dy_dx[:])
    sam_avg = np.average(df_sample.dy_dx[:])
    ratio = sam_avg/vac_avg
    
    for j in df_sample.index:
        ax.plot(df_sample.iloc[j].x_shifted, np.divide(df_sample.iloc[j].dy_dx,vac_avg), '--', color = colors[-1])
   
    
    ax.plot(df_sample.iloc[0].x_shifted, ratio, linewidth = 2, color = colors[0], label = 'sample avg./vacuum avg.')

    ax.legend()   
    
    plt.tight_layout()
    #plt.savefig("Derivatives.pdf")
    plt.show()
    
    return







def ewd(x, y, nbins):  
    """
    INPUT:
        x: 
        y:
        nbins: 
            
    OUTPUT:
        df_train:
        cuts1:
        cuts2:
    
    Apply Equal Width Discretization (EWD) to x and y data to determine variances
    """
    #TODO: I think everything that was here isn't needed?? since x is already sorted, and a 1D array
    #df_train = np.array(np.c_[x,y])
    cuts1, cuts2 = pd.cut(x, nbins, retbins=True)
    
    return cuts1, cuts2

def CI_high(data, confidence=0.68):
    ## remove the lowest and highest 16% of the values
    
    a = 1.0 * np.array(data)
    n = len(a)
    b = np.sort(data)

    highest = np.int((1-(1-confidence)/2)*len(a))
    high_a = b[highest]
 
    return high_a

def CI_low(data, confidence=0.68):
    ## remove the lowest and highest 16% of the values
    
    a = 1.0 * np.array(data)
    n = len(a)
    b = np.sort(data)
    lowest = np.int(((1-confidence)/2)*len(a))
    low_a = b[lowest]

    return low_a

def get_mean(data):
    return np.mean(data)



def binned_statistics(x,y, nbins):
    """Find the mean, variance and number of counts within the bins described by ewd"""
    
    #df_train, 
    cuts1, cuts2 = ewd(x,y, nbins)
    mean, edges, binnum = scipy.stats.binned_statistic(x,y, statistic='mean', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='mean', bins=cuts2)
    var, edges, binnum = scipy.stats.binned_statistic(x,y, statistic='std', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='std', bins=cuts2)
    count, edges, binnum = scipy.stats.binned_statistic(x,y,statistic='count', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='count', bins=cuts2)
    low, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=CI_low, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=CI_low, bins=cuts2)
    high, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=CI_high, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=CI_high, bins=cuts2)
    mean2, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=get_mean, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=get_mean, bins=cuts2)
    
    
    
    return [mean, var, count, low, high, mean2], edges

def get_median(x,y,nbins):
    #df_train, 
    cuts1, cuts2 = ewd(x,y, nbins)
    median, edges, binnum = scipy.stats.binned_statistic(x,y,statistic='median', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='median', bins=cuts2)
    return median


def reduce_data(df, n_bins, x_min = -0.5, x_max = 20, method = 1):
    df_reduced = pd.DataFrame(columns=['x_r','y_r'])
    if method == 1:
        x = df.iloc[0].x_shifted
        y = df.iloc[0].y_norm
        for i in range(1,len(df)):
            x = np.concatenate((x, df.iloc[i].x_shifted))
            y = np.concatenate((y, df.iloc[i].y_norm))
        stats, x_edges = binned_statistics(x,y, nbins)
        y_mean, sigma, x_edges = stats[0], stats[1]
    return df





"""
#TODO: unused????
# =============================================================================
# def vectorize_variance(x,y, nbins):
#     #Apply the binned variances to the original training data
#     
#     df_train, cuts1, cuts2 = ewd(x,y, nbins)
#     mean, std, count = binned_statistics(x,y, nbins)
#     variance=[]
#     m=0
#     i=0
#     while i<len(count):
#         maximum = count[i]
# 
#         while m < maximum:
#             variance.append(std[i])
#             m+=1
#         else:
#             m=0
#             i+=1
#     return np.array(variance)
# 
# def vectorize_mean(x,y, nbins):
#     
#     #df_train, 
#     cuts1, cuts2 = ewd(x,y, nbins)
#     mean, std, count = binned_statistics(x,y,nbins)
#     means=[]
#     m=0
#     i=0
#     while i<len(count):
#         maximum = count[i]
# 
#         while m < maximum:
#             means.append(mean[i])
#             m+=1
#         else:
#             m=0
#             i+=1
#     return np.array(means)
# 
# def get_mean_pseudodata(x,y, nbins):
#     #df_train, 
#     #cuts1, cuts2 = ewd(x, y, nbins)
#     mean, std, count = binned_statistics(x,y,nbins)
#     meanvector = vectorize_mean(x,y,nbins)
#     stdvector = vectorize_variance(x,y,nbins)
#     return mean, std, meanvector, stdvector
# =============================================================================
"""

   
    
"""Neural network functions: """
"""
#TODO: UNUSED???
# =============================================================================
# 
# def custom_cost(y_true, y_pred):
#     '''Chi square function'''
#     return tf.reduce_mean(tf.square((y_true-y_pred)/sigma))
# 
# def mean_pred(y_true, y_pred):
#     return K.mean(y_pred)
# 
# 
##ΤΟDO: find places bootstrap is called and add df_train, replace names here 
# def bootstrap(df_train):
#     df_train_a, df_train_b = train_test_split(df_train, test_size=0.5)
#     df_train_1, df_train_2 = train_test_split(df_train_a, test_size=0.5)
#     df_train_3, df_train_4 = train_test_split(df_train_b, test_size=0.5)
#     
#     return df_train_1, df_train_2, df_train_3, df_train_4
#   
# =============================================================================
""" 


   
def smooth(x,window_len=10,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    """
    
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    index = int(window_len/2)
    return y[(index-1):-(index)]

def gaussian(x, amp, cen, std):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    y = (amp) * np.exp(-(x-cen)**2 / (2*std**2))
    return y

    
def window(x,y, minval, maxval):
    """Function applies a window to arrow"""
    
    low = next(i for i, val in enumerate(x) if val > minval)
    treshold_min = str(low)
    treshold_min = int(treshold_min)
    up = next(i for i, val in enumerate(x) if val > maxval)
    treshold_max = str(up)
    treshold_max = int(treshold_max)
    x = x[treshold_min:treshold_max]
    y = y[treshold_min:treshold_max]
    
    return x,y
    

def residuals(prediction, y, std):
    res = np.divide((prediction - y), std)
    return res