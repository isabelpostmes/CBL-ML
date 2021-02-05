#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:20:45 2021

@author: isabel
TRAINING ZLP MODEL
"""
import numpy as np
import pandas as pd
from copy import copy
import scipy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc, cm



#TODO: change from binned statistics to eliminate hortizontal uncertainty?


#FROM LAURIEN
def CI_high(data, confidence=0.68):
    ## remove the lowest and highest 16% of the values
    
    a = np.array(data)
    n = len(a)
    b = np.sort(data)

    highest = np.int((1-(1-confidence)/2)*n)
    high_a = b[highest]
 
    return high_a

def CI_low(data, confidence=0.68):
    ## remove the lowest and highest 16% of the values
    
    a = np.array(data)
    n = len(a)
    b = np.sort(data)
    lowest = np.int(((1-confidence)/2)*n)
    low_a = b[lowest]

    return low_a

def get_median(x,y,nbins):
    #df_train, 
    cuts1, cuts2 = ewd(x,y, nbins)
    median, edges, binnum = scipy.stats.binned_statistic(x,y,statistic='median', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='median', bins=cuts2)
    return median

def get_mean(data):
    return np.mean(data)

def ewd(x, nbins):  
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



def binned_statistics(x,y, nbins, stats = None):
    """Find the mean, variance and number of counts within the bins described by ewd"""
    if stats is None:
        stats = []
        edges = None
    
    x = np.tile(x, len(y))
    y = y.flatten()
        
    
    
    #df_train, 
    cuts1, cuts2 = ewd(x,nbins)
    result = []
    if "mean" in stats:
        mean, edges, binnum = scipy.stats.binned_statistic(x,y, statistic='mean', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='mean', bins=cuts2)
        result.append(mean)
    if "var" in stats:
        var, edges, binnum = scipy.stats.binned_statistic(x,y, statistic='std', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='std', bins=cuts2)
        result.append(var)
    if "count" in stats:
        count, edges, binnum = scipy.stats.binned_statistic(x,y,statistic='count', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='count', bins=cuts2)
        result.append(count)
    if "low" in stats:
        low, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=CI_low, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=CI_low, bins=cuts2)
        result.append(low)
    if "high" in stats:
        high, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=CI_high, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=CI_high, bins=cuts2)
        result.append(high)
    if "mean2" in stats:
        mean2, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=get_mean, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=get_mean, bins=cuts2)
        result.append(mean2)
    
    return result, edges



def gaussian(x, amp, cen, std):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    y = (amp) * np.exp(-(x-cen)**2 / (2*std**2))
    return y


def smooth_lau(x,window_len=10,window='hanning'):
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

def smooth_im(self, window_len=10,window='hanning', keep_original = False):
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
    #TODO: add comnparison
    window_len += (window_len+1)%2
    s=np.r_['-1', self.data[:,:,window_len-1:0:-1],self.data,self.data[:,:,-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    #y=np.convolve(w/w.sum(),s,mode='valid')
    surplus_data = int((window_len-1)*0.5)
    if keep_original:
        self.data_smooth = np.apply_along_axis(lambda m: np.convolve(m, w/w.sum(), mode='valid'), axis=1, arr=s)[:,:,surplus_data:-surplus_data]
    else:
        self.data = np.apply_along_axis(lambda m: np.convolve(m, w/w.sum(), mode='valid'), axis=1, arr=s)[:,:,surplus_data:-surplus_data]
    
    
    return #y[(window_len-1):-(window_len)]

def smooth(data, window_len=10,window='hanning', keep_original = False):
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
    #TODO: add comnparison
    window_len += (window_len+1)%2
    s=np.r_['-1', data[:,window_len-1:0:-1],data,data[:,-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    #y=np.convolve(w/w.sum(),s,mode='valid')
    surplus_data = int((window_len-1)*0.5)
    return np.apply_along_axis(lambda m: np.convolve(m, w/w.sum(), mode='valid'), axis=1, arr=s)[:,surplus_data:-surplus_data]



def fun_clusters(clusters, function, **kwargs):
    #TODO
    pass

def smooth_clusters(image, clusters, window_len = None):
    smoothed_clusters = np.zeros((image.n_clusters), dtype = object)
    for i in range(image.n_clusters):
        smoothed_clusters[i] = smooth(clusters[i])
    return smoothed_clusters

def derivative_clusters(image, clusters):
    dx = image.ddeltaE
    der_clusters = np.zeros((image.n_clusters), dtype = object)
    for i in range(image.n_clusters):
        der_clusters[i] = (clusters[i][:,1:]-clusters[i][:,:-1])/dx
    return der_clusters
    

def residuals(prediction, y, std):
    res = np.divide((prediction - y), std)
    return res


def train_NN(image, spectra):#, vacuum_in):
    #oud:
    wl1 = 50
    wl2 = 100
    
    #new??? #TODO
    wl1 = round(image.l/20)
    wl2 = wl1*2
    nbins = round(image.l/4)#150
    
    
    #filter out negatives and 0's
    
    
    
    spectra_smooth = smooth_clusters(image, spectra, wl1)
    dy_dx = derivative_clusters(image, spectra_smooth)
    smooth_dy_dx = smooth_clusters(image, dy_dx, wl2)
    dE1s = find_clusters_dE1(image, smooth_dy_dx, spectra_smooth)
    
    
    dE1 = determine_dE1(image, dE1s, dy_dx)
    
    
    #TODO: instead of the binned statistics, just use xth value to dischart -> neh says Juan    
    
    dE2 = determine_dE2(image, spectra[0], nbins, dE1)
    
    print("dE1 & dE2:", dE1, dE2)
    
    
    
    
    
    
    pass

def find_dE1(image, dy_dx, y_smooth):
    #crossing
    #first positive derivative after dE=0:
    
    crossing = (dy_dx > 0)
    up = np.argmax(crossing[np.argmax(y_smooth)+1:]) + np.argmax(y_smooth) +1
    pos_der = image.deltaE[up]
    return pos_der
    


def find_clusters_dE1(image, dy_dx_clusters, y_smooth_clusters):
    dE1_clusters = np.zeros(image.n_clusters, dtype=object)
    for i in range(image.n_clusters):
        dy_dx_cluster = dy_dx_clusters[i]
        y_smooth_cluster = y_smooth_clusters[i]
        dE1_cluster = np.zeros(len(y_smooth_cluster))
        for j in range(len(dy_dx_cluster)):
            dy_dx = dy_dx_cluster[j]
            y_smooth = y_smooth_cluster[j]
            dE1_cluster[j] = find_dE1(image, dy_dx, y_smooth)
        dE1_clusters[i] = dE1_cluster
        i_avg = round(np.average(dE1_clusters[i]),4)
        i_std = round(np.std(dE1_clusters[i]), 4)
        i_min = round(np.min(dE1_clusters[i]),4)
        print("dE1 cluster ", i, " avg: ", i_avg, ", std: ", i_std, ", min: ", i_min)
    return dE1_clusters
    pass

def determine_dE1(image, dE1_clusters, dy_dx_clusters = None, check_with_user =True):
    dE1_min_avg = np.average(dE1_clusters[0])
    for i in range(1,image.n_clusters):
        dE1_avg_cluster = np.average(dE1_clusters[i])
        if dE1_avg_cluster < dE1_min_avg:
            dE1_min_avg = dE1_avg_cluster
    
    if not check_with_user:
        return dE1_min_avg
    
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if len(colors) < image.n_clusters:
        print("thats too many clusters to effectively plot, man")
        return dE1_min_avg
        #TODO: be kinder
    der_deltaE = image.deltaE[:-1]
    plt.figure()
    for i in range(image.n_clusters):
        dE1_i_avg = np.average(dE1_clusters[i])
        dE1_i_std = np.std(dE1_clusters[i])
        #dE1_i_min = np.min(dE1_clusters[i])
        dx_dy_i_avg = np.average(dy_dx_clusters[i], axis = 0)
        dx_dy_i_std = np.std(dy_dx_clusters[i], axis = 0)
        #dx_dy_i_min = np.min(dy_dx_clusters[i], axis = 0)
        plt.fill_between(der_deltaE, dx_dy_i_avg-dx_dy_i_std, dx_dy_i_avg+dx_dy_i_std, color = colors[i], alpha = 0.2)
        plt.axvspan(dE1_i_avg-dE1_i_std, dE1_i_avg+dE1_i_std, color = colors[i], alpha=0.1)
        plt.vlines(dE1_i_avg, -2, 1, ls = '--', color= colors[i])
        if i == 0:
            lab = "vacuum"
        else:
            lab = "sample cl." + str(i)
        plt.plot(der_deltaE, dx_dy_i_avg, color = colors[i], label = lab)
    plt.plot([der_deltaE[0], der_deltaE[-1]],[0,0], color = 'black')
    plt.title("derivatives of EELS per cluster, and range of first \npositive derivative of EELSs per cluster")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("dy/dx")
    plt.legend()
    plt.xlim(dE1_min_avg/2, dE1_min_avg*3)
    plt.ylim(-3e3,2e3)
    
    
    plt.figure()
    dx_dy_0_avg = np.average(dy_dx_clusters[0], axis = 0)
    dx_dy_0_std = np.std(dy_dx_clusters[0], axis = 0)
    for i in range(1,image.n_clusters):
        dE1_i_avg = np.average(dE1_clusters[i])
        dE1_i_std = np.std(dE1_clusters[i])
        #dE1_i_min = np.min(dE1_clusters[i])
        dx_dy_i_avg = np.average(dy_dx_clusters[i], axis = 0)
        dx_dy_i_std = np.std(dy_dx_clusters[i], axis = 0)
        #dx_dy_i_min = np.min(dy_dx_clusters[i], axis = 0)
        #plt.fill_between(image.deltaE, dx_dy_i_avg-dx_dy_i_std, dx_dy_i_avg+dx_dy_i_std, color = colors[i], alpha = 0.2)
        #plt.axvspand(dE1_i_avg-dE1_i_std, dE1_i_avg+dE1_i_std, color = colors[i], alpha=0.1)
        plt.vlines(dE1_i_avg, -2, 1, color= colors[i])
        lab = "sample cl." + str(i)
        plt.plot(der_deltaE, dx_dy_i_avg/dx_dy_0_avg, color = colors[i], label = lab)
    plt.plot([der_deltaE[0], der_deltaE[-1]],[1,1], color = 'black')
    plt.title("ratio between derivatives of EELS per cluster and the  \nderivative of vacuum cluster, and average of first positive \nderivative of EELSs per cluster")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("ratio dy/dx sample and dy/dx vacuum")
    plt.legend()
    plt.xlim(dE1_min_avg/2, dE1_min_avg*3)
    plt.ylim(-1,2)
    plt.show()
    print("please review the two auxillary plots on the derivatives of the EEL spectra. \n"+\
          "dE1 is the point before which the influence of the sample on the spectra is negligiable.") #TODO: check spelling
    return user_check("dE1", dE1_min_avg)

        
def determine_dE2(image, vacuum_cluster, nbins, dE1, check_with_user=True):
    x_bins = np.linspace(image.deltaE.min(),image.deltaE.max(), nbins)
    [y_0_bins, sigma_0_bins], edges = binned_statistics(image.deltaE, vacuum_cluster, nbins, ["mean", "var"])
    ratio_0_std = np.divide(y_0_bins,sigma_0_bins)
    #ratio_0_var = np.divide(y_0_bins,np.power(sigma_0_bins,2))
    
    dE2 = np.min(x_bins[(x_bins>dE1) * (ratio_0_std <1)])
    
    if not check_with_user:
        return dE2
    
    plt.plot(x_bins, ratio_0_std)
    plt.title("I_vacuum_bins/std_vacuum_bins")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("ratio")
    plt.plot([x_bins[0], x_bins[-1]],[1,1])
    plt.show()
    """
    plt.plot(x_bins, ratio_0_var)
    plt.title("I_vacuum_bins/var_vacuum_bins")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("ratio")
    plt.plot([x_bins[0], x_bins[-1]],[1,1])
    plt.show()
    """
    print("please review the auxillary plot on the ratio between the variance and the amplitude of "\
          +"the intensity of the vacuum EEL spectra. \n"+\
          "dE2 is the point after which the influence of the ZLP on the spectra is negligiable.") #TODO: check spelling
    return user_check("dE2", dE2)


def user_check(dE12, value):
    #TODO: opschonen?
    ans = input("Are you happy with a " + dE12 + " of " + str(round(value, 4)) + "? [y/n/wanted "+dE12+"] \n")
    if ans[0] not in["y", "n","0","1","2","3","4","5","6","7","8","9"]:
        ans = input("Please respond with either 'yes', 'no', or your wanted " + dE12 + ", otherwise assumed yes: \n")
    if ans[0] not in["y", "n","0","1","2","3","4","5","6","7","8","9"]:
        print("Stupid, assumed yes, using " + dE12 + " of " + str(round(value, 4)))
        return value
    elif ans[0] == 'y':
        print("Perfect, using " + dE12 + " of " + str(round(value, 4)))
        return value
    elif ans[0] == 'n':
        ans = input("Please input your desired " + dE12 + ": \n")
    if ans[0] not in["0","1","2","3","4","5","6","7","8","9"]:
        ans = input("Last chance, input your desired " + dE12 + ": \n")
    if ans[0] not in["0","1","2","3","4","5","6","7","8","9"]:
        print("Stupid, using old " + dE12 + " of " + str(round(value, 4)))
        return value
    else: 
        try:
            return (float(ans))
        except:
            print("input was invalid number, using original " + dE12)
            return value

#ZLP FITTING





