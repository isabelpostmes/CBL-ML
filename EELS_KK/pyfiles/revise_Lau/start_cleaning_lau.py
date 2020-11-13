#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 00:13:27 2020

@author: isabel
"""
#SHOULDN'T BE NEEDED RIGHT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from matplotlib import cm
import os
import csv
import warnings
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
import tensorflow.compat.v1 as tf
from copy import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from load_data_revised import *
from functions_revised import *

warnings.filterwarnings('ignore')

#VARIABLES ESSENTIAL TO CHANGE FOR EACH DATASET
path_vacuum = "data_new/vacuum"
path_sample = "data_new/sample"

Eloss_min = -4.054 # eV
Eloss_max = +45.471 # eV

#TODO: WHERE TO DEFINE THESE VALUES, they seem less important than the rest
wl1 = 50
wl2 = 100

df_vacuum, df_sample = load_data(path_vacuum, path_sample, [Eloss_min, Eloss_max])
df_vac_der = process_data(df_vacuum)
df_sam_der = process_data(df_sample)


plot_bool = True
if plot_bool:
    plot_deriv(df_vac_der, df_sam_der)
#%%

vac_avg = np.average(df_vac_der.dy_dx[:])
sam_avg = np.average(df_sam_der.dy_dx[:])
df_vac_der['ratio'] = np.empty
df_sam_der['ratio'] = np.empty
df_vac_der['dE2'] = np.nan
df_sam_der['dE2'] = np.nan
for i in df_sam_der.index:
    df_sam_der.at[i,'ratio'] = np.divide(df_sam_der.iloc[i].dy_dx, vac_avg)
    df_sam_der.at[i,'dE1'] = df_sam_der.iloc[i].x_shifted[(df_sam_der.iloc[i]['ratio'] > 0) & (df_sam_der.iloc[i]['ratio'] < 1)].min()
#for i in df_vac_der.index:    
#    df_vac_der.at[i,'ratio'] = np.divide(df_vac_der.iloc[i].dy_dx, sam_avg)
#    df_vac_der.at[i,'dE2'] = df_vac_der.iloc[i].x_shifted[df_vac_der['ratio'] < 1].min()
dE1 = np.min(df_sam_der.dE1)
dE2 = max(df_vac_der.dE2.max(), df_sam_der.dE2.max())






















