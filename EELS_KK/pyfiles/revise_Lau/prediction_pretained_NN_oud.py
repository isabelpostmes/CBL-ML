#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 01:05:56 2020

@author: isabel
"""
print('Importing packages...')

import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib

import tensorflow as tf 
import tensorflow.compat.v1 as tf
from tensorflow import keras

from matplotlib import rc
from copy import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import fnmatch
import csv
import pickle
import warnings
import scipy
from scipy import optimize
from scipy.optimize import leastsq
from datetime import datetime
from matplotlib import cm
#from lmfit import Model
from scipy.optimize import curve_fit

print('done')


from Functions import *
from Load_data import *

tf.reset_default_graph()
cols=['y14', 'x14', 'y15', 'x15', 'y16', 'x16', 'y17', 'x17', 'y19', 'x19', 'y20', 'x20', 'y21', 'x21', 'y22', 'x22', 'y23', 'x23']

ZLP_data = pd.concat((file14, file15, file16, file17, file19, file20, file21, file22, file23), axis=1)
ZLP_data = ZLP_data.drop(['x', 'y_norm'],axis=1).rename(columns={'x_shifted': 'x'})
ZLP_data.columns = cols

print(ZLP_data)

## Window the data file to the desired energy range
E_min = -.3
#CHANGE
E_min= -4
E_max = 20
original = ZLP_data[(ZLP_data['x14'] >= E_min) & (ZLP_data['x14'] <= E_max)]



d_string = '07.09.2020'

path_to_data = 'Data_oud/Results/%(date)s/'% {"date": d_string} 

path_predict = r'Predictions_*.csv'
path_cost = r'Cost_*.csv' 

all_files = glob.glob(path_to_data + path_predict)

li = []
for filename in all_files:
    df = pd.read_csv(filename, delimiter=",",  header=0, usecols=[0,1,2], names=['x', 'y', 'pred'])
    li.append(df)
    

training_data = pd.concat(li, axis=0, ignore_index=True)



all_files_cost = glob.glob(path_to_data + path_cost)


import natsort

all_files_cost_sorted = natsort.natsorted(all_files_cost)

chi2_array = []
chi2_index = []

for filename in all_files_cost_sorted:
    df = pd.read_csv(filename, delimiter=",", header=0, usecols=[0,1], names=['train', 'test'])
    best_try = np.argmin(df['test'])
    chi2_array.append(df.iloc[best_try,0])
    chi2_index.append(best_try)

chi_data  = pd.DataFrame()
chi_data['Best chi2 value'] = chi2_array
chi_data['Epoch'] = chi2_index
    
print("total length of files:", len(chi2_array))


good_files = []
count = 0
threshold = 3

for i,j in enumerate(chi2_array):
    if j < threshold:
        good_files.append(1) 
        count +=1 
    else:
        good_files.append(0)

print("Setting the threshold at", threshold, ", the number of files that survived the selection is", count)




tf.get_default_graph
tf.disable_eager_execution()

def make_model(inputs, n_outputs):
    hidden_layer_1 = tf.layers.dense(inputs, 10, activation=tf.nn.sigmoid)
    hidden_layer_2 = tf.layers.dense(hidden_layer_1, 15, activation=tf.nn.sigmoid)
    hidden_layer_3 = tf.layers.dense(hidden_layer_2, 5, activation=tf.nn.relu)
    output = tf.layers.dense(hidden_layer_3, n_outputs, name='outputs', reuse=tf.AUTO_REUSE)
    return output

x = tf.placeholder("float", [None, 1], name="x")
predictions = make_model(x, 1)


d_string = '07.09.2020'
prediction_file = pd.DataFrame()

predict_x = np.linspace(-0.5, 20, 1000).reshape(1000,1)
predict_x = ZLP_data.x14.values.reshape(len(ZLP_data.x14),1)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(0,len(good_files)):
        if good_files[i] == 1:
            
            best_model = 'Models_oud/Best_models/%(s)s/best_model_%(i)s'% {'s': d_string, 'i': i}
            saver = tf.train.Saver(max_to_keep=1000)
            saver.restore(sess, best_model)

            extrapolation = sess.run(predictions,
                                    feed_dict={
                                    x: predict_x
                                    })
            #prediction_file['prediction_%(i)s' % {"i": i}] = extrapolation.reshape(1000,)
            prediction_file['prediction_%(i)s' % {"i": i}] = extrapolation.reshape(len(ZLP_data.x14),)


dE1 = np.round(max(training_data['x'][(training_data['x']< 3)]),2)
dE2 = np.round(min(training_data['x'][(training_data['x']> 3)]),1)
dE0 = np.round(dE1 - .5, 2) 

print('The values for dE0, dE1 and dE2:', dE0, dE1, dE2)

### Definition for the matching procedure

def matching(x, y_NN, y_ZLP):
    
    total = pd.DataFrame({"x": x, "prediction y": y_NN, "data ZLP": y_ZLP})
    
    delta = np.divide((dE1 - dE0), 3)

    factor_NN = np.exp(- np.divide((x[(x<dE1) & (x >= dE0)] - dE1)**2, delta**2))
    factor_ZLP = 1 - factor_NN
    
    range_0 = total[total['x'] < dE0]['data ZLP'] * 1
    range_1 = total[(total['x'] < dE1) & (total['x'] >= dE0)]['prediction y'] * factor_NN + total[(total['x'] < dE1) & (total['x'] >= dE0)]['data ZLP'] * factor_ZLP
    range_2 = total[(total['x'] >= dE1) & (total['x'] < 3 * dE2)]['prediction y'] * 1 
    range_3 = total[(total['x'] >= 3 * dE2)]['prediction y'] * 0
    totalfile = np.concatenate((range_0, range_1, range_2, range_3), axis=0)
    
    return totalfile




nbins = len(original['x14'])
li = []
diff = []

for i in range(0, len(prediction_file.columns)):
    df = pd.DataFrame()
    #df['x'] = predict_x.reshape(1000,)
    df['x'] = predict_x.reshape(len(ZLP_data.x14),)
    df['prediction'] = prediction_file.iloc[:,i]
    df['k'] = i
    li.append(df)

extrapolation = pd.concat(li, axis=0, ignore_index = True)

### Window the prediction data to the same energy range as the original spectra
    
extrapolation = extrapolation[(extrapolation['x'] >= E_min) & (extrapolation['x'] <= E_max)]




lo = []

for k in range(count): 
    exp_k = extrapolation[extrapolation['k'] == k ]
    nbins = len(original['x14'])  
    mean_k, var_k, count = binned_statistics(exp_k['x'], exp_k['prediction'], nbins)[0:3]
    
    replica_file = pd.DataFrame({"k": k, \
                                 "x14": original['x14'], \
                                 "x15": original['x15'], \
                                 "x16": original['x16'], \
                                 "x19": original['x19'], \
                                 "x20": original['x20'], \
                                 "x21": original['x21'],\
                                 #
                               "prediction log(y1)": mean_k, \
                               "prediction y": np.exp(mean_k), \
                                 #
                               "data y14": original['y14'], \
                               "data y15": original['y15'], \
                               "data y16": original['y16'], \
                               "data y19": original['y19'], \
                               "data y20": original['y20'], \
                               "data y21": original['y21'], \
                                #
                                     "match14": matching(original['x14'], np.exp(mean_k), original['y14']), \
                                     "match15": matching(original['x15'], np.exp(mean_k), original['y15']), \
                                     "match16": matching(original['x16'], np.exp(mean_k), original['y16']), \
                                     "match19": matching(original['x19'], np.exp(mean_k), original['y19']), \
                                     "match20": matching(original['x20'], np.exp(mean_k), original['y20']), \
                                     "match21": matching(original['x21'], np.exp(mean_k), original['y21']), })
    lo.append(replica_file)
    
total_replicas = pd.concat(lo, axis=0, ignore_index = True, sort=False)

### Subtracted spectra:  difference = original spectrum - matched spectrum

for i in ([14, 15, 16, 19, 20, 21]):
    total_replicas['dif%(i)s'%{"i": i}] = total_replicas['data y%(i)s'%{"i": i}] - total_replicas['match%(i)s'%{"i": i}]

#total_replicas.to_csv('Data/Results/Replica_files/final_%(s)s' % {"s": dE1})


















