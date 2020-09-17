#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:47:40 2020

@author: isabel
"""

###############################################
import numpy as np
from scipy import integrate
import pandas as pd
import os



def load_spectra(path, rg):
    """
    INPUT:
        path: path to directory with spectra, please make sure these are the 
                only files in this directory
        rg: [Eloss_min, Eloss_max], range over which the spectra are measured, 
                in eV
                
    OUTPUT:
        df_spectra: pandas data frame with all vacuum spectra
    
    This function loads all the spectra in the input directories.
    """
    #TODO: add possibility for different ranges??
    #delta_dE = (range_vacuum[1]-range_vacuum[0])/len(spectra_vacuum[0])
    
    
    #TODO: change x & y to more explainatory values?
    df_spectra = pd.DataFrame(columns = ['x', 'y'])
    for filename in os.listdir(path):
        #DO STUFF
        if filename.endswith(".txt"):
            spectrum = np.loadtxt(path + '/' + filename)
            dE = np.linspace(rg[0], rg[1], len(spectrum))
            
            df_spectra = df_spectra.append({'x': dE, 'y': spectrum}, ignore_index = True)
    
    
    return df_spectra
    



def load_data(path_vacuum, path_sample, range_vacuum, range_sample = False, pr = False):
    """
    INPUT:
        path_vacuum: path to directory with all available vacuum spectra, please 
                make sure these are the only files in this directory
        path_sample: path to directory with all available sample spectra, please 
                make sure these are the only files in this directory
        range_vacuum: [Eloss_min, Eloss_max], range over which the vacuum 
                spectra are measured, in eV
        range_sample: [Eloss_min, Eloss_max], default False, range over which 
                the sample spectra are measured, in eV. If undefined, the same 
                range as the vacuum is used
        pr: default = False, boolean whether you want a summary of loaded files
                printed
    
    OUTPUT:
        df_vacuum: pandas DataFrame with columns ('x','y','x_shifted','y_norm') 
                with all vacuum spectra
        df_sample: pandas DataFrame with columns ('x','y','x_shifted','y_norm')
                with all sample spectra
        
    This function loads and prepares all the spectra in the two input directories
    for elvaluation. 
    """
    
    if not range_sample:
        range_sample = range_vacuum
    
    spectra_vacuum = load_spectra(path_vacuum, range_vacuum)
    spectra_sample = load_spectra(path_sample, range_sample)
    
    
    spectra_vacuum = shift_norm(spectra_vacuum)
    spectra_sample = shift_norm(spectra_sample)
    
    spectra_vacuum = spectra_vacuum.dropna()
    spectra_sample = spectra_sample.dropna()
    
    #TODO: figure out what these lines should do
    #spectra_vacuum = spectra_vacuum.sort_values('x').reset_index().drop('index', axis=1)#.dropna()
    #spectra_sample = spectra_sample.sort_values('x').reset_index().drop('index', axis=1)

    #TODO evaluate if this column is needed
    #spectra_sample['log_y'] = np.log(spectra_sample['y'])
    
    
    
    if(pr):
        print('\n Total samples file: "df" \n', spectra_sample.describe())
        print('\n Total vacuum file: "df_vacuum" \n', spectra_vacuum.describe())
    
    return spectra_vacuum, spectra_sample



def shift_norm(df_spectra):
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
    
    df_sn = pd.DataFrame(columns = ['x', 'y', 'x_shifted', 'y_norm'])
    
    for i in df_spectra.index:
        
        
        print(df_spectra.iloc[i].x)
        
        
        df_sn = df_sn.append(df_spectra.iloc[i])
        i_max_y = np.argmax(df_spectra.iloc[i].y)
        zeropoint = df_spectra.iloc[i].x[i_max_y]
        x_shifted = df_spectra.iloc[i].x - float(zeropoint)
        
        y = df_spectra.iloc[i].y
        y_int = integrate.cumtrapz(y, x_shifted, initial=0)
        normalization = y_int[-1]
        
        
        df_sn.at[i, 'x_shifted'] = x_shifted
        df_sn.at[i, 'y_norm'] = y/float(normalization)
    

    return df_sn


#TODO: delete below
"""
LAURIENS VERISON:
###### Load spectra ###########################

fname = 'data/Specimen_4/'
    
ZLP_y14 = np.loadtxt(fname + "(14)_m4d054eV_45d471eV.txt")
ZLP_y15 = np.loadtxt(fname + "(15)_m4d054eV_45d471eV.txt")
ZLP_y16 = np.loadtxt(fname + "(16)_m4d054eV_45d471eV.txt")
ZLP_y17 = np.loadtxt(fname + "(17)_Vacuum_m4d054eV_45d471eV.txt")
ZLP_y19 = np.loadtxt(fname + "(19)_m4d054eV_45d471eV.txt")
ZLP_y20 = np.loadtxt(fname + "(20)_m4d054eV_45d471eV.txt")
ZLP_y21 = np.loadtxt(fname + "(21)_m4d054eV_45d471eV.txt")
ZLP_y22 = np.loadtxt(fname + "(22)_Vacuum_m4d054eV_45d471eV.txt")
ZLP_y23 = np.loadtxt(fname + "(23)_Vacuum_m4d054eV_45d471eV.txt")

###############################################

ndat=int(len(ZLP_y14))

# Energy loss values
ZLP_x14 = np.zeros(ndat)
Eloss_min = -4.054 # eV
Eloss_max = +45.471 # eV
i=0
while(i<ndat):
    ZLP_x14[i]= Eloss_min +\
        ( Eloss_max - Eloss_min)*((i+0.5)*1.0/ndat) 
    i = i + 1

ZLP_x15, ZLP_x16, ZLP_x17, ZLP_x19, ZLP_x20, ZLP_x21, ZLP_x22, ZLP_x23 = \
    ZLP_x14,  ZLP_x14,  ZLP_x14,  ZLP_x14,  ZLP_x14, ZLP_x14,  ZLP_x14, ZLP_x14




file14 = pd.DataFrame({"x":ZLP_x14, "y":ZLP_y14})
file15 = pd.DataFrame({"x":ZLP_x15, "y":ZLP_y15})
file16 = pd.DataFrame({"x":ZLP_x16, "y":ZLP_y16})
file17 = pd.DataFrame({"x":ZLP_x17, "y":ZLP_y17})
file19 = pd.DataFrame({"x":ZLP_x19, "y":ZLP_y19})
file20 = pd.DataFrame({"x":ZLP_x20, "y":ZLP_y20})
file21 = pd.DataFrame({"x":ZLP_x21, "y":ZLP_y21})
file22 = pd.DataFrame({"x":ZLP_x22, "y":ZLP_y22})
file23 = pd.DataFrame({"x":ZLP_x23, "y":ZLP_y23})


################## Shift spectra to have peak position at dE = 0  ##################

for i, file in enumerate([file14, file15, file16, file17, file19, file20, file21, file22, file23]):
    zeropoint = file[file['y'] == file['y'].max()]['x']
    file['x_shifted'] = file['x'] - float(zeropoint)
    x = file['x_shifted']
    y = file['y']
    y_int = integrate.cumtrapz(y, x, initial=0)
    normalization = y_int[-1]
    file['y_norm'] = file['y'] / float(normalization)
    
    
    
##############  Put all datafiles into one DataFrame  ##############################

df = pd.concat((file14, file15, file16, file19, file20, file21))
df = df.sort_values('x').reset_index().drop('index', axis=1)

df_vacuum = pd.concat((file17, file22, file23))
df_vacuum = df_vacuum.sort_values('x').reset_index().drop('index', axis=1).dropna()

df['log_y'] = np.log(df['y'])
#df_vacuum['log_y'] = np.log(df_vacuum['y'])



################ Use [x_shifted, y_norm] values as training inputs #################


x14, y14 = file14['x_shifted'], file14['y_norm']
x15, y15 = file15['x_shifted'], file15['y_norm']
x16, y16 = file16['x_shifted'], file16['y_norm']
x17, y17 = file17['x_shifted'], file17['y_norm']
x19, y19 = file19['x_shifted'], file19['y_norm']
x20, y20 = file20['x_shifted'], file20['y_norm']
x21, y21 = file21['x_shifted'], file21['y_norm']
x22, y22 = file22['x_shifted'], file22['y_norm']
x23, y23 = file23['x_shifted'], file23['y_norm']



print('Files have been created \n')

print('\n Sample files:')
for i in ([14,15,16,19,20,21]):
    print('file' + str(i))
print('\n Vacuum files:')
for i in ([17,22,23]):
    print('file' + str(i))
    
    
print('\n Total samples file: "df" \n', df.describe())
print('\n Total vacuum file: "df_vacuum" \n', df_vacuum.describe())

"""