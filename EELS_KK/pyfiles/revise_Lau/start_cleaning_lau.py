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

path_vacuum = "data/vacuum"
path_sample = "data/sample"