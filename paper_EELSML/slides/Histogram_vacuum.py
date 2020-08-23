import numpy as np
import math
import scipy
from scipy import optimize
import matplotlib
from matplotlib import gridspec
from  matplotlib import rc
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd

########## Load data ####################################
chi_data = pd.read_csv('data/chi_data')

####################
fig, ax = plt.subplots((1))
fig.set_size_inches((8,5))

plt.title('$\chi^2$ distribution for 500 replicas', fontsize=16)
plt.xlabel('$\chi^2$', fontsize=16)
plt.xlim([0, 3])

plt.hist(chi_data['Best chi2 value'], density=True, bins=90, range=[0, 4], alpha=.8, label='training')
plt.hist(chi_data['val'], density=True, bins=90, range=[0, 4], alpha=.4, label='validation')
plt.legend(fontsize=14)
#plt.ylabel('Occurence', fontsize=16)
ax.tick_params(which='major',direction='in',length=10, labelsize=14)
plt.savefig('Histogram.pdf')

plt.show()