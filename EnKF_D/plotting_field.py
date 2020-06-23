#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:17:22 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker

font = {'family' : 'Times New Roman',
        'size'   : 12}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
data = np.load('data_4.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_8 = data['utrue']
uobs_8 = data['uobs']
uw_8 = data['uw']
ua_8 = data['ua']

data = np.load('data_8.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_12 = data['utrue']
uobs_12 = data['uobs']
uw_12 = data['uw']
ua_12 = data['ua']

data = np.load('data_20.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_20 = data['utrue']
uobs_20 = data['uobs']
uw_20 = data['uw']
ua_20 = data['ua']


diff_8 = utrue_8 - ua_8
diff_12 = utrue_12 - ua_12
diff_20 = utrue_20 - ua_20

#%%
vmin = -10
vmax = 10
fig, ax = plt.subplots(3,3,figsize=(12,7.5))

axs = ax.flat

field = [utrue_8,utrue_12,utrue_20, ua_8,ua_12,ua_20, diff_8,diff_12,diff_20]
label = ['True','True','True','DEnKF','DEnKF','DEnKF','Error','Error','Error']


for i in range(9):
    cs = axs[i].contourf(T,X,field[i],60,cmap='coolwarm',vmin=vmin,vmax=vmax,zorder=-9)
    axs[i].set_rasterization_zorder(-1)
    axs[i].set_title(label[i])
    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$u$')
    for c in cs.collections:
        c.set_edgecolor("face")

m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(utrue_8)
m.set_clim(vmin, vmax)
#fig.colorbar(m,ax=axs[0],ticks=np.linspace(vmin, vmax, 6))

fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.25, -0.02, 0.5, 0.025])
fig.colorbar(m, cax=cbar_ax,orientation='horizontal')

fig.tight_layout()
plt.show() 
fig.savefig('field_plot_denkf.pdf',bbox_inches='tight')
fig.savefig('field_plot_denkf.eps',bbox_inches='tight')
fig.savefig('field_plot_denkf.png',bbox_inches='tight',dpi=300)

