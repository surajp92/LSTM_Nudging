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
data = np.load('data_2.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_8 = data['utrue']
uobs_8 = data['uobs']
uw_8 = data['uw']
ua_8 = data['ulstm2']

data = np.load('data_3.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_12 = data['utrue']
uobs_12 = data['uobs']
uw_12 = data['uw']
ua_12 = data['ulstm2']

data = np.load('data_4.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_20 = data['utrue']
uobs_20 = data['uobs']
uw_20 = data['uw']
ua_20 = data['ulstm2']

data = np.load('data_2e.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_8e = data['utrue']
uobs_8e = data['uobs']
uw_8e = data['uw']
ua_8e = data['ua']

data = np.load('data_3e.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_12e = data['utrue']
uobs_12e = data['uobs']
uw_12e = data['uw']
ua_12e = data['ua']

data = np.load('data_4e.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_20e = data['utrue']
uobs_20e = data['uobs']
uw_20e = data['uw']
ua_20e = data['ua']

data = np.load('data_2d.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_8d = data['utrue']
uobs_8d = data['uobs']
uw_8d = data['uw']
ua_8d = data['ua']

data = np.load('data_3d.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_12d = data['utrue']
uobs_12d = data['uobs']
uw_12d = data['uw']
ua_12d = data['ua']

data = np.load('data_4d.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_20d = data['utrue']
uobs_20d = data['uobs']
uw_20d = data['uw']
ua_20d = data['ua']


diff_8 = utrue_8 - ua_8
diff_12 = utrue_12 - ua_12
diff_20 = utrue_20 - ua_20

diff_8e = utrue_8e - ua_8e
diff_12e = utrue_12e - ua_12e
diff_20e = utrue_20e - ua_20e

diff_8d = utrue_8d - ua_8d
diff_12d = utrue_12d - ua_12d
diff_20d = utrue_20d - ua_20d

#%%
vmin = -10
vmax = 10
fig, ax = plt.subplots(7,3,figsize=(12,15.0))

axs = ax.flat

field = [utrue_8,utrue_12,utrue_20,  
         ua_8e,ua_12e,ua_20e, 
         diff_8e,diff_12e,diff_20e, 
         ua_8d,ua_12d,ua_20d, 
         diff_8d,diff_12d,diff_20d, 
         ua_8,ua_12,ua_20, 
         diff_8,diff_12,diff_20]

label = ['True','True','True',
         'EnKF','EnKF','EnKF',
         'Error (EnKF)','Error (EnKF)','Error (EnKF)',
         'DEnKF','DEnKF','DEnKF',
         'Error (DEnKF)','Error (DEnKF)','Error (DEnKF)',
         'LSTM','LSTM','LSTM',
         'Error (LSTM)','Error (LSTM)','Error (LSTM)']


for i in range(21):
    cs = axs[i].contourf(T,X,field[i],60,cmap='coolwarm',vmin=vmin,vmax=vmax,zorder=-9)
    axs[i].set_rasterization_zorder(-1)
    
    axs[i].set_title(label[i])
#    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$u$')
    for c in cs.collections:
        c.set_edgecolor("face")

axs[18].set_xlabel(r'$t$')
axs[19].set_xlabel(r'$t$')
axs[20].set_xlabel(r'$t$')

m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(utrue_8)
m.set_clim(vmin, vmax)
#fig.colorbar(m,ax=axs[0],ticks=np.linspace(vmin, vmax, 6))

fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.25, -0.02, 0.5, 0.015])
fig.colorbar(m, cax=cbar_ax,orientation='horizontal')

fig.tight_layout()
plt.show() 


fig.savefig('field_plot_Ns_400.pdf',bbox_inches='tight')
fig.savefig('field_plot_Ns_400.eps',bbox_inches='tight')
fig.savefig('field_plot_Ns_400.png',bbox_inches='tight',dpi=300)

#%%
vmin = -10
vmax = 10
fig, ax = plt.subplots(3,3,figsize=(12,7.5))

axs = ax.flat

field = [diff_8e,diff_12e,diff_20e, 
         diff_8d,diff_12d,diff_20d, 
         diff_8,diff_12,diff_20]

label = ['Error (EnKF)','Error (EnKF)','Error (EnKF)',
         'Error (DEnKF)','Error (DEnKF)','Error (DEnKF)',
         'Error (LSTM)','Error (LSTM)','Error (LSTM)']


for i in range(9):
    cs = axs[i].contourf(T,X,field[i],60,cmap='coolwarm',vmin=vmin,vmax=vmax,zorder=-9)
    axs[i].set_rasterization_zorder(-1)
    
    axs[i].set_title(label[i])
    axs[i].set_xlabel(r'$t$')
    axs[i].set_ylabel(r'$u$')
    for c in cs.collections:
        c.set_edgecolor("face")

#axs[18].set_xlabel(r'$t$')
#axs[19].set_xlabel(r'$t$')
#axs[20].set_xlabel(r'$t$')

m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(utrue_8)
m.set_clim(vmin, vmax)
#fig.colorbar(m,ax=axs[0],ticks=np.linspace(vmin, vmax, 6))

fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.25, -0.02, 0.5, 0.025])
fig.colorbar(m, cax=cbar_ax,orientation='horizontal')

fig.tight_layout()
plt.show() 


fig.savefig('error_plot_Ns_400.pdf',bbox_inches='tight')
fig.savefig('error_plot_Ns_400.eps',bbox_inches='tight')
fig.savefig('error_plot_Ns_400.png',bbox_inches='tight',dpi=300)

#%%
#my_array = np.random.rand(500,500)
#
#fig, ax = plt.subplots(2,1,figsize=(6,6))
#ax[0].contourf(my_array)
#ax[1].contourf(my_array)
#plt.savefig('contourf1.pdf')
#plt.close()
#
#
#fig, ax = plt.subplots(2,1,figsize=(6,6))
#ax[0].contourf(my_array,zorder=-9)
#ax[1].contourf(my_array,zorder=-9)
#ax[0].set_rasterization_zorder(-1)
#ax[1].set_rasterization_zorder(-1)
#plt.savefig('contourf3.pdf')
#plt.close()
