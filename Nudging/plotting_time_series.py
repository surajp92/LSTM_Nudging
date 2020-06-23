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
data = np.load('data_20_50.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_8 = data['utrue']
uobs_8 = data['uobs']
uw_8 = data['uw']
ua_8 = data['ua']

data = np.load('data_20_100.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_12 = data['utrue']
uobs_12 = data['uobs']
uw_12 = data['uw']
ua_12 = data['ua']

data = np.load('data_20_200.npz')
t = data['t']
tobs = data['tobs']
T = data['T']
X = data['X']
utrue_20 = data['utrue']
uobs_20 = data['uobs']
uw_20 = data['uw']
ua_20 = data['ua']


#%%
fig, ax = plt.subplots(3,3,sharex=True,figsize=(10,4))
ymin = -12
ymax = 14
n = [9,20,38]

c = 0
for i in range(3):
    ax[i,c].plot(t,utrue_8[n[i],:],'k-')
    ax[i,c].plot(t,uw_8[n[i],:],'b--')
    ax[i,c].plot(t,ua_8[n[i],:],'g-.')
    if i == 0:
        ax[i,c].plot(tobs,uobs_8[n[i],:],'ro',fillstyle='none', markersize=4,markeredgewidth=1,zorder=0)
    

    ax[i,c].set_xlim([0,np.max(t)])
    ax[i,c].set_ylim([ymin,ymax])
    ax[i,c].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')
ax[i,c].set_xlabel(r'$t$')

line_labels = ['True','Erroneous','Nudging','Observations']
plt.figlegend(line_labels, loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)

c = 1
for i in range(3):
    ax[i,c].plot(t,utrue_12[n[i],:],'k-')
    ax[i,c].plot(t,uw_12[n[i],:],'b--')
    ax[i,c].plot(t,ua_12[n[i],:],'g-.')
    if i == 0:
        ax[i,c].plot(tobs,uobs_12[n[i],:],'ro',fillstyle='none', markersize=4,markeredgewidth=1,zorder=0)
    
    ax[i,c].set_xlim([0,np.max(t)])
    ax[i,c].set_ylim([ymin,ymax])
    ax[i,c].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')
ax[i,c].set_xlabel(r'$t$')

c = 2
for i in range(3):
    ax[i,c].plot(t,utrue_20[n[i],:],'k-')
    ax[i,c].plot(t,uw_20[n[i],:],'b--')
    ax[i,c].plot(t,ua_20[n[i],:],'g-.')
    if i == 0:
        ax[i,c].plot(tobs,uobs_20[n[i],:],'ro',fillstyle='none', markersize=4,markeredgewidth=1,zorder=0)

    ax[i,c].set_xlim([0,np.max(t)])
    ax[i,c].set_ylim([ymin,ymax])
    ax[i,c].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')
ax[i,c].set_xlabel(r'$t$')


fig.tight_layout()
plt.show() 
fig.savefig('time_series_nudging_tau.pdf',bbox_inches='tight')
fig.savefig('time_series_nudging_tau.eps',bbox_inches='tight')
fig.savefig('time_series_nudging_tau.png',bbox_inches='tight',dpi=300)