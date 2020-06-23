#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:15:00 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(22)
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
def rhs(ne,u,fr):
    v = np.zeros(ne+3)
    v[2:ne+2] = u
    v[1] = v[ne+1]
    v[0] = v[ne]
    v[ne+2] = v[2]
    
    r = np.zeros(ne)
    
#    for i in range(2,ne+2):
#        r[i-2] = v[i-1]*(v[i+1] - v[i-2]) - v[i] + fr
    
    r = v[1:ne+1]*(v[3:ne+3] - v[0:ne]) - v[2:ne+2] + fr
    
    return r

#def rhs(ne,u,fr):

#    r = np.zeros(ne)
#    r[0] = u[ne-1]*(u[1] - u[ne-2]) - u[0] + fr
#    r[1] = u[0]*(u[2] - u[ne-1]) - u[1] + fr
#    for i in range(2,ne-1):
#        r[i] = u[i-1]*(u[i+1] - u[i-2]) - u[i] + fr
#    r[ne-1] = u[ne-2]*(u[0] - u[ne-3]) - u[ne-1] + fr
#    r = v[1:ne+1]*(v[3:ne+3] - v[0:ne]) - v[2:ne+2] + fr
#    return r

def jacrhs(ne,u):
    #first: set zero elements
    jf = np.zeros((ne,ne))
    
    #second: set non-zero elements with periodic bc (it is sparse)
    jf[0,0] = -1.0
    jf[0,1] = u[ne-1]
    jf[0,ne-2] = -u[ne-1]
    jf[0,ne-2] = -u[ne-2]
    
    jf[1,0] = u[2] - u[ne-1]
    jf[1,1] = -1.0
    jf[1,2] = u[0]
    jf[1,ne-1] = -u[0]
    
    jf[ne-1,0] = u[ne-2]
    jf[ne-1,ne-3] = -u[ne-2]
    jf[ne-1,ne-2] = u[0] - u[ne-3]
    jf[ne-1,ne-1] = -1.0
    
    #third: set non-zero elements for internal points (it is sparse)
    for i in range(2,ne-1):
        jf[i,i-2] = -u[i-1]
        jf[i,i-1] = u[i+1] - u[i-2]
        jf[i,i] = -1.0
        jf[i,i+1] = u[i-1]
    
    return jf

def jacrks4(ne,u,dt):
    jf = jacrhs(ne,u)
    jf2 = jf @ jf
    jf3 = jf2 @ jf
    jf4 = jf3 @ jf
    
    dm = np.eye(ne) + dt*jf + 0.5*dt**2*jf2 + (1.0/6.0)*(dt**3)*jf3 + (1.0/24.0)*(dt**4)*jf4
    
    return dm
    
def rk4(ne,dt,u,fr):
    r1 = rhs(ne,u,fr)
    k1 = dt*r1
    
    r2 = rhs(ne,u+0.5*k1,fr)
    k2 = dt*r2
    
    r3 = rhs(ne,u+0.5*k2,fr)
    k3 = dt*r3
    
    r4 = rhs(ne,u+k3,fr)
    k4 = dt*r4
    
    un = u + (k1 + 2.0*(k2 + k3) + k4)/6.0
    
    return un
       
#%%
ne = 40

dt = 0.005
tmax = 10.0
tini = 5.0
ns = int(tini/dt)
nt = int(tmax/dt)
fr = 10.0
nf = 10         # frequency of observation
nb = int(nt/nf) # number of observation time

u = np.zeros(ne)
utrue = np.zeros((ne,nt+1))
uinit = np.zeros((ne,ns+1))

#-----------------------------------------------------------------------------#
# generate true solution trajectory
#-----------------------------------------------------------------------------#
ti = np.linspace(-tini,0,ns+1)
t = np.linspace(0,tmax,nt+1)
tobs = np.linspace(0,tmax,nb+1)
x = np.linspace(1,ne,ne)

X,T = np.meshgrid(x,t,indexing='ij')
Xi,Ti = np.meshgrid(x,ti,indexing='ij')

u[:] = fr
u[int(ne/2)-1] = fr + 0.01
uinit[:,0] = u

# generate initial condition at t = 0
for k in range(1,ns+1):
    un = rk4(ne,dt,u,fr)
    uinit[:,k] = un
    u = np.copy(un)

# assign inital condition
u = uinit[:,-1]
utrue[:,0] = uinit[:,-1]

# generate true forward solution
for k in range(1,nt+1):
    un = rk4(ne,dt,u,fr)
    utrue[:,k] = un
    u = np.copy(un)

#%%
vmin = -12
vmax = 12
fig, ax = plt.subplots(2,1,figsize=(6,4))
cs = ax[0].contourf(Ti,Xi,uinit,120,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(uinit)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

cs = ax[1].contourf(T,X,utrue,120,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))

fig.tight_layout()
plt.show()

#%%
#-----------------------------------------------------------------------------#
# generate observations
#-----------------------------------------------------------------------------#
mean = 0.0
sd2 = 1.0e-2 # added observation noise (variance)
sd1 = np.sqrt(sd2) # added noise (standard deviation)

oib = [nf*k for k in range(nb+1)]

uobs = utrue[:,oib] + np.random.normal(mean,sd1,[ne,nb+1])

#-----------------------------------------------------------------------------#
# generate erroneous soltions trajectory
#-----------------------------------------------------------------------------#
uw = np.zeros((ne,nt+1))
k = 0
si2 = 1.0e-2
si1 = np.sqrt(si2)

u = utrue[:,0] + np.random.normal(mean,si1,ne)
uw[:,0] = u

for k in range(1,nt+1):
    un = rk4(ne,dt,u,fr)
    uw[:,k] = un
    u = np.copy(un)

#%%
#-----------------------------------------------------------------------------#
# EnKF model
#-----------------------------------------------------------------------------#    

# number of observation vector
me = 20
freq = int(ne/me)
oin = [freq*i-1 for i in range(1,me+1)]
roin = np.int32(np.linspace(0,me-1,me))
print(oin)

#%%
z = np.zeros((me,nb+1))

ua = np.zeros((ne,nt+1)) # analysis solution (to store)
pa = np.zeros((ne,ne)) # analysis covariance
pf = np.zeros((ne,ne)) # forecast covariance
uf = np.zeros(ne)        # forecast

km = np.zeros((ne,me))

z[:,:] = uobs[oin,:]

# initial ensemble
k = 0
se2 = 1e-4 #np.sqrt(sd2)
se1 = np.sqrt(se2)

Q = se2 * np.eye(ne)
R = sd2 * np.eye(me)

u = uw[:,k] #+ np.random.normal(mean,si1,ne)
ua[:,k] = u

# initial estimate of the analysis covariance
pa = si2 * np.eye(ne)

dh = np.zeros((me,ne))
dh[roin,oin] = 1.0

#%%
kobs = 1

# RK4 scheme
for k in range(1,nt+1):
    
    # forecast field
    uf = rk4(ne,dt,u,fr)
    
    dm = jacrks4(ne,u,dt)
        
    # forecast covariance
    pf = dm @ pa @ dm.T + Q
    pf = 0.5*(pf + pf.T)
    pa = np.copy(pf)
    
    # perform analysis at observation points
    if k == oib[kobs]:        
        dm = jacrks4(ne,u,dt)
        
        # forecast covariance
#        pf = dm @ pa @ dm.T + Q
#        
#        pf = 0.5*(pf + pf.T)
        
        cc = dh @ pf @ dh.T + R
        ci = np.linalg.pinv(cc)
        
        cc_ch = np.linalg.cholesky(cc)
        
        ph = pf @ dh.T
        
        # compute Kalman gain
        km = ph @ ci
        
        # analysis update    
        kmd = km @ (z[:,kobs] - uf[oin])
        uf = uf + kmd
        
        # compute analysis covaraince
        kd = np.eye(ne) - km @ dh
        pa = kd @ pf
        
        kobs = kobs+1
    
    # mean analysis for plotting
    ua[:,k] = uf
    u = np.copy(uf)    

#%%
np.savez('data_'+str(me)+'.npz',t=t,tobs=tobs,T=T,X=X,utrue=utrue,uobs=uobs,uw=uw,ua=ua,oin=oin)

#%%
fig, ax = plt.subplots(3,1,sharex=True,figsize=(6,5))

n = [16,29,36]
for i in range(3):
    if i == 1:
        ax[i].plot(tobs,uobs[n[i],:],'ro',fillstyle='none', markersize=6,markeredgewidth=2)
    ax[i].plot(t,utrue[n[i],:],'k-')
    ax[i].plot(t,uw[n[i],:],'b--')
    ax[i].plot(t,ua[n[i],:],'g-.')
    

    ax[i].set_xlim([0,tmax])
    ax[i].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')

ax[i].set_xlabel(r'$t$')
line_labels = ['True','Wrong','EKF','Observation']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()
plt.show() 
fig.savefig('m_'+str(me)+'.pdf')

#%%
vmin = -10
vmax = 10
fig, ax = plt.subplots(3,1,figsize=(6,7.5))

cs = ax[0].contourf(T,X,utrue,30,cmap='coolwarm',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))
ax[0].set_title('True')

cs = ax[1].contourf(T,X,ua,30,cmap='coolwarm',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(ua)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))
ax[1].set_title('EKF')

diff = ua - utrue
cs = ax[2].contourf(T,X,diff,30,cmap='coolwarm',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(diff)
#m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[2])#,ticks=np.linspace(vmin, vmax, 6))
ax[2].set_title('Difference')

fig.tight_layout()
plt.show() 
fig.savefig('f1_'+str(me)+'.pdf')    

#%%
print(np.linalg.norm(diff))
    
























































