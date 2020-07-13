# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:41:10 2019

@author: Suraj
"""
#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import simps
import pyfftw

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import pandas as pd
import time as clck
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import keras.backend as K
K.set_floatx('float64')

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
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
#-----------------------------------------------------------------------------#
# Neural network Routines
#-----------------------------------------------------------------------------#
def create_training_data_lstm(features,labels, m, n, lookback):
    # m : number of snapshots 
    # n: number of states
    ytrain = [labels[i,:] for i in range(m)]
    ytrain = np.array(ytrain)    
    
    xtrain = np.zeros((m-lookback+1,lookback,n))
    for i in range(m-lookback+1):
        a = features[i,:]
        for j in range(1,lookback):
            a = np.vstack((a,features[i+j,:]))
        xtrain[i,:,:] = a
    return xtrain , ytrain

def deploy_input(features, m, n, lookback):
    xtest = np.zeros((m-lookback+1,lookback,n))
    for i in range(m-lookback+1):
        a = features[i,:]
        for j in range(1,lookback):
            a = np.vstack((a,features[i+j,:]))
        xtest[i,:,:] = a
    return xtest 


#%% Main program:
ne = 40
npe = 400
fr = 10.0
dt = 0.005
tmax = 10.0
nt = int(tmax/dt)
lookback = 1

nf = 10         # frequency of observation
nb = int(nt/nf) # number of observation time

oib = [nf*k for k in range(nb+1)]
tobs = np.linspace(0,tmax,nb+1)

t = np.linspace(0,tmax,nt+1)
x = np.linspace(1,ne,ne)

X,T = np.meshgrid(x,t,indexing='ij')

Training = True

data = np.load('../lstm_data_sparse.npz')
utrue = data['utrue']   
uobs = data['uobs']   
uwe = data['uwe']   

uobs_lstm = np.zeros((ne,nt+1))
oib = [nf*k for k in range(nb+1)]

for p in range(nb):
    for q in range(nf):
        uobs_lstm[:,10*p+q] = uobs[:,p]

uobs_lstm[:,nt] = uobs[:,nb]

#%%
# number of observation vector
me = 4
freq = int(ne/me)
oin = [freq*i-1 for i in range(1,me+1)]

da_model = 3

if da_model == 1:
    nfeat = 2*me
elif da_model == 2:
    nfeat = ne
elif da_model == 3:
    nfeat = ne+me

for n in range(npe):
    if da_model == 1:
        features = np.hstack((uwe[oin,n,oib].T,uobs[oin,:].T))
    elif da_model == 2:
        features = uwe[:,n,:].T 
    elif da_model == 3:
        features = np.hstack((uwe[:,n,oib].T,uobs[oin,:].T)) #
    labels = utrue[:,oib].T - uwe[:,n,oib].T 
    xt, yt = create_training_data_lstm(features, labels, nb+1, nfeat, lookback)
    
    if n == 0:
        xtrain = xt
        ytrain = yt
    else:
        xtrain = np.vstack((xtrain,xt))
        ytrain = np.vstack((ytrain,yt))
    
#%%
data = xtrain # modified GP as the input data
labels = ytrain
        
#%%
# Scaling data
p,q,r = data.shape
data2d = data.reshape(p*q,r)

scalerIn = MinMaxScaler(feature_range=(-1,1))
scalerIn = scalerIn.fit(data2d)
data2d = scalerIn.transform(data2d)
data = data2d.reshape(p,q,r)

scalerOut = MinMaxScaler(feature_range=(-1,1))
scalerOut = scalerOut.fit(labels)
labels = scalerOut.transform(labels)

xtrain = data
ytrain = labels

xtrain, xvalid, ytrain, yvalid = train_test_split(data, labels, test_size=0.2 , shuffle= True)

#%%
mx,lx,nx = xtrain.shape # m is number of training samples, n is number of output features [i.e., n=nr]
my,ny = ytrain.shape
    
if Training:
    # create the LSTM architecture
    model = Sequential()
    #model.add(Dropout(0.2))
    model.add(LSTM(80, input_shape=(lookback, nx), return_sequences=True, activation='relu'))
    #model.add(LSTM(80, input_shape=(lookback, nx), return_sequences=True, activation='relu'))
    #model.add(LSTM(40, input_shape=(lookback, n+1), return_sequences=True, activation='relu', kernel_initializer='uniform'))
    model.add(LSTM(80, input_shape=(lookback, nx), activation='relu'))
    model.add(Dense(ny))
    
    # compile model
    #model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    
    # run the model
    history = model.fit(xtrain, ytrain, epochs=2500, batch_size=128, validation_data= (xvalid,yvalid))
    #history = model.fit(xtrain, ytrain, epochs=600, batch_size=32, validation_split=0.2)
    
    # evaluate the model
    scores = model.evaluate(xtrain, ytrain, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure()
    epochs = range(1, len(loss) + 1)
    plt.semilogy(epochs, loss, 'b', label='Training loss')
    plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
    plt.title('Training and validation loss')
    plt.legend()
    filename = 'loss.png'
    plt.savefig(filename, dpi = 400)
    plt.show()
    
    # Save the model
    filename = 'da_lstm_m'+str(da_model)+'_'+str(me)+'.hd5'
    model.save(filename)

mx,lx,nx = xtrain.shape # m is number of training samples, n is number of output features [i.e., n=nr]
my,ny = ytrain.shape

#%% deployment
#-----------------------------------------------------------------------------#
# generate erroneous soltions trajectory
#-----------------------------------------------------------------------------#
uw = np.zeros((ne,nt+1))
k = 0
mean = 0
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
# correct erroneous soltions trajectory with lstm static 
# (compute correction at every step)
#-----------------------------------------------------------------------------#   
filename = 'da_lstm_m'+str(da_model)+'_'+str(me)+'.hd5'
    
model = load_model(filename)
ulstm = np.zeros((ne,nt+1)) 
ulstm_c = np.zeros((ne,nt+1)) 

k = 0
ulstm[:,k] = uw[:,k]
u = ulstm[:,k]

ucorr = np.zeros((ne,nb+1)) 

xtest = np.hstack((ulstm[:,k].T,uobs[oin,k].T)).reshape(1,-1)
xtest_sc = scalerIn.transform(xtest)
xtest_sc = xtest_sc.reshape(1,lookback,nx)
ytest_sc = model.predict(xtest_sc)
ytest = scalerOut.inverse_transform(ytest_sc) # residual/ correction
ucorr[:,k] = ytest.reshape(-1,)

ytest2 = np.zeros((nt+1,ne))

ulstm_obs = np.zeros((ne,nb+1))
ulstm_obs[:,0] = u

for k in range(1,nt+1):
    un = rk4(ne,dt,u,fr)
    ulstm[:,k] = un
    
#    unc = rk4(ne,dt,uc,fr)
#    ulstm_c[:,k] = unc
    
    if k%freq == 0:
#        print(k, ' ', int(k/freq))
        xtest = np.hstack((ulstm[:,k].T,uobs[oin,int(k/nf)].T)).reshape(1,-1)
        xtest_sc = scalerIn.transform(xtest)
        xtest_sc = xtest_sc.reshape(1,lookback,nx)
        ytest_sc = model.predict(xtest_sc)
        ytest = scalerOut.inverse_transform(ytest_sc) # residual/ correction
        ytest2[k,:] = ytest
        ucorr[:,int(k/nf)] = ytest.reshape(-1,)
        
        ulstm[:,k] = un + ytest.reshape(-1,) #ucorr[:,int(k/freq)]
        #ulstm_c[:,k] = un + ytest.reshape(-1,)
        ulstm_obs[:,int(k/nf)] = un + ytest.reshape(-1,)

    u = ulstm[:,k] #np.copy(un)
    #uc = ulstm_c[:,k]
    
#ulstm_obs = np.zeros((ne,nb+1)) 
#ulstmd[:,0] = ulstm[:,0]
#ulstmd[:,1:] = ulstm[:,1:] + ucorr[:,:-1]
#
#ulstmd = ulstm + ucorr

#%%    
#-----------------------------------------------------------------------------#
# correct erroneous soltions trajectory with lstm static 
# (compute correction at every step)
#-----------------------------------------------------------------------------#   
filename = 'da_lstm_m'+str(da_model)+'_'+str(me)+'.hd5'
    
model = load_model(filename)
ulstm_v2 = np.zeros((ne,nt+1)) 
ulstm_v2_c = np.zeros((ne,nt+1)) 

k = 0
ulstm_v2[:,k] = uw[:,k]
u = ulstm_v2[:,k]

ulstm_v2_c[:,k] = uw[:,k]
uc = ulstm_v2_c[:,k]

ucorr = np.zeros((ne,nb+1)) 

xtest = np.hstack((ulstm_v2[:,k].T,uobs[oin,k].T)).reshape(1,-1)
xtest_sc = scalerIn.transform(xtest)
xtest_sc = xtest_sc.reshape(1,lookback,nx)
ytest_sc = model.predict(xtest_sc)
ytest = scalerOut.inverse_transform(ytest_sc) # residual/ correction
ucorr[:,k] = ytest.reshape(-1,)

ytest2 = np.zeros((nt+1,ne))

ulstm_v2_obs = np.zeros((ne,nb+1))
ulstm_v2_obs[:,0] = u

for k in range(1,nt+1):
    un = rk4(ne,dt,u,fr)
    ulstm_v2[:,k] = un
    
    unc = rk4(ne,dt,uc,fr)
    ulstm_v2_c[:,k] = unc
    
    if k%freq == 0:
#        print(k, ' ', int(k/freq))
        xtest = np.hstack((ulstm_v2[:,k].T,uobs[oin,int(k/nf)].T)).reshape(1,-1)
        xtest_sc = scalerIn.transform(xtest)
        xtest_sc = xtest_sc.reshape(1,lookback,nx)
        ytest_sc = model.predict(xtest_sc)
        ytest = scalerOut.inverse_transform(ytest_sc) # residual/ correction
        ytest2[k,:] = ytest
        ucorr[:,int(k/nf)] = ytest.reshape(-1,)
        
        ulstm_v2[:,k] = un 
        ulstm_v2_c[:,k] = un + ytest.reshape(-1,)
        ulstm_v2_obs[:,int(k/nf)] = un + ytest.reshape(-1,)

    u = ulstm_v2[:,k] #np.copy(un)
    uc = ulstm_v2_c[:,k]
    
#ulstm_obs = np.zeros((ne,nb+1)) 
#ulstmd[:,0] = ulstm[:,0]
#ulstmd[:,1:] = ulstm[:,1:] + ucorr[:,:-1]
#
#ulstmd = ulstm + ucorr

#%%
np.savez('data_'+str(me)+'.npz',t=t,tobs=tobs,T=T,X=X,utrue=utrue,uobs=uobs,
         uw=uw,ulstm1=ulstm,ulstm2=ulstm_v2_c,oin=oin)

#%%
t = np.linspace(0,tmax,nt+1)
x = np.linspace(1,ne,ne)

fig, ax = plt.subplots(3,2,sharex=True,figsize=(10,6))

n = [0,15,39]

c = 0
for i in range(3):
    ax[i,c].plot(tobs,uobs[n[i],:],'ro',lw=3)
    ax[i,c].plot(t,utrue[n[i],:],'k-')
#    ax[i,c].plot(t,uw[n[i],:],'b--')    
#    ax[i,c].plot(tobs,ulstm_obs[n[i],:],'o',color='green',markersize=4,lw=1)
    ax[i,c].plot(t,ulstm[n[i],:],'m-.')

    ax[i,c].set_xlim([0,tmax])
    ax[i,c].set_ylabel(r'$x_{'+str(n[i]+1)+'}$')

ax[i,c].set_xlabel(r'$t$')
line_labels = ['Observation','True','LSTM']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=5, labelspacing=0.)

c = 1
for i in range(3):
    ax[i,c].plot(tobs,uobs[n[i],:],'ro',lw=3)
    ax[i,c].plot(t,utrue[n[i],:],'k-')
#    ax[i,c].plot(t,uw[n[i],:],'b--')
#    ax[i,c].plot(tobs,ulstm_v2_obs[n[i],:],'o',color='green',markersize=4,lw=1)
    ax[i,c].plot(t,ulstm_v2_c[n[i],:],'m-.')
    
    ax[i,c].set_xlim([0,tmax])
    ax[i,c].set_ylabel(r'$x_{'+str(n[i]+1)+'}$')

ax[i,c].set_xlabel(r'$t$')

fig.tight_layout()
fig.savefig('ml_'+str(me)+'.pdf')
fig.savefig('ml_'+str(me)+'.png')

#%%
vmin = -10
vmax = 10
fig, ax = plt.subplots(3,1,figsize=(6,7.5))

cs = ax[0].contourf(T,X,utrue,60,cmap='coolwarm',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))
ax[0].set_title('True')

cs = ax[1].contourf(T,X,ulstm,60,cmap='coolwarm',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(ulstm)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))
ax[1].set_title('LSTM')

diff_v1 = utrue - ulstm
diff_v2 = utrue - ulstm_v2_c

cs = ax[2].contourf(T,X,ulstm_v2_c,60,cmap='coolwarm',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(ulstm_v2_c)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[2],ticks=np.linspace(vmin, vmax, 6))
ax[2].set_title('LSTM V2')
#ax[2].grid()

fig.tight_layout()
plt.show() 
fig.savefig('fl_'+str(me)+'.pdf')    
fig.savefig('fl_'+str(me)+'.png')     

print ('V1 %.4f  V2 %.4f ' % (np.linalg.norm(diff_v1), np.linalg.norm(diff_v2)))