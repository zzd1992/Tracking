#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:57:51 2017

@author: zhang
"""

import numpy as np

import theano
import theano.tensor as T
from theano.tensor import fft
from theano.tensor.signal.pool import pool_2d
import lasagne as l
import lasagne.layers as ll
import matplotlib.pyplot as plt
from lib import *
def show(n):
    plt.figure(1)
    for i in range(16):
        plt.subplot(4,4,i)
        plt.imshow(result[n+i,0])
    plt.figure(2)
    for i in range(16):
        plt.subplot(4,4,i)
        plt.imshow(label[n+i,0])
    plt.show()

def toydata(shape):
    x = np.zeros(shape,dtype='float32')
    y, label = np.zeros_like(x), np.zeros_like(x)
    size = shape[2:]
    lx = np.random.randint(np.floor(0.3*size[0]),np.floor(0.6*size[0]),shape[0])
    ly = np.random.randint(np.floor(0.3*size[1]),np.floor(0.6*size[1]),shape[0])
    rx = lx+10+np.floor(np.random.rand(shape[0])*5).astype('int')
    ry = ly+10+np.floor(np.random.rand(shape[0])*5).astype('int')
    
    dx = np.floor(np.random.randn(shape[0])*3).astype('int')
    dy = np.floor(np.random.randn(shape[0])*3).astype('int')
    s2, s3 = shape[2]/2-1, shape[3]/2-1
    
    for i in range(shape[0]):
        x[i,:,lx[i]:rx[i],ly[i]:ry[i]] = 0.5+0.1*np.random.randn(shape[1],rx[i]-lx[i],ry[i]-ly[i]).astype('float32')
        y[i,:,lx[i]+dx[i]:rx[i]+dx[i],ly[i]+dy[i]:ry[i]+dy[i]] = x[i,:,lx[i]:rx[i],ly[i]:ry[i]]
        #label[i,:,lx[i]+dx[i]:rx[i]+dx[i],ly[i]+dy[i]:ry[i]+dy[i]] = 1
        label[i,:,s2+dx[i]-(rx[i]-lx[i])/2:s2+dx[i]+(rx[i]-lx[i])/2,s3+dy[i]-(ry[i]-ly[i])/2:s3+dy[i]+(ry[i]-ly[i])/2] = 1
    X = np.fft.fftn(x,axes=(2,3),norm='ortho')
    Y = np.fft.fftn(y,axes=(2,3),norm='ortho')
    #print X.shape
    X_conj = np.conj(X)
    #Y_conj = np.conj(Y)
    #print X_conj is X
    #X_conj = copy.deepcopy(X)
    #Y_conj = copy.deepcopy(Y)
    #X_conj[:,:,:,]
    XX = np.array(np.real(np.fft.ifftn(X_conj*X,axes=(2,3),norm='ortho')),dtype='float32')
    XY = np.array(np.real(np.fft.ifftn(X_conj*Y,axes=(2,3),norm='ortho')),dtype='float32')
    XX = np.fft.ifftshift(XX,axes=(2,3))
    XY = np.fft.ifftshift(XY,axes=(2,3))
    return XX, XY, label, dx, dy


frame, targets = T.tensor4(), T.tensor4()
net = ll.InputLayer((None,2,100,100),input_var=frame)
net = ll.Conv2DLayer(net,32,(5,5),b=None,pad='same')
net = ll.Pool2DLayer(net,(2,2), mode='average_inc_pad')
net = ll.Conv2DLayer(net,8,(3,3),b=None,pad='same',nonlinearity=l.nonlinearities.LeakyRectify(0.1))
net = ll.Pool2DLayer(net,(2,2), mode='average_inc_pad')
net = ll.DenseLayer(net,625,b=None,nonlinearity=None)
net = ll.ReshapeLayer(net,([0],1,25,25))
predict = ll.get_output(net)
targets_pool = pool_2d(targets, ds=(4,4), mode='average_inc_pad')


loss = T.mean((predict-targets_pool)**2)
params = ll.get_all_params(net,trainable=True)
updates = l.updates.adam(loss,params,0.01)

train_f = theano.function([frame,targets],[loss,predict],updates=updates)
data = premnist()
errlist = []
for i in range(6000):
    x, y, move, label = mnist_data(data,(32,1,100,100),noise=None,heatmap=True,down=1)
    xx, xy = fftprocess(x,y)
    err, result = train_f(np.concatenate((xx,xy),axis=1),label)
    errlist.append(err)
    if (i+1)%10==0:
        print i+1,err
np.savez('toymodel.npz',*ll.get_all_param_values(net))
