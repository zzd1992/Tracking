#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:16:22 2017

@author: zhang
"""

import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt
import cv2

def crop(im, pos, model_size, original_size):
    #print 'start crop'
    [h,w,c] = im.shape
    avg_chans = np.zeros(c)
    for i in range(c):
        avg_chans[i] = np.mean(im[:,:,i])
    #print 'size: ',size
    if not (type(original_size) is np.ndarray):
        original_size = np.array([original_size,original_size],dtype='int32')
    sz = original_size.astype('int32')
    result = np.zeros((sz[0],sz[1],c),dtype=np.float32)
    
    # print 'result shape',result.shape
    r = (sz+1)/2
    y_min = int(round(pos[0] - r[0]))   
    y_max = int(sz[0] + y_min) 
    
    
    x_min = int(round(pos[1] - r[1]))
    x_max = int(sz[1] + x_min)
    
    top_pad = max(0, 0 - y_min)    
    down_pad = max(0, y_max - h) 
    left_pad = max(0, 0 - x_min)
    right_pad = max(0, x_max - w)
    
    y_min += top_pad
    y_max += top_pad
    x_min += left_pad
    x_max += left_pad
    #print 'x_min,x_max,y_min, y_max',x_min,x_max,y_min, y_max
    #print 'top_pad,down_pad,left_pad, right_pad',top_pad,down_pad,left_pad, right_pad
    #print type(top_pad),type(down_pad),type(left_pad),type(right_pad)
    for i in range(c):
        img = np.lib.pad(im[:,:,i],((top_pad,down_pad),(left_pad,right_pad)),'constant',constant_values=avg_chans[i])
        #print 'img shape:',img.shape
        result[:,:,i] = img[y_min:y_max,x_min:x_max]
    #result = vggmean(result)
    if (model_size != sz).any:
        result = cv2.resize(result,dsize=(model_size,model_size),interpolation=cv2.INTER_CUBIC)    
    return result   
    
def premnist():
    data = np.load('/home/elliott/zzd_tool/iccv_track/mnist.npz')
    x_train, y_train = data['x_train'], data['y_train']
    del data
    data = np.zeros((10,1000,784))
    for _ in range(10):
        data[_,:,:] = x_train[y_train==_,:][:1000]
    del x_train, y_train
    return np.reshape(data,(10,1000,28,28))
    
def mnist_data(data,shape,noise=None,heatmap=False,down=1):
    n = shape[0]
    f1 = np.zeros(shape,dtype='float32')
    f2 = np.zeros_like(f1)
    c, num = np.random.randint(0,10,n), np.random.randint(0,1000,(n,2))
    pos = np.random.randint(40,61,(n,2))
    move = np.random.randint(-14,15,(n,2))
    for _ in range(shape[0]):
        a, b = pos[_,0], pos[_,1]
        da, db = a+move[_,0], b+move[_,1]
        f1[_,0,a-14:a+14,b-14:b+14] = data[c[_],num[_,0]]
        f2[_,0,da-14:da+14,db-14:db+14] = data[c[_],num[_,1]]
        
    if noise!=None:
        f1 += noise*np.random.randn(shape[0],shape[1],shape[2],shape[3]).astype('float32')
        f2 += noise*np.random.randn(shape[0],shape[1],shape[2],shape[3]).astype('float32')
        
    if heatmap:
        heat = -np.ones((shape[0],shape[1],shape[2]/down,shape[3]/down),'float32')
        Pos = pos+move
        Pos = Pos/down
        for _ in range(n):
            heat[_,0,49+move[_,0]-4:49+move[_,0]+5,49+move[_,1]-4:49+move[_,1]+5] = 1
        return f1, f2, move, heat
    else:
        return f1, f2, move
        
def fftprocess(x,y):
    
    X = np.fft.fftn(x,axes=(2,3),norm='ortho')
    Y = np.fft.fftn(y,axes=(2,3),norm='ortho')
    XX = np.array(np.real(np.fft.ifftn(np.conj(X)*X,axes=(2,3),norm='ortho')),dtype='float32')
    XY = np.array(np.real(np.fft.ifftn(X*Y,axes=(2,3),norm='ortho')),dtype='float32')
    XX = np.fft.ifftshift(XX,axes=(2,3))
    XY = np.fft.ifftshift(XY,axes=(2,3))
    return XX, XY
    
def fft_KCF(x,y,lam):
    X = np.fft.fftn(x,axes=(2,3),norm='ortho')
    Y = np.fft.fftn(y,axes=(2,3),norm='ortho')
   
    X_conj = np.conj(X)
    '''
    XX = np.array(np.real(np.fft.ifftn(X_conj*X,axes=(2,3),norm='ortho')),dtype='float32')
    XY = np.array(np.real(np.fft.ifftn(X_conj*Y,axes=(2,3),norm='ortho')),dtype='float32')
    XX = np.fft.ifftshift(XX,axes=(2,3))
    XY = np.fft.ifftshift(XY,axes=(2,3))
    '''
    out = X_conj*Y/(X_conj*X+lam)
    out = np.fft.ifftshift(np.fft.ifftn(out,norm='ortho',axes=(2,3)))
    return np.real(out).astype('float32')
    
def show(x,y,n,save=False):
    plt.figure(1)
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(x[n+i,0])
    plt.figure(2)
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(y[n+i,0])
    plt.show()
    if save:
        plt.savefig()
    
def bgr2gray(x,keepdim=True):
    xg = 0.114*x[:,0,:,:]+0.587*x[:,1,:,:]+0.299*x[:,2,:,:]
    if keepdim:
        return xg[:,np.newaxis,:,:]
    else:
        return xg

def score2move(x,per=0.75):
    s_0, s_1 = x.shape[0]/2+np.mod(x.shape[0],2)-1, x.shape[1]/2+np.mod(x.shape[1],2)-1
    #x = x-np.min(x)
    #x = x/np.max(x)
    if False:
        v = np.argmax(x)
        b = np.mod(v,x.shape[1])
        a = (v-b)/x.shape[1]
    
    x[x<per] = 0
    X, Y = np.array(range(x.shape[0])), np.array(range(x.shape[1])) 
    X = np.reshape(X,(x.shape[0],1))
    X = np.tile(X,(1,x.shape[1]))
    Y = np.reshape(Y,(1,x.shape[1]))
    Y = np.tile(Y,(x.shape[0],1))
    sum_value = np.sum(x)
    return np.sum(x*X)/sum_value-s_0, np.sum(x*Y)/sum_value-s_1
    
def Loss_log(predict,targets):
    return T.sum(T.mean(T.log(1+T.exp(-predict*targets)),axis=(1,2,3)))
    
def Loss_svm(predict,targets,w,lam):
    return T.mean(T.clip(1-predict*targets,0,1000),axis=(1,2,3))+lam*T.sum(w**2)
    T.mean()
def hanningwindow(n):
    x = np.hanning(n)
    x = np.array(x,dtype='float32')
    x = x[:,np.newaxis]
    x = np.dot(x,x.transpose())    
    return x[np.newaxis,np.newaxis,:,:]
    
def fillscore(x,size,value=-1):
    s = x.shape
    y = value*np.ones((s[0]+2*size,s[1]+2*size),dtype=x.dtype)
    y[size:s[0]+size,size:s[1]+size] = x
    return y
    
def vggmean(im):
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68 
    return im
    
if __name__=='__main__':
    x = np.random.rand(100,100)
    a,b = score2move(x)
    