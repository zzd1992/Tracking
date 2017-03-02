# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 21:25:36 2017

@author: Zhangping He
"""
import numpy as np
import theano
import theano.tensor as T
import Customfft as fft
import lasagne as l
import lasagne.layers as ll
from lib import *

def build_correlation_fft(x,y,size):
    pnet = ll.InputLayer((None,3,101,101),input_var=None)
    pnet = ll.BatchNormLayer(pnet)
    pnet = ll.Conv2DLayer(pnet,64,(3,3),pad='same',nonlinearity=None)
    pnet = ll.NonlinearityLayer(ll.BatchNormLayer(pnet),nonlinearity=l.nonlinearities.LeakyRectify(0.1))
    pnet = ll.Pool2DLayer(pnet,(3,3),stride=(2,2))
    pnet = ll.Conv2DLayer(pnet,64,(3,3),pad='same',nonlinearity=None)
    pnet = ll.NonlinearityLayer(ll.BatchNormLayer(pnet),nonlinearity=l.nonlinearities.LeakyRectify(0.1))
    pnet = ll.Conv2DLayer(pnet,32,(3,3),pad='same',nonlinearity=None)
    pnet = ll.BatchNormLayer(pnet)
    x_p, y_p = ll.get_output(pnet,x), ll.get_output(pnet,y)
    x_p, y_p = fft.rfft(x_p,'ortho'), fft.rfft(y_p,'ortho')
    
    XX, XY = T.zeros_like(x_p), T.zeros_like(y_p)
    XX = T.set_subtensor(XX[:,:,:,:,0],x_p[:,:,:,:,0]*x_p[:,:,:,:,0]+x_p[:,:,:,:,1]*x_p[:,:,:,:,1])
    XY = T.set_subtensor(XY[:,:,:,:,0],x_p[:,:,:,:,0]*y_p[:,:,:,:,0]+x_p[:,:,:,:,1]*y_p[:,:,:,:,1])
    XY = T.set_subtensor(XY[:,:,:,:,1],x_p[:,:,:,:,0]*y_p[:,:,:,:,1]-x_p[:,:,:,:,1]*y_p[:,:,:,:,0])
    xx = fft.irfft(XX,'ortho')
    xy = fft.irfft(XY,'ortho')
    
    z_p = T.concatenate((xx,xy),axis=1)
    z_p *= T.constant(hanningwindow(50))
    net = ll.InputLayer((None,64,50,50),input_var=z_p)
    net = ll.BatchNormLayer(net)
    net = ll.NonlinearityLayer(ll.BatchNormLayer(ll.Conv2DLayer(net,64,(5,5),pad='same',nonlinearity=None)))
    net = ll.Pool2DLayer(net,(2,2), mode='average_inc_pad')
    net = ll.NonlinearityLayer(ll.BatchNormLayer(ll.Conv2DLayer(net,64,(5,5),pad='same',nonlinearity=None)))
    net = ll.BatchNormLayer(ll.Conv2DLayer(net,10,(1,1),nonlinearity=None))
    net = ll.DenseLayer(net,size**2,b=None,nonlinearity=None)
    net = ll.ReshapeLayer(net,([0],1,size,size))
    return pnet, net
   
def build_fft_scale(x,y,size):
    W = []
    pnet = ll.InputLayer((None,3,101,101),input_var=None)
    pnet = ll.Conv2DLayer(pnet,64,(3,3),pad='same',nonlinearity=None)
    pnet = ll.NonlinearityLayer(ll.BatchNormLayer(pnet))
    pnet = ll.Pool2DLayer(pnet,(3,3),(2,2))
    pnet = ll.Conv2DLayer(pnet,64,(3,3),pad='same',nonlinearity=None)
    pnet = ll.NonlinearityLayer(ll.BatchNormLayer(pnet),nonlinearity=l.nonlinearities.LeakyRectify(0.1))
    pnet = ll.Conv2DLayer(pnet,32,(3,3),pad='same',nonlinearity=None)
    pnet = ll.BatchNormLayer(pnet)
    x_p, y_p = ll.get_output(pnet,x), ll.get_output(pnet,y)
    z_p = Customfftlayer(x_p,y_p)
    net = ll.InputLayer((None,64,50,50),input_var=z_p)
    net = ll.BatchNormLayer(net)
    net = ll.NonlinearityLayer(ll.BatchNormLayer(ll.Conv2DLayer(net,64,(5,5),pad='same',nonlinearity=None)))
    net = ll.Pool2DLayer(net,(2,2), mode='average_inc_pad')
    net = ll.NonlinearityLayer(ll.BatchNormLayer(ll.Conv2DLayer(net,64,(5,5),pad='same',nonlinearity=None)))
    net = ll.BatchNormLayer(ll.Conv2DLayer(net,10,(1,1),nonlinearity=None))
    
    # return scale different: x_new/x_lod-1
    p_scale = ll.get_output(net)
    #p_scale = theano.gradient.disconnected_grad(p_scale)
    net_scale = ll.InputLayer((None,10,25,25),p_scale)
    net_scale = ll.DenseLayer(net_scale,50,b=None,nonlinearity=l.nonlinearities.tanh)
    W.append(net_scale.get_params()[0])
    net_scale = ll.DenseLayer(net_scale,2,b=None,nonlinearity=None)
    # return heatmap with 2 times upsample of size
    net_heat = ll.DenseLayer(net,size**2,b=None,nonlinearity=None)
    W.append(net_heat.get_params()[0])
    net_heat = ll.BatchNormLayer(net_heat)
    net_heat = ll.ReshapeLayer(net_heat,([0],1,size,size))
    net_heat = ll.Deconv2DLayer(net_heat,64,(5,5),(2,2),b=None,crop ='same',nonlinearity=None)
    net_heat = ll.BatchNormLayer(net_heat)
    net_heat = ll.Conv2DLayer(net_heat,1,(3,3),b=None,pad='same',nonlinearity=None)
    W.append(net_heat.get_params()[0])
    return pnet, net_scale, net_heat, W
    
def Customfftlayer(x_p,y_p):
    x_p, y_p = fft.rfft(x_p,'ortho'), fft.rfft(y_p,'ortho')
    
    XX, XY = T.zeros_like(x_p), T.zeros_like(y_p)
    XX = T.set_subtensor(XX[:,:,:,:,0],x_p[:,:,:,:,0]*x_p[:,:,:,:,0]+x_p[:,:,:,:,1]*x_p[:,:,:,:,1])
    XY = T.set_subtensor(XY[:,:,:,:,0],x_p[:,:,:,:,0]*y_p[:,:,:,:,0]+x_p[:,:,:,:,1]*y_p[:,:,:,:,1])
    XY = T.set_subtensor(XY[:,:,:,:,1],x_p[:,:,:,:,0]*y_p[:,:,:,:,1]-x_p[:,:,:,:,1]*y_p[:,:,:,:,0])
    xx = fft.irfft(XX,'ortho')
    xy = fft.irfft(XY,'ortho')
    z_p = T.concatenate((xx,xy),axis=1)
    z_p *= T.constant(hanningwindow(50))
    return z_p
    
def build_TOY(x,y):
    z_p = T.concatenate((x,y),axis=1)

    net = ll.InputLayer((None,2,100,100),input_var=z_p)
    net = ll.BatchNormLayer(net)
    net = ll.NonlinearityLayer(ll.BatchNormLayer(ll.Conv2DLayer(net,64,(5,5),pad='same',nonlinearity=None)))
    net = ll.Pool2DLayer(net,(2,2), mode='average_inc_pad')
    net = ll.NonlinearityLayer(ll.BatchNormLayer(ll.Conv2DLayer(net,64,(5,5),pad='same',nonlinearity=None)))
    net = ll.Pool2DLayer(net,(2,2), mode='average_inc_pad')
    net = ll.BatchNormLayer(ll.Conv2DLayer(net,10,(1,1),nonlinearity=None))
    net = ll.DenseLayer(net,625,b=None,nonlinearity=None)
    net = ll.ReshapeLayer(net,([0],1,25,25))
    return net
    

    
    
