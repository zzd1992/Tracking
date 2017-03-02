# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 17:32:22 2017

@author: Zhang
"""

import numpy as np
import theano
import theano.tensor as T
import lasagne as l
import lasagne.layers as ll
from lib import *
from model import build_fft_scale
#from preprocessforVideo import *
from VOCsample import VOCsample
import cv2
import matplotlib.pyplot as plt
from sampleFromVideoAndImgs import sample

np.random.seed(1)
p_size = 31

    
def iterate_minibatches(inputs, targets, labels,trans, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], labels[excerpt],trans[excerpt]
        
x, y, targets, target_scale = T.tensor4(), T.tensor4(), T.tensor4(), T.matrix()
lr_var, ls_var = T.scalar(), T.scalar()
pnet, net_scale, net_heat, W = build_fft_scale(x,y,p_size)
params = ll.get_all_params([pnet,net_heat,net_scale],trainable=True)

print params
w_0, w_1, w_2 = params[-2], params[-3], params[-9]
loss_w = 0.0001*(T.sum(T.abs_(w_0))+T.sum(T.abs_(w_1))+T.sum(T.abs_(w_2)))
p_heat, p_scale = ll.get_output(net_heat), ll.get_output(net_scale)
loss_heat = Loss_log(p_heat,targets)
loss_scale = T.sum(T.mean(T.abs_(p_scale-target_scale),axis=1))
Loss = loss_heat+ls_var*loss_scale+loss_w

adam = l.updates.adam(Loss,params,lr_var)
sgd = l.updates.nesterov_momentum(Loss,params,lr_var,0.9)
f_adam = theano.function([x,y,targets,target_scale,lr_var,ls_var],[loss_heat,loss_scale,Loss,p_heat,p_scale],updates=adam)
f_sgd = theano.function([x,y,targets,target_scale,lr_var,ls_var],[loss_heat,loss_scale,Loss,p_heat,p_scale],updates=sgd)

t_heat = ll.get_output(net_heat,deterministic=True)
t_scale = ll.get_output(net_scale,deterministic=True)
t_loss_h, t_loss_s = T.mean((t_heat-targets)**2), T.mean(T.abs_(t_scale-target_scale))
test_f = theano.function([x,y,targets,target_scale],[t_loss_h,t_loss_s,t_heat,t_scale])

#debug = theano.function([x,targets],s)
epoch = 20
lr = np.zeros(epoch,'float32')
lr[:20], lr[20:] = 0.003, 0.001
ls = np.zeros(epoch,'float32')
ls[:10] = 1
ls[10:] = 2
error_list = np.zeros((epoch,2), dtype = np.float32)
batchsize = 256
for i in xrange(epoch):
    x,y,labels,scales = sample()
    num = 0
    for j in range(5):
        for xx, xy, label,scale in iterate_minibatches(x, y, labels,scales, batchsize, shuffle=True):
            if i<20:
                err, err_t, err_total, p_heat, p_scale = f_adam(xx,xy,label,scale,lr[i],ls[i])
            else:
                err, err_t, err_total, p_heat, p_scale = f_sgd(xx,xy,label,scale,lr[i],ls[i])
            num += 1
            error_list[i,0] += err
            error_list[i,1] += err_t
    error_list[i] /= num 
    print 'average loss: {}\n'.format(error_list[i])
np.save('VOC_loss',error_list)

np.savez('model_VOC_VOT.npz',*ll.get_all_param_values([pnet,net_scale,net_heat]))

x,y,labels,scales = sample(1000)
err, error = np.zeros(2), np.zeros(2)
out_h = np.zeros((x.shape[0],1,p_size,p_size),'float32')
out_t = np.zeros((x.shape[0],2),'float32')
num = 0

for xx, xy, label,scale in iterate_minibatches(x, y, labels, scales, batchsize, shuffle=False):
    err[0], err[1],result,result_scale = test_f(xx,xy,label,scale)
    out_h[num*batchsize:(num+1)*batchsize] = result
    out_t[num*batchsize:(num+1)*batchsize] = result_scale
    num += 1
    error[0] += err[0]
    error[1] += err[1]
print error/num   

# save model

np.savez('result_VOC_VOT_2loss',labels,out_h,scales,out_t)