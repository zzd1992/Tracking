#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:44:30 2017
@author: Zhangping He
"""
import numpy as np
from preprocessforVideo import*
from VOCsample import VOCsample

def sample(num=10000,noise=None,model_size = 101,label_size=101,label_clip=20):
    target,search,label,scales,trans,real_trans= sample_from_video(int(num*0.5),model_size,label_size,label_clip)    
    target2,search2,label2,scales2,trans2,real_trans2= VOCsample(int(num*0.25),model_size,label_size,label_clip)
    
    target = np.concatenate((target,target2),axis = 0)
    search = np.concatenate((search,search2),axis = 0)
    scales = np.concatenate((scales,scales2),axis = 0)
    labels = np.concatenate((label,label2),axis = 0)
    if noise is not None or noise!=0:
        s = target.shape
        target += noise*np.random.randn(s[0],s[1],s[2],s[3]).astype('float32')
        search += noise*np.random.randn(s[0],s[1],s[2],s[3]).astype('float32')
    return target,search,labels,scales

if __name__ == '__main__':
    target,search,labels,scales = sample()
