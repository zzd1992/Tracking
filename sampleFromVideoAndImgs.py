#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:44:30 2017

@author: Zhangping He
"""
import numpy as np
from preprocessforVideo import*
from VOCsample import VOCsample

def sample(num=10000,model_size = 101,label_size=101,label_clip=20):
    target1,search1,label1,scales1,trans1,real_trans1= sample_from_video(int(num*0.5),model_size,label_size,label_clip)    
    target2,search2,label2,scales2,trans2,real_trans2= VOCsample(int(num*0.25),model_size,label_size,label_clip)
    
    target = np.concatenate((target1,target2),axis = 0)
    search = np.concatenate((search1,search2),axis = 0)
    scales = np.concatenate((scales1,scales2),axis = 0)
    labels = np.concatenate((label1,label2),axis = 0)
    return target,search,labels,scales

if __name__ == '__main__':
    target,search,labels,scales = sample()