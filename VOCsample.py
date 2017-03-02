# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 21:11:02 2016

@author: Zhangping He
"""
import os
import numpy as np
import random
import cv2
from lib import *
import math
#from preprocessforVideo import crop

def VOCsample(path,model_size =101,label_size = 101,label_clip=20, num=2000,sampleDist = 'Laplace'):
    
    x = np.load(path+'/VOCground.npy').item()
    keys = random.sample(x.keys(),num)
    path += '/JPEGImages/'
    pad_context = 0.5
    num_each_bbox = 4       
    num_samples = num_each_bbox*num
    targets = np.zeros((num_samples,3,model_size,model_size) ,dtype = 'float32')
    searches = np.zeros((num_samples,3,model_size,model_size) ,dtype = 'float32')
    labels = -np.ones((num_samples,1,label_size,label_size) ,dtype = 'float32')
    scales = np.zeros((num_samples,2),dtype = 'float32')
    trans =  np.zeros((num_samples,2),dtype = 'float32')
    real_trans = np.zeros((num_samples,2),dtype = 'float32')
    label_r = 5
    index = 0
    b_x = 1/5.0             # Laplace scale or decay parameter
    b_s = 1/15.0
    for key in keys:
        files = path+key+'.jpg'
        #print 'sample from',key
        im = np.array(cv2.imread(files),dtype=np.float32)
        [im_h,im_w,im_c] = im.shape
        if im.ndim ==2:
            im = im[...,np.newaxis].repeat(3,2)
        for bbox in x[key]:
            w = bbox[2]-bbox[0]
            h = bbox[3]-bbox[1]
            #print 'w,h',w,h
            if max(w,h)>=50 and max(w,h)<=200 and min(w,h) > 0 and w<im_w*0.6 and h<im_h*0.6:
          
                pos = np.array([(bbox[1]+bbox[3])/2.0,(bbox[0]+bbox[2])/2.0],dtype=np.float32)
                hc_z = h + pad_context*(w+h)
                wc_z = w + pad_context*(w+h)
                #print 'hc_z,wc_z',hc_z,wc_z
                sz = round(math.sqrt(hc_z*wc_z))    #实际采样的目标大小
                target = crop(im,pos,model_size,sz)
                target = np.transpose(target,(2,0,1))
                
                if sampleDist == 'Laplace':
                    move = np.random.laplace(loc = 0.0, scale = b_x, size=(num_each_bbox,2))
                    move[move > 0.5] = 0.5
                    move[move < -0.5] = -0.5
                    
                    move[:,0] = move[:,0]*h
                    move[:,1] = move[:,1]*w                 
                   
                    size_scale = np.random.laplace(loc = 1.0, scale = b_s, size=num_each_bbox)
                    size_scale[size_scale<0.7] = 0.7
                    size_scale[size_scale>1.3] = 1.3
                    
                    for n in range(num_each_bbox):
                        size_s = size_scale[n]
                        search = crop(im,pos+move[n,:],model_size,sz*size_s)
                        search = np.transpose(search,(2,0,1))
                        
                        targets[index,...] = target
                        searches[index,...] = search
                        
                        move_actual = -move[n,:]*model_size/(sz*size_s)
                        trans[index,:] = move_actual
                        real_trans[index,:] = -move[n,:]
                        
                        new_pos = np.round(label_size/2+move_actual-1)
                        new_pos = new_pos.astype('int32')
                        labels[index,0,new_pos[0]-label_r:new_pos[0]+label_r,new_pos[1]-label_r:new_pos[1]+label_r] =1 
                        scales[index,:] = np.array([1.0/size_s -1.0,1.0/size_s - 1.0])
                       
                        
                        index += 1
                        
                        if index == num_samples:
                            targets = targets/255.0 -0.5
                            searches = searches/255.0 -0.5
                            labels = labels[:,:,label_clip:(label_size-label_clip),label_clip:(label_size-label_clip)]
                            return 2*targets,2*searches,labels, scales,trans,real_trans
        
       

    targets = targets[:index,...]
    searches = searches[:index,...]    
    labels = labels[:index,...]    
    scales = scales[:index,...]
    trans = trans[:index,...]
    real_trans = real_trans[:index,...]
    
    targets = targets/255.0 - 0.5
    searches = searches/255.0 - 0.5
    labels = labels[:,:,label_clip:(label_size-label_clip),label_clip:(label_size-label_clip)]
    return 2*targets,2*searches,labels,scales,trans,real_trans     

def show(targets,searches,num):
    data_1 = targets[num,...]
    data_1 = np.transpose(data_1,(1,2,0))
    data_1 = data_1 + 0.5
    
    
    data_2 = searches[num,...]
    data_2 = np.transpose(data_2,(1,2,0))
    data_2 = data_2 + 0.5
    
    cv2.namedWindow("target x",cv2.WINDOW_NORMAL) 
    cv2.imshow("target x",data_1)
      
    cv2.namedWindow("search region",cv2.WINDOW_NORMAL) 
    cv2.imshow("search region", data_2)  
    cv2.waitKey(10000) 
    cv2.destroyAllWindows()                  

if __name__ == '__main__':
    voc_path = '/home/elliott/hezhping/VOC2012'
    targets,searches,labels,scales,trans,real_trans  = VOCsample(voc_path)
        
        
                    
                    
                
                
                
        
    
