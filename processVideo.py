# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:19:20 2017

@author: elliott
"""
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import re
import numpy as np
from lib import *

#video_base_path = '/home/elliott/hezhping/OTB50' 
video_base_path = '/home/hp1/HeZhangPing/vot2016'

videoes = os.listdir(video_base_path)  
videoes.sort()

for n in xrange(len(videoes)-11):
    video_name = videoes[n+11]
    
    #video_name = 'Ball'
    video_path = video_base_path + '/' + video_name
    gt_path = video_path +'/groundtruth_rect.txt'
    
    gt = open(gt_path,'r')
    gt_arr = [line.strip() for line in gt.readlines()]
    gt_arr = np.array([[float(col) for col in re.split(' |\,|\t',row)]for row in gt_arr],dtype='float32')
    #gt_arr = get_axis_aligned_BB(gt_arr) 
    #gt_arr[:,:2] -= gt_arr[:,2:4]/2
    '''
    gt_rect_path = video_path +'/groundtruth.txt'
    np.savetxt(gt_rect_path, gt_arr, fmt='%.4f',delimiter=' ') 
    '''

        

    img_path = video_base_path + '/' + video_name + '/img'        
    img_files = os.listdir(img_path)
    img_files.sort() 
    
    im_arr = [np.array(cv2.imread(img_path + '/' + im),dtype=np.float32)for im in img_files]
    if im_arr[0].ndim ==2:
        im_arr = [im[...,np.newaxis].repeat(3,2) for im in im_arr]
    num_frame = min(len(img_files),len(gt_arr))
    print '{} images of '.format(num_frame),video_name 
    
    for i in range(num_frame):
        
        if i==0:
            cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)  
            #cv2.namedWindow(video_name, cv2.WND_PROP_ASPECT_RATIO)  
        img = im_arr[i]/255.0
#        tl = (gt_arr[i,0], gt_arr[i,1])      
#        br = (gt_arr[i,2]+gt_arr[i,0],  gt_arr[i,3]+gt_arr[i,1])    
#        cv2.rectangle(img,tl,br,(0,250,0),2)
        
        pts = np.array([gt_arr[i,2:4],gt_arr[i,0:2],gt_arr[i,6:8],gt_arr[i,4:6]],dtype=np.int32)
        pts = np.reshape(pts,(4,1,2))
        cv2.polylines(img,[pts],True,(0,250,0),2)
        cv2.imshow(video_name,img)
        cv2.waitKey(10)
    cv2.destroyAllWindows() 
    
'''
# 重命名文件
for i in range(len(img_files)):
    file_name = img_files[i]
    #print 'file_name',file_name
    #id_frame = int(file_name[:-4])
    #print ' id_frame', id_frame
    #new_file_name = '{:04d}.jpg'.format(id_frame)
    new_file_name = '{:04d}.jpg'.format(i+1)
    #print 'new_file_name',new_file_name
    os.rename(img_path + '/'+file_name,img_path + '/' + new_file_name)
'''
'''   
with open(gt_path,'r') as gt:
    gt_arr = [line.strip() for line in gt.readlines()]
    gt_arr = np.array([[int(col) for col in re.split(' |,|\t',row)]for row in gt_arr],dtype=np.uint16)
    #gt_arr = gt_arr[300:]
    print 'line number:',len(gt_arr)
with open(video_path +'/groundtruth_rect_1.txt','w') as gt: 
    for i in range(len(gt_arr)):
        line = str(gt_arr[i,0])+','+str(gt_arr[i,1])+','+str(gt_arr[i,2])+','+str(gt_arr[i,3])+'\n'
        gt.write(line)
'''
        