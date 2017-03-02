# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:33:05 2017

@author: Zhangping He
"""
import numpy as np
import theano
import theano.tensor as T
#from theano.tensor import fft
import Customfft as fft
import lasagne as l
import lasagne.layers as ll
from preprocessforVideo import *
#from preprocess import crop
#from utility import *
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import re
from lib import *
from model import *
#from utility import get_trans

class correlationNNTracker(object):
    def __init__(self,video_base_path,show_visualization=True, show_plots=True):
        self.video_base_path = video_base_path 
        self.show_visualization = show_visualization
        self.show_plots = show_plots    
        self.pad_context = 0.5       
        self.input_size = 101 #input size of network  
        self.label_size = 61   
        
        # build network
        print 'building model...' 
        x, y = T.tensor4(), T.tensor4()
        pnet,_,net,_ = build_fft_scale(x,y,31)
        params = ll.get_all_params(net)
        print params
        #params_path = '/home/elliott/hezhping/correlation_net/model_hanning_Gaussian.npz'    
        params_path = '/home/hp1/HeZhangPing/correlation_net/model_101label.npz'
        global param_values
        with np.load(params_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))] 
        ll.set_all_param_values([pnet,net],param_values)       
        pre = ll.get_output(net,deterministic=True)          
        self.pre_f = theano.function([x,y],pre)
        
#        if video_name == 'all': #测试所有视频
#            self.show_visualization=False
#            self.show_plots = False
#            
#            videoes = os.listdir(video_base_path)            
#            for i in range(len(videoes)):
#                video_name = videoes[i]
#                self.track(video_name)                
#        else: #测试某一视频
#            self.track(video_name)    
                     
    def track(self,video_name): 
        #cur_path = os.getcwd()        
#        start = time.time()    
        
        # read gt_bbox from gt file
        print 'read bbox...'
        if video_name in ['Skating2', 'jump2']:
            gt_path = (self.video_base_path +'/'+ video_name+'/groundtruth_rect.{}.txt').format(np.random.randint(1,2))
        else:
            gt_path = self.video_base_path +'/'+ video_name+'/groundtruth_rect.txt'
        gt = open(gt_path,'r')
        gt_arr = [line.strip() for line in gt.readlines()]
        gt_arr = np.array([[int(col) for col in re.split(' |,|\t',row)]for row in gt_arr],dtype=np.int32)
        center_arr = np.round(gt_arr[:,:2] + gt_arr[:,2:4]/2-1)
        
        # read images
        img_path = self.video_base_path + '/' + video_name + '/img'        
        img_files = os.listdir(img_path)
        img_files.sort()       
        im_arr = [np.array(cv2.imread(img_path + '/' + im),dtype=np.float32)for im in img_files]
        if im_arr[0].ndim ==2:
            im_arr = [im[...,np.newaxis].repeat(3,2) for im in im_arr]
        num_frame = min(len(img_files),len(gt_arr))
        print '{} images of '.format(num_frame),video_name        
        predict = np.zeros((num_frame,4),dtype=np.int32) #save track result
        cur_path = os.getcwd()
        # start tracking
        print 'start tracking...'
        for i in range(num_frame):
            
            # print 'gt:',gt_arr[i]
            if i==0: #intialization
                predict[i,:] = gt_arr[i,:]
                pos = (center_arr[0,:][::-1]).astype('float32')
                target_sz = gt_arr[0,2:4][::-1]                
                hc_z = target_sz[0] + self.pad_context*np.sum(target_sz)
                wc_z = target_sz[1] + self.pad_context*np.sum(target_sz)
                sz = int(math.sqrt(hc_z*wc_z))    # 实际采样的目标大小
                print 'sz:',sz
                
                target = crop(im_arr[0],pos,self.input_size,sz)  # save the query target
                # preprocess for network
                
                #target = cv2.resize(target,dsize=(self.input_size,self.input_size),interpolation=cv2.INTER_CUBIC)  
                
                target = np.transpose(target,(2,0,1))                 
                target = 2*target/255.0-1 
                target = target[np.newaxis,...]
                #target = bgr2gray(target)
                predict[i,:] = gt_arr[i,:]
                print 'pos: ',pos
                
                if self.show_visualization:            
                    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)  
            else:                
                search_region = crop(im_arr[i],pos,self.input_size,sz)
                # preprocess for network
                #search_region = cv2.resize(search_region,dsize=(self.input_size,self.input_size),interpolation=cv2.INTER_CUBIC)
               
                search_region = np.transpose(search_region,(2,0,1))                
                search_region = 2*search_region/255.0-1
                search_region = search_region[np.newaxis,...]
                #search_region = bgr2gray(search_region)
                
                # enter to the regression network
                #target,search_region = bgr2gray(target), bgr2gray(search_region)
                #target,search_region = fftprocess(target,search_region)
                #target_fft, search_region_fft = fftprocess(target,search_region)
                global score_map 
                
                score_map = np.array(self.pre_f(target,search_region))                
                score_map = score_map[0,0] 
                score_map = np.lib.pad(score_map,((20,20),(20,20)),'constant', constant_values = -1)
                score_map = score_map-np.min(score_map)
                score_map = score_map/np.max(score_map)
                score_map = cv2.GaussianBlur(score_map,(3,3),1.0)                 
               
#                cv2.namedWindow('score_map', cv2.WINDOW_NORMAL)  
#                cv2.imshow('score_map',score_map)
#                cv2.waitKey (0)
                #score_map = cv2.resize(score_map,dsize=(self.label_size,self.label_size),interpolation=cv2.INTER_CUBIC)
                score_map = cv2.resize(score_map,dsize=(sz,sz),interpolation=cv2.INTER_CUBIC)
                score_map = cv2.GaussianBlur(score_map,(7,7),2.0)  

#                index = score_map.argmax()
#                tran_x = index/int(self.label_size)
#                tran_y = index%int(self.label_size)
#                trans = np.array([tran_x,tran_y]) - np.array([self.label_size,self.label_size])/2
                tran_x,tran_y = score2move(score_map,per=0.85)                
                trans = np.array([tran_x,tran_y],dtype='float32')
                print 'trans:',trans
                #trans = trans*sz/self.label_size                
                #print'trans:',trans
                #trans[0,:] = (trans[0,:]-25)/50*context_size
                pos += trans
                print 'real trans:', center_arr[i,:][::-1] - center_arr[i-1,:][::-1]
                #print 'pos: ',pos
                # save predict
                predict[i,:2] = pos[::-1] - predict[i-1,2:4]/2
                predict[i,2:4] = predict[i-1,2:4]
                # update center,size                
                #center = pos
                #size = np.array([down-top+1,right-left+1],dtype=np.int16)
                
                # resample target                 
#                target = crop(im_arr[i],center,context_size)  # save the query target
#                target = cv2.resize(target,dsize=(self.input_size,self.input_size),interpolation=cv2.INTER_CUBIC)                
#                target = np.transpose(target,(2,0,1))
#                target = target[newaxis,...]
#                target = target/255-0.5                            
            
                # show visualization
                #print 'predict:',predict[i]

                        #cv2.namedWindow(video_name, cv2.WND_PROP_ASPECT_RATIO)  
                img = im_arr[i]/255.0               
                tl = (gt_arr[i,0],gt_arr[i,1])
                br = (gt_arr[i,0] + gt_arr[i,2], gt_arr[i,1] +gt_arr[i,3])
                cv2.rectangle(img,tl,br,(0,250,0),2)
                
                tlE = (predict[i,0],predict[i,1])
                brE = (predict[i,0] + predict[i,2], predict[i,1] + predict[i,3])
                cv2.rectangle(img,tlE,brE,(250,0,250),2)
                
                cv2.imshow(video_name,img)
                cv2.waitKey (10) 
#               save tracked result
#             if os.exists()
                cv2.imwrite(cur_path+'/track_result/'+video_name+'/'+'{:04d}.png'.format(i),img*255.0, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        #return score_map[0]         
#        np.savetxt(video_name + '_output.txt', predict, delimiter=',') 
#        np.save('track_result',predict)
#        #cv2.destroyAllWindows() 
#        end = time.time()
#        print 'process time is %ds.' %(end-start), 'fps is: {}'.format(num_frame/(end-start)) 
        self.sucess_plot(gt_arr,predict)        
                 
                  
    def IoU(self,a,b):   
        if len(b.shape)==1:
            bT = np.zeros((1,4))
            bT[0,:] = b
            b = bT
        res = np.zeros(len(a),dtype=float)
        if b.shape[0]==1:
            b = b.repeat(len(a),0)
            
        InterH = np.minimum(a[:,0]+a[:,2],b[:,0]+b[:,2])-np.maximum(a[:,0],b[:,0])
        InterH = np.maximum(0,InterH)
        
        InterW = np.minimum(a[:,1]+a[:,3],b[:,1]+b[:,3])-np.maximum(a[:,1],b[:,3])
        InterW = np.maximum(0,InterW)
        
        Inter = InterH*InterW
        Union = a[:,2]*a[:,3]+b[:,2]*b[:,3]-Inter        
        res = Inter.astype(float)/Union
        return res 
              
    def sucess_plot(self,gtArr,prediction):
        
        num = gtArr.shape[0]
        if prediction.shape[0] != gtArr.shape[0]:
            num = min(prediction.shape[0],gtArr.shape[0])
        prediction = prediction[:num,:]
        gtArr = gtArr[:num,:]        
        InterOverUnion = self.IoU(prediction,gtArr)
        num_step = 50
        success = np.zeros((num_step,1))
        step = 0
        for s in np.linspace(0,1,num_step):
            success[step] =  np.float(np.sum(InterOverUnion>=s))/num
            step += 1
        if self.show_plots:                
            plt.figure()          
            plt.plot(success,color='Red',linewidth=2,label='Success plot')        
            plt.xlabel('threshold')  
            plt.ylabel('success')  
        return success              
    
    
if __name__ == '__main__':
    #video_path = '/home/elliott/hezhping/OTB50'
    video_path = '/home/hp1/HeZhangPing/OTB50'
    video_name = 'BlurFace'
    
    global fftNNT
    #fftNNT =  correlationNNTracker(video_path,video_name)  
    fftNNT =  correlationNNTracker(video_path)  
    fftNNT.track(video_name)