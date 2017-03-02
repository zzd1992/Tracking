# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 21:11:02 2016

@author: Zhangping He
"""

import os
import numpy as np
import re # re.split
import time
import shutil
import random
import cv2
import math

np.random.seed(1000)
def sample(img,center,size,num,type='Gaussian'):
    if type == 'Gaussian':        
        result = np.zeros((1,2),dtype = np.int16)
        result[0,:] = center
        result = result.repeat(num,0)   
        #result += np.round(np.min(size)/100*np.minimum(1,np.maximum(-1,np.random.randn(num,2))))
#        result[:,0] = result[:,0]+ np.round(size[0]/24*np.minimum(1,np.maximum(-1,np.random.randn(num))))
#        result[:,1] = result[:,1]+ np.round(size[1]/24*np.minimum(1,np.maximum(-1,np.random.randn(num))))
        result[:,0] += np.random.randint(-size[0]/4,size[0]/4,num)
        result[:,1] += np.random.randint(-size[1]/4,size[1]/4,num)
        result = np.maximum(0,result)
#        result[:,0] = np.minimum(result[:,0],img.shape[1])
#        result[:,1] = np.minimum(result[:,1],img.shape[0])
        return result
        
def get_Gaussian_label(sigma_f,label_size, obj_size, trans):
    if type(obj_size) is int:
        sigma = sigma_f*obj_size
    else:
        sigma = sigma_f*math.sqrt(obj_size.prod())
    sigma = np.minimum(sigma_f*label_size,sigma)
    if type(label_size) is int:
        w = h = label_size
    else:   
        h = label_size[0]
        w = label_size[1]
    c_0 = round(h/2.0-1)
    c_1 = round(w/2.0-1)    
    trans_0 = round(trans[0] + c_0)
    trans_1 = round(trans[1] + c_1)
    nx = np.linspace(0,w-1,w) - trans_1
    ny = np.linspace(0,h-1,h) - trans_0
    xx,xy = np.meshgrid(nx,ny)
    label = np.exp(-0.5*(xx**2 + xy**2)/sigma)
    label[label<0.3] = -1
    return label   
    
def crop(im, pos, model_size, original_size):
    #print 'start crop'
    [h,w,c] = im.shape
    avg_chans = np.zeros(c)
    for i in range(c):
        avg_chans[i] = np.mean(im[:,:,i])
    #print 'size: ',size
    sz = int(original_size)
    result = np.zeros((sz,sz,c),dtype=np.float32)
    
    # print 'result shape',result.shape
    r = (sz+1)/2
    x_min = int(round(pos[0] - r))   
    x_max = int(sz + x_min) 
    
    
    y_min = int(round(pos[1] - r))
    y_max = int(sz + y_min)
    
    top_pad = max(0, 0 - x_min)    
    down_pad = max(0, x_max - h) 
    left_pad = max(0, 0 - y_min)
    right_pad = max(0, y_max - w)
    
    x_min += top_pad
    x_max += top_pad
    y_min += left_pad
    y_max += left_pad
    #print 'x_min,x_max,y_min, y_max',x_min,x_max,y_min, y_max
    #print 'top_pad,down_pad,left_pad, right_pad',top_pad,down_pad,left_pad, right_pad
    #print type(top_pad),type(down_pad),type(left_pad),type(right_pad)
    for i in range(c):
        img = np.lib.pad(im[:,:,i],((top_pad,down_pad),(left_pad,right_pad)),'constant',constant_values=avg_chans[i])
        #print 'img shape:',img.shape
        result[:,:,i] = img[x_min:x_max,y_min:y_max]
    if model_size != sz:
        result = cv2.resize(result,dsize=(model_size,model_size),interpolation=cv2.INTER_CUBIC) 
        result = cv2.GaussianBlur(result,(5,5),1.0)   
    return result    
    
def sample_from_video(input_size =100, label_size =51):
        
    #input_size = 100 
    num_sample = 8000 
    # time_s = time.time()
    # num_sample_frame = 3
    pad_context = 0.5    
    num_each_video = 400
    spatial_sigma_f = 0.08   #Gaussian label sigma factor
    
    #video_path = '/home/elliott/hezhping/OTB50' 
    video_path = '/home/hp1/HeZhangPing/OTB50'
    videoes = os.listdir(video_path)  
    videoes.sort()
    videoes =  videoes[:41] #取前30个视频用作训练集
#    cur_path = os.getcwd()    
#    #result_path = '/home/elliott/hezhping/video_train_set'
#    data_path = cur_path + '/data'
#    if os.path.exists(data_path):
#        shutil.rmtree(data_path)
#        
#    os.mkdir(data_path)
    #
    #bbox_path = result_path + '/bbox.txt'
    ##fid = open(bbox_path,'w')   
    
    
    data = np.zeros((num_sample,3,input_size,input_size),dtype=np.float32)
    region = np.zeros((num_sample,3,input_size,input_size),dtype=np.float32)
    label = -np.ones((num_sample,1,label_size,label_size),dtype=np.float32) 
    translate = np.zeros((num_sample,2),dtype= np.float32)
    real_trans = np.zeros((num_sample,2),dtype= np.float32)
    index = 0
    max_frame_ds = 3 #最大帧间差
    
    #video = videoes[random.randint(0,len(videoes)-1)]
    sample_video = random.sample(videoes,15)
    #sample_video = videoes
    #label_r = 8    #label的中心半径r
    for s_ in xrange(len(sample_video)):
        video = sample_video[s_]
        cur_video_path = video_path + '/' + video     
        
        if video in ['Skating2', 'jump2']:
            gt_path = (cur_video_path +'/groundtruth_rect.{}.txt').format(random.randint(1,2))
        else:
            gt_path = cur_video_path +'/groundtruth_rect.txt'
        gt = open(gt_path,'r')
        gt_arr = [line.strip() for line in gt.readlines()]
        gt_arr = np.array([[int(col) for col in re.split(' |,|\t',row)]for row in gt_arr],dtype=np.int32)
        #print gt_arr.dtype
        center_arr = (np.round(gt_arr[:,:2] + gt_arr[:,2:4]/2-1)).astype(np.int32)
        
        #gt_file = pd.read_csv(gt_path,header=None, names = ['x','y','w','h'],sep = ',' )  
        #gt_arr = np.array(gt_file.values, dtype=np.uint16) 
        img_path = os.path.join(cur_video_path,'img')
        img_files = os.listdir(cur_video_path + '/img')
        img_files.sort()
        #im_arr = [np.array(Image.open(img_path + '/' + im),dtype=np.float32)for im in img_files]
        im_arr = [np.array(cv2.imread(img_path + '/' + im),dtype=np.float32)for im in img_files]
        if im_arr[0].ndim ==2:
            im_arr = [im[...,np.newaxis].repeat(3,2) for im in im_arr]            
        
        num_frame = min(len(img_files),len(gt_arr))
        print '{} images of '.format(num_frame),video
        
        #[height,width,channel] = [im_arr[0].shape[0],im_arr[0].shape[1],im_arr[0].shape[2]]
        
        # start sampling      
        sample_num = min(num_each_video-max_frame_ds-1,num_frame-max_frame_ds-1)
        #print 'sample_num',sample_num
        sample_list = np.arange(0,sample_num)
        sample_list = random.sample(sample_list,sample_num)
        #for i in range(len(gt_arr)-1):
        for i in sample_list:
            #print 'i:',i       
            pre_im = im_arr[i]
           
            
            target_sz = np.array([gt_arr[i][3],gt_arr[i][2]], dtype=np.float32)
            hc_z = target_sz[0] + pad_context*np.sum(target_sz)
            wc_z = target_sz[1] + pad_context*np.sum(target_sz)
            sz = int(math.sqrt(hc_z*wc_z))    #实际采样的目标大小
            max_trans = np.minimum(math.sqrt(target_sz.prod())*0.5,20) 
                        
            #pos = np.array([center_arr[i][1], center_arr[i][0]],dtype=np.float32) 
            pos = center_arr[i][::-1]-1
#            max_move = np.min(np.minimum(np.floor(target_sz*0.2),8))
#            move = np.random.randint(-max_move,max_move,2)
            target = crop(pre_im,pos,input_size,sz) 
            target = np.transpose(target,(2,0,1))
            
            for frame_ds in range(max_frame_ds):
                cur_frame_id = i + frame_ds + 1
                #print 'cur_frame_id',cur_frame_id
                cur_im = im_arr[cur_frame_id] 
                #cur_pos = np.array([center_arr[cur_frame_id][1], center_arr[cur_frame_id][0]],dtype=np.float32)
                cur_pos = center_arr[cur_frame_id][::-1]-1
                trans = cur_pos - pos
                #print 'trans:',trans
                if np.max(np.abs(cur_pos - pos)) < max_trans: 
                                                  
                    search_region = crop(cur_im,pos,input_size,sz)                    
                    search_region = np.transpose(search_region,(2,0,1))
                    data[index,...] = target    
                    region[index,...] = search_region            
                    real_trans[index,...] = trans
                    #relative_move = input_size/2 + move*input_size/size
                    relative_move = (cur_pos-pos)*(label_size)/float(sz)
                    translate[index,...] = relative_move
                    
                    #label[index,0,label_size/2-1+relative_move[0]-label_r:(input_size/down)/2-1+relative_move[0]+label_r,(input_size/down)/2-1+relative_move[1]-label_r:(input_size/down)/2-1+relative_move[1]+label_r] =1 
                    label[index,0,...] = get_Gaussian_label(spatial_sigma_f,label_size,sz,relative_move)
        #            redius = 0.3*np.array([gt_arr[i+1][3],gt_arr[i+1][2]],dtype=np.float32)*(input_size/down)/size
        #            range_move_left = (relative_move - redius)
        #            range_move_right = (relative_move + redius)
        #            range_move_left.astype(np.int32)
        #            range_move_right.astype(np.int32)
        #            #print range_move_left, range_move_right
        #            label[index,0,range_move_left[0]:range_move_right[0],range_move_left[1]:range_move_right[1]] = 1
                    #label[index,0,...] = get_Gaussian_label(spatial_sigma_f,input_size/down,size,relative_move/down)
                    
                    index += 1  
                    if index==num_sample:
                        data = 2*data/255.0-1
                        region = 2*region/255.0-1
                        label = label[:,:,20:81,20:81]
                        #print num_wrong_sample
                        return data,region,label,translate,real_trans
    
        #print index-1
#        data_1 = data[index-1,...]
#        data_1 = np.transpose(data_1,(1,2,0))
#        img = data_1/255
#        
#        
#        data_2 = region[index-1,...]
#        data_2 = np.transpose(data_2,(1,2,0))
#        img1 = data_2/255
#        
#        cv2.namedWindow("target x") 
#        cv2.imshow("target x",img)
#          
#        cv2.namedWindow("search region") 
#        cv2.imshow("search region", img1)  
#        cv2.waitKey(10)   
        
#        time_e = time.time()
#        print 'time cost:{}s'.format(time_e-time_s)
    data = data[:index,...]
    region = region[:index,...]
    data = 2*data/255.0-1
    region = 2*region/255.0-1
    label = label[:index]  
    label = label[:,:,20:81,20:81]
    translate = translate[:index]
    real_trans = real_trans[:index]
    #print num_wrong_sample
    #data = data/255-0.5
    #region = region/255-0.5
    return data,region,label,translate,real_trans
    
if __name__ == '__main__':
    #label_size = np.array([5,5])
#    label_size = 5
#    trans = np.array([2,2])
#    obj_size = np.array([3,3])
#    label = get_Gaussian_label(0.1,label_size, obj_size, trans)
    [data,region,label,trans,real_trans]= sample_from_video()
    #np.savez('sampleFromVideo.npz',data=data,region=region,label=label,trans=trans)
    
