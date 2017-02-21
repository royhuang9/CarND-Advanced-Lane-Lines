#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 20:44:33 2017

@author: roy
"""

from PIL import Image
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from glob import glob


# read calibration data
with open('./cal_data.pk', 'rb') as fp:
    data = pickle.load(fp)
    
camera_mt = data['cal']
dist_coeff = data['dist']

# read perspective matrix data
warp_data_file = './warp_data.pk'
with open(warp_data_file, 'rb') as fp:
    data = pickle.load(fp)
persp_mt = data['mt']
persp_size = data['img_size']

img_msk = './test_images/straight_lines*.jpg'
img_files = glob(img_msk)

for img_file in img_files:
    #read in image
    image = np.asarray(Image.open(img_file).convert('L')).astype('float')
    
    #undistort
    undst_img = cv2.undistort(image,camera_mt, dist_coeff, None, camera_mt)
    
    img_size=(image.shape[1]//2, image.shape[0])
    #warp perspective
    img_warped = cv2.warpPerspective(undst_img, persp_mt, img_size)

    
    img_save = np.uint8(255*img_warped/np.max(img_warped))
    #plt.figure(figsize=(12,9))
    #plt.gray()
    #plt.plot(img_save[300,:], 'ro')
    #plt.show()
    
    img_pil = Image.fromarray(img_save)
    output_filename=img_file.replace('test_images', 'output_images')
    img_pil.save(output_filename)
    
    plt.figure(figsize=(12,9))
    plt.gray()
    plt.imshow(img_warped)
    plt.title(img_file)
