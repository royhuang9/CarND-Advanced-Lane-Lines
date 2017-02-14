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
    
cameraMt = data['cal']
distCoff = data['dist']

# read perspective matrix data
warp_data_file = './warp_data.pk'
with open(warp_data_file, 'rb') as fp:
    persp_mt = pickle.load(fp)

img_msk = './test_images/test*.jpg'
img_files = glob(img_msk)

for img_file in img_files:
    #read in image
    image = np.asarray(Image.open(img_file).convert('L'))
    
    #undistort
    undst_img = cv2.undistort(image,cameraMt, distCoff, None, cameraMt)
    
    img_size=(image.shape[1], image.shape[0])
    #warp perspective
    img_warped = cv2.warpPerspective(undst_img, persp_mt, img_size)
    plt.figure(figsize=(12,9))
    plt.gray()
    plt.imshow(img_warped)
    plt.title(img_file)
