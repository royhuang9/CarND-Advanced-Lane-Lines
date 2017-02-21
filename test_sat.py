#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 21:27:33 2017

@author: roy
"""

from PIL import Image
import numpy as np

from glob import glob
import matplotlib.pyplot as plt
import cv2

  
def sat_filter(image, s_threshold=(120,255)):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    plt.figure(figsize=(12,9))
    plt.gray()
    plt.imshow(s_channel)
    plt.title('S channel')
    
    #clear the higher part half of image v_channel
    img_h, img_w = v_channel.shape
    v_channel[:img_h//2, :] = 0
    
    # rescale with the lower part
    v_scaled = (255 * v_channel/np.max(v_channel))
    
    #threshold the pixel with low v value
    v_scaled[v_scaled < 125.5] = 0
    #combine them together

    plt.figure(figsize=(12,9))
    plt.gray()
    plt.imshow(v_scaled)
    plt.title('v scaled')
    
    combined = np.zeros_like(s_channel)
    combined[ (s_channel >= s_threshold[0]) & (s_channel <= s_threshold[1]) & \
             (v_scaled > 0)] = 1
    
    plt.figure(figsize=(12,9))
    plt.gray()
    plt.imshow(combined)
    plt.title('combined')

    return combined
    
files = glob('./tough/test149.png')
#files = glob('./orgdir/test149.png')
for file_name in files:
    image = np.asarray(Image.open(file_name))
    sat_filter(image)