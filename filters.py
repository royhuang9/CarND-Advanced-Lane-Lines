#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:59:16 2017

@author: roy
"""
from PIL import Image
import numpy as np

import cv2
from glob import glob
import matplotlib.pyplot as plt

def sat_filter(image, s_threshold=(170,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    #l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    #s_channel = image
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_threshold[0]) & (s_channel <= s_threshold[1])] = 1
    return s_binary
    
def sobel_filter(image_gray, orient='x', kern_size = 3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    if orient == 'x':
        sobel_img = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=kern_size)
    elif orient == 'y':
        sobel_img = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=kern_size)
    
    sobel_img = np.abs(sobel_img)
    scaled_sobel = np.uint8(255 * sobel_img/np.max(sobel_img))
    grad_binary = np.zeros_like(image_gray)
    grad_binary[( scaled_sobel >= thresh[0] ) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_filter(image_gray, kern_size = 3, thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=kern_size)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=kern_size)
    
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_mag = np.uint8(255 * mag/np.max(mag))
 
    mag_binary = np.zeros_like(sobelx)
    mag_binary[(scaled_mag >= thresh[0]) & (scaled_mag <= thresh[1])] = 1
    return mag_binary

def direct_filter(image_gray, kern_size = 3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=kern_size)
    sobelx = np.abs(sobelx)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=kern_size)
    sobely = np.abs(sobely)
    
    direct_value = np.arctan2(sobely, sobelx)
    direct_binary = np.zeros_like(sobelx)
    direct_binary[(direct_value > thresh[0]) & (direct_value < thresh[1])] = 1    
    return direct_binary
             
             
def pipeline(image):
    s_th=(170, 255)
    s_bin = sat_filter(image, s_threshold = s_th)

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    sobel_kern = 5
    sobel_th = (50, 150)
    sobelx_bin = sobel_filter(img_gray, orient='x', kern_size=sobel_kern, thresh=sobel_th)
    sobely_bin = sobel_filter(img_gray, orient='y', kern_size=sobel_kern, thresh=sobel_th)
    
    mag_kern = 9
    mag_th = (100, 250)
    mag_bin = mag_filter(img_gray, kern_size=mag_kern, thresh=mag_th)
    
    dir_kern = 9
    dir_th = (0.7, 1.3)
    direct_bin = direct_filter(img_gray, kern_size=dir_kern, thresh=dir_th)
    
    combined_bin = np.zeros_like(img_gray)
    #combined_bin[ (s_bin==1) | ((sobelx_bin==1) & (sobely_bin==1)) \
    #             | ((mag_bin==1) & (direct_bin==1))] = 1
    combined_bin[ (s_bin==1) | ((sobelx_bin==1) & (sobely_bin==1))] = 1
    
    '''
    plt.figure(figsize=(12,9))
    plt.gray()
    plt.imshow(s_bin)
    plt.title('S channel {},{}'.format(s_th[0], s_th[1]))
    
    plt.figure(figsize=(12,9))
    plt.gray()
    plt.imshow(sobelx_bin)
    plt.title('Sobel x {}, {}, {}'.format(sobel_kern, sobel_th[0], sobel_th[1]))
    
    plt.figure(figsize=(12,9))
    plt.gray()
    plt.imshow(sobely_bin)
    plt.title('Sobel y {}, {}, {}'.format(sobel_kern, sobel_th[0], sobel_th[1]))

    plt.figure(figsize=(12,9))
    plt.gray()
    plt.imshow(mag_bin)
    plt.title('Mag {}, {}, {}'.format(mag_kern, mag_th[0], mag_th[1]))
    
    plt.figure(figsize=(12,9))
    plt.gray()
    plt.imshow(direct_bin)
    plt.title('direct {}, {}, {}'.format(dir_kern, dir_th[0], dir_th[1]))
    
    plt.figure(figsize=(12,9))
    plt.gray()
    plt.imshow(combined_bin)
    plt.title('combined')
    '''
    
    return combined_bin

#image = np.asarray(Image.open('../images/test5.jpg'))
#dst = pipeline(image)
