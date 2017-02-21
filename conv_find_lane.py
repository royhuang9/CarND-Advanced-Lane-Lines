#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:03:46 2017

@author: roy
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1)*height):
        int(img_ref.shape[0] - level*height),
        max(0, int(center - width/2)):
            min(int(center + width/2), img_ref.shape[1])] = 1
            
def find_window_centroids(image, window_width, window_height, margin):
    img_h, img_w = image.shape
    window_centroids = []
    window = np.ones(window_width)
    
    l_sum = np.sum(image[(3*img_h//4):,:(img_w//2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width//2
    
    r_sum = np.sum(warped[(3*img_h//4):, (img_w//2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) + img_w//2 - window_width//2
    print('img_h {}, img_w {}, l_center {} r_center {}'.format(img_h, img_w, l_center, r_center))
    
    window_centroids.append((l_center, r_center))
    
    for level in range(1, (img_h//window_height)):
        image_layer = np.sum(image[(img_h - (level+1)*window_height)
                            :(img_h - level*window_height), :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        offset = window_width//2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset, img_w))
        l_center_cur = np.argmax(conv_signal[l_min_index: l_max_index]) + l_min_index - offset
        if l_center_cur > 0: l_center = l_center_cur 
        print('l_min_index {} l_max_index {} l_center_cur {}'.format(l_min_index, l_max_index, l_center_cur))
        
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, img_w))
        r_center_cur = np.argmax(conv_signal[r_min_index: r_max_index]) + r_min_index - offset
        if r_center_cur > 0: r_center = r_center_cur
        
        print('l_center:{}, r_center:{}'.format(l_center, r_center))
        window_centroids.append((l_center, r_center))

    return window_centroids
        
warped = np.asarray(Image.open('./warped.jpg'))

window_width = 50
window_height = 80
margin = 100

window_centroids = find_window_centroids(warped, window_width, window_height, margin)

#print(window_centroids)

# If we found any window centers
if len(window_centroids) > 0:

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows 	
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
	    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
	    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
	    # Add graphic points from window mask here to total pixels found 
	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channle 
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 
# If no window centers found, just display orginal road image
else:
    output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

# Display the final results
plt.figure(figsize=(12,9))
plt.imshow(output)
plt.title('window fitting results')
plt.show()
