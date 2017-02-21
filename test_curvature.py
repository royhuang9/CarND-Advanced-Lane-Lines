#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 09:32:07 2017

@author: roy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 19:35:19 2017

@author: roy
"""
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

#src = np.asarray([[527, 500], [763, 500], [910, 600], [392, 600]], dtype=np.float32)
src = np.asarray([[524, 500], [768, 500], [910, 600], [392, 600]], dtype=np.float32)
#dst = np.asarray([[392, 300], [910, 300], [910, 600], [390, 600]], dtype=np.float32)
dst = np.asarray([[196, 300], [455, 300], [455, 600], [196, 600]], dtype=np.float32)

image = np.asarray(Image.open('./test_images/straight_undist2.jpg').convert('L'))


mt_persp=cv2.getPerspectiveTransform(src, dst)

img_size = (image.shape[1]//2, image.shape[0])

img_warped = cv2.warpPerspective(image, mt_persp, img_size)

plt.figure(figsize=(12,9))
plt.gray()
plt.imshow(img_warped)

def lane_search(image):
    nwindows = 10
    margin = 100
    minpix = 50
    
    img_h, img_w = image.shape
    window_height = img_h//nwindows
    
    hstg = np.sum(image[img_h//2:, :], axis=0)
    img_out = np.dstack((image, image, image))*255
    
    midpoint = img_w//2
    leftx_base = np.argmax(hstg[:midpoint])
    rightx_base = np.argmax(hstg[midpoint:]) + midpoint
    
    nonzero = image.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]
    
    leftx_current = leftx_base
    rightx_current = rightx_base
                            
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = img_h - (window + 1)*window_height
        win_y_high = img_h - window * window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
    
        #cv2.rectangle(img_out, (win_xleft_low, win_y_low), 
        #              (win_xleft_high, win_y_high), (0, 255, 0), 2)
        #cv2.rectangle(img_out, (win_xright_low, win_y_low), 
        #              (win_xright_high, win_y_high), (0, 255, 0), 2)
    
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
    
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    print(left_fit)
    
    radius_l = cal_radius(leftx, lefty)
    print(radius_l)

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    right_fit = np.polyfit(righty, rightx, 2)
    print(right_fit)
    
    radius_r = cal_radius(rightx, righty)
    print(radius_r)
    
    # find indice for the whole image
    whole_img = np.indices((img_h, img_w))
    imgx= whole_img[1].flatten()
    imgy = whole_img[0].flatten()
    
    #print('imgx shape {}, imgy shape {}'.format(imgx.shape, imgy.shape))
    # got indice between left lane and right lane
    lane_inds = ((imgx >= (left_fit[0]*(imgy**2) + left_fit[1] * imgy + left_fit[2])) 
                    & (imgx < (right_fit[0]*(imgy**2) + right_fit[1] * imgy + right_fit[2])))
    # paint the lane to green
    img_out[imgy[lane_inds], imgx[lane_inds]] = [0, 255, 0]
    
    #paint left track to Red, right track to green
    img_out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    img_out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    

    return img_out

def cal_radius(x, y):
    mppx = 3.7/270
    #mppx = 3.7/540
    mppy = 3.0/100
    fit_rw = np.polyfit(y * mppy, x * mppx, 2)
    py_rw = 350 * mppy
    radius = (1.0 + (2*fit_rw[0]*py_rw+fit_rw[1])**2)**1.5/(2.0*fit_rw[0])
    return radius
    
def sobel_filter(image_gray, orient='x', kern_size = 3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    if orient == 'x':
        sobel_img = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=kern_size)
    elif orient == 'y':
        sobel_img = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=kern_size)
    
    sobel_img[ sobel_img < 0 ] =0
    #sobel_img = np.abs(sobel_img)
    scaled_sobel = np.uint8(255 * sobel_img/np.max(sobel_img))
    grad_binary = np.zeros_like(image_gray)
    grad_binary[( scaled_sobel >= thresh[0] ) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

img3 = sobel_filter(img_warped, orient='x', kern_size=5, thresh=(100, 255))
plt.figure(figsize=(12,9))
plt.gray()
plt.imshow(img3)

img4 = lane_search(img3)

plt.figure(figsize=(12,9))
plt.gray()
plt.imshow(img4)
