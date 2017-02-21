#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 20:58:48 2017

@author: roy
"""

from PIL import Image
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from glob import glob

from filters import pipeline


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img) 
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def find_lane(image, camera_mt, dist_coeff, persp_mt):
    # call pipeline to get a binary image
    img_bin = pipeline(image)
    
    img_reg = region_of_interest(img_bin, vertices)
    
    #undist the image
    img_undist = cv2.undistort(img_reg, camera_mt, dist_coeff, None, camera_mt)
    
    #transform the binary image
    img_size=(image.shape[1], image.shape[0])
    
    #warp perspective
    img_warped = cv2.warpPerspective(img_undist, persp_mt, img_size)
    
    img_save = np.uint8(255*img_warped/img_warped.max())
    pil_img = Image.fromarray(img_save)
    pil_img.save('./warped.jpg')

    # find lane
    plt.figure(figsize=(12,9))
    plt.gray()
    plt.imshow(img_warped)

    histogram = np.sum(img_warped[int(img_warped.shape[0]/2):, :], axis=0)
    plt.figure(figsize=(12,9))
    plt.gray()
    plt.plot(histogram)


# read calibration data
with open('./cal_data.pk', 'rb') as fp:
    data = pickle.load(fp)
    
camera_mt = data['cal']
dist_coeff = data['dist']

# read perspective matrix data
warp_data_file = './warp_data.pk'
with open(warp_data_file, 'rb') as fp:
    persp_mt = pickle.load(fp)
    
#image = np.asarray(Image.open('./test_images/test1.jpg'))
image = np.asarray(Image.open('./test_images/straight_lines2.jpg'))

imshape = image.shape
vertices = np.array([[(0,imshape[0]),(imshape[1]//2-50, imshape[0]//2), 
            (imshape[1]//2+50, imshape[0]//2), 
             (imshape[1],imshape[0])]], dtype=np.int32)
    
find_lane(image, camera_mt, dist_coeff, persp_mt)