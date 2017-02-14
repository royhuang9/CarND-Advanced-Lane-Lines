#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:23:52 2017

@author: roy
"""

import os
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

def transform(image, camera_mt, dist_coeff, persp_mt):
    # call pipeline to get a binary image
    img_bin = pipeline(image)
    
    img_reg = region_of_interest(img_bin, vertices)
    
    #undist the image
    img_undist = cv2.undistort(img_reg, camera_mt, dist_coeff, None, camera_mt)
    
    #transform the binary image
    img_size=(image.shape[1], image.shape[0])
    
    #warp perspective
    img_warped = cv2.warpPerspective(img_undist, persp_mt, img_size)
    
    return img_warped

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
    
# read calibration data
with open('./cal_data.pk', 'rb') as fp:
    data = pickle.load(fp)
    
camera_mt = data['cal']
dist_coeff = data['dist']

# read perspective matrix data
warp_data_file = './warp_data.pk'
with open(warp_data_file, 'rb') as fp:
    persp_mt = pickle.load(fp)

msk = './orgdir/test*.png'
fullfile_names = glob(msk)
#print(fullfile_names)
#file_names=[os.path.basename(f) for f in fullfile_names]
#print(fullfile_names)

fullfile_names.sort(key=natural_keys)
#print(fullfile_names)



for file_name in fullfile_names:
    image = np.asarray(Image.open(file_name))
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(imshape[1]//2-50, imshape[0]//2), 
                           (imshape[1]//2+50, imshape[0]//2), 
                            (imshape[1],imshape[0])]], dtype=np.int32)
    
    img_warped = transform(image, camera_mt, dist_coeff, persp_mt)
    file_output = file_name.replace('orgdir', 'bindir')
    
    img_save = np.uint8(255*img_warped/img_warped.max())
    pil_img = Image.fromarray(img_save)
    pil_img.save(file_output)