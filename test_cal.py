#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:52:01 2017

@author: roy
"""

from PIL import Image
import numpy as np

import cv2
import matplotlib.pyplot as plt
import pickle
from glob import glob


with open('./cal_data.pk', 'rb') as fp:
    data = pickle.load(fp)
    
cameraMt = data['cal']
distCoff = data['dist']

file_msk = './camera_cal/calibration*.jpg'
allfiles = glob(file_msk)

for filename in allfiles:
    #test code for individial image
    image = np.asarray(Image.open(filename).convert('L'))
    dst_img = cv2.undistort(image,cameraMt, distCoff, None, cameraMt)

    plt.figure(figsize=(12,9))
    plt.gray()
    plt.subplot(1,2,1)
    plt.imshow(image)
    
    plt.subplot(1,2,2)
    plt.imshow(dst_img)

