#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:52:01 2017

@author: roy

Calibrate the camera and store the parameter in file

"""

from PIL import Image
import numpy as np

import cv2
from glob import glob
#import matplotlib.pyplot as plt
import pickle

#the number of inside corners in y
nx = 9
#the number of inside corners in y 
ny = 6

img_msk = './camera_cal/calibration*.jpg'
all_imgnames = glob(img_msk)

#print(all_imgnames)

img_points = []
obj_points = []

objp = np.zeros((ny*nx, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

for imgname in all_imgnames:
    image = np.asarray(Image.open(imgname).convert('L'))
    
    ret, corners = cv2.findChessboardCorners(image, (nx, ny), None)
    if ret == True:
        obj_points.append(objp)
        img_points.append(corners)
        
        '''
        cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
        plt.figure()
        plt.gray()
        plt.imshow(image)
        '''
        
ret, cameraMtx, distCoeff, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image.shape[::-1], None, None)

calfile = './cal_data.pk'
data = {'cal':cameraMtx,
        'dist':distCoeff}

with open(calfile, 'wb') as fp:
    pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)