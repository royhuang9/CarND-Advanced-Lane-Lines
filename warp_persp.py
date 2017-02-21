#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 19:35:19 2017

@author: roy
"""
from PIL import Image
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt


#src = np.asarray([[527, 500], [763, 500], [910, 600], [392, 600]], dtype=np.float32)
src = np.asarray([[524, 500], [768, 500], [910, 600], [392, 600]], dtype=np.float32)
#dst = np.asarray([[392, 300], [910, 300], [910, 600], [390, 600]], dtype=np.float32)
dst = np.asarray([[196, 400], [455, 400], [455, 600], [196, 600]], dtype=np.float32)

image = np.asarray(Image.open('./test_images/straight_undist2.jpg').convert('L'))

mt_persp=cv2.getPerspectiveTransform(src, dst)

#img_size = (image.shape[1]//2, image.shape[0])
img_size = (image.shape[1]//2, image.shape[0])

data = {'mt':mt_persp, 'img_size':img_size}
warp_data_file = './warp_data.pk'
with open(warp_data_file, 'wb') as fp:
    pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)

img_h, img_w = image.shape

img_warped = cv2.warpPerspective(image, mt_persp, img_size)

plt.figure(figsize=(12,9))
plt.gray()
plt.imshow(img_warped)

roi1 = img_warped[:, 180:220]

img3 = np.dstack((img_warped, img_warped, img_warped));

img3[:, (180+257):(220+257), 2] = roi1

plt.figure(figsize=(12,9))
plt.gray()
plt.imshow(img3)

'''
image2 = np.asarray(Image.open('./test_images/straight_undist2.jpg').convert('L'))

img2_warped = cv2.warpPerspective(image2, mt_persp, img_size)
plt.figure(figsize=(12,9))
plt.gray()
plt.imshow(img2_warped)
'''
