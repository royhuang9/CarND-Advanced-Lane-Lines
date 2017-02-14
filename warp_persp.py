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

src = np.asarray([[525, 500], [762, 500], [914, 600], [380, 600]], dtype=np.float32)
dst = np.asarray([[380, 200], [914, 200], [914, 600], [380, 600]], dtype=np.float32)


image = np.asarray(Image.open('./output_images/straight_lines1.jpg').convert('L'))

mt_persp=cv2.getPerspectiveTransform(src, dst)


warp_data_file = './warp_data.pk'
with open(warp_data_file, 'wb') as fp:
    pickle.dump(mt_persp, fp, pickle.HIGHEST_PROTOCOL)

image_size = (image.shape[1], image.shape[0])

img_warped = cv2.warpPerspective(image, mt_persp, image_size)

plt.figure(figsize=(12,9))
plt.gray()
plt.imshow(img_warped)


image2 = np.asarray(Image.open('./output_images/straight_lines2.jpg').convert('L'))

img2_warped = cv2.warpPerspective(image2, mt_persp, image_size)
plt.figure(figsize=(12,9))
plt.gray()
plt.imshow(img2_warped)
