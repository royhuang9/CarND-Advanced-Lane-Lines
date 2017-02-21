#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:41:14 2017

@author: roy
"""

from PIL import Image
import numpy as np
import cv2

import matplotlib.pyplot as plt

file_name='./tough/test55.png'
image = np.asarray(Image.open(file_name))

red = image[:,:,0]
green = image[:,:,1]
blue = image[:,:,2]

plt.figure(figsize=(12,9))
plt.imshow(image)
plt.title(file_name)

plt.figure(figsize=(12,9))
plt.imshow(red, cmap='Greys_r')
plt.title('Red')

plt.figure(figsize=(12,9))
plt.plot(red[450, :])
plt.xlim(0, 1280)

plt.figure(figsize=(12,9))
plt.gray()
plt.imshow(green, cmap='Greys_r')
plt.title('Green')

plt.figure(figsize=(12,9))
plt.plot(green[450, :])
plt.xlim(0, 1280)

plt.figure(figsize=(12,9))
plt.gray()
plt.imshow(blue, cmap='Greys_r')
plt.title('Blue')

plt.figure(figsize=(12,9))
plt.plot(blue[450, :])
plt.xlim(0, 1280)

v = np.amax(image[450,:,:], axis=-1)
plt.figure(figsize=(12 ,9))
plt.plot(v)
plt.xlim(0, 1280)
plt.show()

v2 = (v - np.amin(image[450, :, :], axis=-1))/v
plt.figure(figsize=(12,9))
plt.plot(v2)
plt.xlim(0, 1280)
plt.show()

