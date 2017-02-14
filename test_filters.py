#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 18:47:38 2017

@author: roy
"""

from PIL import Image
import numpy as np

import pickle
import matplotlib.pyplot as plt

from filters import pipeline
    
image = np.asarray(Image.open('../images/test5.jpg'))
dst = pipeline(image)
plt.figure(figsize=(12,9))
plt.gray()
plt.imshow(dst)
