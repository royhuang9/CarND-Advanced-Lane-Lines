#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:55:32 2017

@author: roy
"""
from moviepy.editor import VideoFileClip

clip1 = VideoFileClip("challenge_video.mp4")
clip1.write_images_sequence('./challenge/ch%d.jpg', fps=None, verbose=True, withmask=True)
