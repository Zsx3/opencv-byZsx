# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:57:11 2019

@author: dell
"""

import cv2
import numpy as np
img=cv2.imread('sfz.jpg')
rows,cols,channels = img.shape
HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower=np.array([90,110,46])
upper=np.array([110,255,255])
mask=cv2.inRange(HSV,lower,upper)
#cv2.imshow('1',mask)

##腐蚀膨胀
#erode=cv2.erode(mask,None,iterations=1)
##cv2.imshow('erode',erode)
#dilate=cv2.dilate(erode,None,iterations=1)
##cv2.imshow('2',dilate)

for i in range(rows):
    for j in range(cols):
        if mask[i,j]==255:
            img[i,j]=(255,255,255)#此处替换颜色，为BGR通道
cv2.imshow('res',img)
cv2.waitKey(0)