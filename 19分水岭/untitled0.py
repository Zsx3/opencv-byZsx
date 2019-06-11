# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:41:40 2019

@author: dell
"""
#https://zhuanlan.zhihu.com/p/34866092
import numpy as np
import cv2
from matplotlib import pyplot as plt
img=cv2.imread('jishu.png')
img=cv2.pyrMeanShiftFiltering(img,10,50)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('thresh',thresh)
#去除噪声
kernel=np.ones((3,3),np.uint8)
#opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=1)
#cv2.imshow('open',opening)
#确定背景区域
sure_bg=cv2.dilate(thresh,kernel,iterations=2)
cv2.imshow('sure_bg',sure_bg)
#距离变换
dist_transform=cv2.distanceTransform(thresh,1,5)
ret,sure_fg=cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
cv2.imshow('d_trans',np.uint8(dist_transform*10))
cv2.imshow('sure_fg',sure_fg)
#寻找未知源
sure_fg=np.uint8(sure_fg)
unknown=cv2.subtract(sure_bg,sure_fg)
cv2.imshow('uno',unknown)

ret,markersl=cv2.connectedComponents(sure_fg)#对得到的前景进行标记，它会把将背景标记为 0，其他的对象使用从 1 开始的正整数标记。 

markers=markersl+1
markers[unknown==255]=0

markers3=cv2.watershed(img,markers)
img[markers3==-1]=[0,0,255]
cv2.imshow('1',img)
