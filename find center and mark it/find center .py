# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:41:26 2019

@author: dell
"""

#import numpy as np
#import cv2
#from matplotlib import pyplot as plt
#img=cv2.imread('img.jpg')
#img_1=cv2.pyrMeanShiftFiltering(img,20,80)
#gray=cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
#ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imshow('thresh',thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
##去除噪声
#kernel=np.ones((3,3),np.uint8)
##opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=1)
##cv2.imshow('open',opening)
##确定背景区域
##sure_bg=cv2.dilate(thresh,kernel,iterations=2)
##cv2.imshow('sure_bg',sure_bg)
#
#dist_transform=cv2.distanceTransform(thresh,1,5)
#ret,sure_fg=cv2.threshold(dist_transform,0.45*dist_transform.max(),255,0)
##cv2.imshow('d_trans',np.uint8(dist_transform*10))
##cv2.imshow('sure_fg',sure_fg)
#
#kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#opening=cv2.morphologyEx(sure_fg,cv2.MORPH_OPEN,kernel)
##cv2.imshow('op',opening)
#opening=np.uint8(opening)
#img1,contours,hierarchy=cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##cv2.drawContours(img,contours,-1,(255,0,0),2)
#
#for c in contours:
#    M=cv2.moments(c)
#    cx=int(M['m10']/M['m00'])
#    cy=int(M['m01']/M['m00']) #根据这些矩的值，可以计算出对象的重心
#    print(cx,cy)
#    cv2.circle(img,(cx,cy),2,(0,0,255),2)
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

import numpy as np
import cv2
from matplotlib import pyplot as plt
def ROI(img,points):
    mask=np.zeros(img.shape,np.uint8)
    if img.shape[-1]>2:
        channel_count=img.shape[-1]
        ignore_mask_color=(255,)*channel_count#白色（255，255，255）
    else:
        ignore_mask_color=255
    cv2.fillPoly(mask,points,ignore_mask_color)
    mask_img=cv2.bitwise_and(img,mask)
    return mask_img
#img=cv2.imread('img.jpg')
img=cv2.imread("img2.jpg")
#img_1=cv2.pyrMeanShiftFiltering(img,20,80)
#gray=cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
#ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imshow('thresh',thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#制作一个掩码ROI
#points=np.array([[(3,292),(1427,519),(778,1070),(0,785)]])#img
points=np.array([[(2,161),(1430,144),(1429,638),(1,791)]])#img2

mask_img=ROI(img,points)
#cv2.imshow('ROI',mask_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
img_1=cv2.pyrMeanShiftFiltering(mask_img,20,80)
gray=cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('thresh',thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#修复小的区域
#kernel=np.ones((5,5),np.uint8)
#closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=1)
#cv2.imshow('open',closing)
#cv2.waitKey(0)
#cv2.destroyAllWindows()