# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 20:43:36 2019

@author: dell
"""


# =============================================================================
# 第一个参数src，输入图像，8位，三通道的彩色图像，并不要求必须是RGB格式，HSV、YUV等Opencv中的彩色图像格式均可；
# 
# 第二个参数dst，输出图像，跟输入src有同样的大小和数据格式；
# 
# 第三个参数sp，定义的漂移物理空间半径大小；
# 
# 第四个参数sr，定义的漂移色彩空间半径大小；
# 
# 第五个参数maxLevel，定义金字塔的最大层数；
# 
# 第六个参数termcrit，定义的漂移迭代终止条件，可以设置为迭代次数满足终止，迭代目标与中心点偏差满足终止，或者两者的结合；
# =============================================================================


import cv2
import numpy as np
def nothing(x):
    pass
cv2.namedWindow('image')
cv2.createTrackbar('sp','image',0,10,nothing)#创建轨迹条
cv2.createTrackbar('sr','image',0,10,nothing)#创建轨迹条
image=cv2.imread('fc.png')
while(1):
    p = cv2.getTrackbarPos('sp', 'image')  #获取当前轨迹条的位置 0-100
    r = cv2.getTrackbarPos('sr', 'image') #获取当前轨迹条的位置 0-100
    img = cv2.pyrMeanShiftFiltering(src = image, sp = p, sr = r)
    cv2.imshow('image', img) 
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cv2.destroyAllWindows() 