# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:59:53 2019

@author: dell
"""

目标：
本节我们会学习如何绘制2D直方图，我们会在下一节中使用到它

介绍：
在前面的部分，我们介绍了如何绘制一维直方图，之所以称之为一维，是因为我们只考虑了图像的一个
特征：灰度值。但在2d直方图中，我们就要考虑两个图像的特征，对于彩色的图像的直方图通常下我们要
考虑每个的颜色（Hue）和饱和度（Saturation）。根据这两个特征绘制2D直方图

opencv的官方文档中包含一个创建彩色直方图的例子

#2D直方图
cv.calcHist()来计算直方图既简单又方便，如果要绘制颜色直方图的话，我们首先要将图像的颜色空间从BGR到HSV
(记住，计算一维直方图要从BGR到HSV).计算2D直方图，函数的参数要做如下修改：
channels=[0,1]因为我们同时需要处理H和S两个通道
bins=[180,256] H通道为180 S通道为256
range=[0,180,0,256] H的取值范围在0到180，S的取值范围在0到256

#numpy 的2d直方图
import cv2
import numpy as np
img=cv2.imread('zhongxin.jpg')
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hist,xbins,ybins=np.histogram2d(h.ravel(),s.ravel(),[180,256],[[0,180],[0,256]])

#绘制2D直方图
#方法1：使用cv2.imshow() 我们得到结果是一个180x256的二维数组。
#所以我们可以使用函数cv2.imshow()来显示它。但是这是一个灰度图，除非我们知道不同颜色
#H通道的值，否则我们根本不知道到底代表什么颜色

#方法2：使用matplotlib() 我们还可以使用函数matplotlib.pyplot.imshow()
#来绘制2d直方图。再搭配不同颜色图（color_map）.这样我们会对每个点所表示的数值大小有一个更直观的认识。但
#和前面问题一样，你还是不知道那个数代表什么颜色，虽然如此，我还是更喜欢这个方法，简单好用
#@注意：在使用这个函数时，要记住设置插值函数为nearest

import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('zhongxin.jpg')
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hist=cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
plt.imshow(hist,interpolation='nearest')
plt.show()