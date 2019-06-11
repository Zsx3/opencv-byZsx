# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:59:26 2019

@author: dell
"""
#
#直方图反向投影
#目标：本节我们将要学习直方图的反向投影
#原理：
#可以用来做图像分割，或者在图像中寻找我们感兴趣的部分，简单来说，它会 输出 与输入图(待搜索)同样大小
#的图像，其中每一个像素值代表了输入图像上对应的点属于目标对象的 概率，用更简洁的话说，输出图像 中像素
#越高（越白）的点就越可能代表我们要搜索的目标（在输入图像所在的位置）。这是一个直观的解释。直方图
#投影经常与camshift算法等一起使用
#我们该怎么样实现这个算法呢？首先我们要为一张包含我们要查找目标的图像建立直方图（在我们实例中，我们要
#查找的是草地，其他的都不要）。我们要查找的对象要尽量占满这张图像（换句话说，这张图像上最好是有且仅有我们
#要查找的对象）。最好使用颜色直方图，因为一个物体的颜色要比它的灰度图能更好的被用来进行图像分割和对象识别。
#接着我们再把这个颜色直方图投影到输入图像中寻找我们的目标，也就是找到输入图像中每一个像素点的像素值在直方图
#中对应的概率，这样我们就得到了一个概率图像，最后设置适当的阈值对概率图像进行二值化，嗯。
#
#opencv中的反向投影
#opencv提供的函数cv2.calcBackProject()可以用来做直方图反向投影。它的参数与cv2.calcHist()参数基本相同
#其中的一个参数是我们要查找的目标的直方图


import cv2
import numpy as np
target=cv2.imread('jishu.png')
hsv=cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

roi=cv2.imread('fc_white.png')
hsvt=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

roihist=cv2.calcHist([hsvt],[0,1],None,[180,256],[0,180,0,256])
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)#原始图像，结果图像，映射到结果图像中的最小值，最大值，归一化类型
dst=cv2.calcBackProject([hsv],[0,1],roihist,[0,180,0,256],1)

#此处卷积可以把分散的点连在一起
disc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
dst=cv2.filter2D(dst,-1,disc)
cv2.imshow('dst',dst)
ret,thresh=cv2.threshold(dst,10,255,0)
#cv2.imshow('1',thresh)

thresh=cv2.merge((thresh,thresh,thresh))#三通道图像
res=cv2.bitwise_and(target,thresh)


res=np.vstack((target,thresh,res))
cv2.imshow('1',res)
cv2.waitKey(0)