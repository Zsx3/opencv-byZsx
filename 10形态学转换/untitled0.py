# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:40:38 2019

@author: dell
"""
# =============================================================================
#   https://www.jianshu.com/p/05ef50ac89ac
# 目标：
# 学习不同的形态学操作，例如腐蚀、膨胀、开运算、闭运算等
# 我们要学习的函数有：cv2.erode(),cv2.dilate(),cv2.morphologyEx()等
# 
# 原理：
# 形态学操作是根据图像形状进行的简单操作，一般情况下，对二值化图像进行的操作，需要输入
# 两个参数，一个是原图像，第二个是被称为结构化元素或核，它是用来决定操作的性质的。两个基本
# 的形态学操作是腐蚀和膨胀。他们的变体构成来开运算，闭运算，梯度等。我们会以下图为例逐一介绍他们
# =============================================================================

##1 腐蚀          #详细算法https://max.book118.com/html/2016/1007/57681438.shtm
import cv2
import numpy as np
img=cv2.imread('j.png',0)
kernel=np.ones([5,5],np.uint8)
erosion=cv2.erode(img,kernel,iterations=1)
cv2.imshow('original',img)
cv2.imshow('erosion',erosion)

#玉米粒分割和计数2
#import cv2
#import numpy as np
#yumi=cv2.imread('jishu.png')
#hsv=cv2.cvtColor(yumi,cv2.COLOR_BGR2HSV) #转换到HSV阈值
##设定蓝色阈值
#lower_blue=np.array([1,43,46])
#upper_blue=np.array([34,255,255])  #黄白
##根据阈值构建掩码
#mask=cv2.inRange(hsv,lower_blue,upper_blue)
#kernel=np.ones([12,12],np.uint8)
#erosion=cv2.erode(mask,kernel,iterations=5)
#kernel_p=np.ones([8,8],np.uint8)
#dilate=cv2.dilate(erosion,kernel_p,iterations=5)
## =============================================================================
##     通俗的来讲，这个函数就是判断src中每一个像素是否在[lowerb，upperb]之间，注意集合的开闭。
##     如果结果为是，那么在dst相应像素位置填上255，反之则是0。一般我们把dst当作一个mask来用
## =============================================================================
#cv2.imshow('diolate',dilate)
#k=cv2.waitKey(0)
#cv2.destroyAllWindows()


##2 膨胀
# =============================================================================
# ****这里解释了为什么先腐蚀再膨胀可以去除噪声
#  与腐蚀相反，与卷积和对于的图像，只要有一个是1，中心元素的像素就是1。所以这个操作会增加
#  图像中白色区域（前景），一般在去噪声时，先用腐蚀再用膨胀。因为腐蚀在去掉白噪声的同时，也会使
#  前景对象变小，所有我们再对他进行膨胀，这时噪声已经被去除了，不会再回来了，但是前景还在并且会增加
#  膨胀也可以用来连接两个分开的物体
# =============================================================================

import cv2
import numpy as np
img=cv2.imread('j.png',0)
kernel=np.ones([5,5],np.uint8)
dilation=cv2.dilate(img,kernel,iterations=1)
cv2.imshow('original',img)
cv2.imshow('dilation',dilation)


##3 开运算
# =============================================================================
# 先进行腐蚀再进行膨胀就叫做开运算。就像我们上面介绍的那样，它被用来去除噪声。这里使用的
# 函数是cv2.morphologyEx()
# =============================================================================
import cv2
import numpy as np
img=cv2.imread('j_1.png',0)
kernel=np.ones([5,5],np.uint8)
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
cv2.imshow('original',img)
cv2.imshow('open',opening)


##4 闭运算
# =============================================================================
# 先膨胀再腐蚀，经常被用来填补前景物体中的小洞，或者在前景物体上的小黑点
# =============================================================================
import cv2
import numpy as np
img=cv2.imread('j_2.png',0)
kernel=np.ones([5,5],np.uint8)
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
cv2.imshow('original',img)
cv2.imshow('close',closing)


##5 形态学梯度
# =============================================================================
# 其实就是一副图像膨胀与腐蚀的差别
# 结果看上去就像前景物体的轮廓
# =============================================================================
import cv2
import numpy as np
img=cv2.imread('j.png',0)
kernel=np.ones([5,5],np.uint8)
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
cv2.imshow('original',img)
cv2.imshow('gradient',gradient)


##6 礼帽
# =============================================================================
# 原始图像与开运算后图像的差，下面这个例子就是用一个9x9的核进行礼帽操作的结果（因为是9*9,所有在开运算（腐蚀+膨胀）的腐蚀步骤时，j就已经残缺不堪了再膨胀就变成3个大点）
# 原图减去3个大点得到的图像就是j多来三个大窟窿
# =============================================================================
import cv2
import numpy as np
img=cv2.imread('j.png',0)
kernel=np.ones([9,9],np.uint8)
tophat=cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
cv2.imshow('original',img)
cv2.imshow('tophat',tophat)


##7 黑帽
# =============================================================================
# 闭运算后与原图像的差
# =============================================================================
import cv2
import numpy as np
img=cv2.imread('j.png',0)
kernel=np.ones([9,9],np.uint8)
blackhat=cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
cv2.imshow('original',img)
cv2.imshow('blackhat',blackhat)