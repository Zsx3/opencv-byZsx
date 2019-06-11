# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:05:15 2019

@author: dell
"""
#简单阈值
# =============================================================================
# 与名字一样，像素值高于阈值时，给这个像素赋予一个新值（可能是白色/黑色）,否则就赋予另外一种
#  颜色，这个函数就是：
#  cv2.threshold()
# 第一个参数是原图像
# 第二个参数是用来对像素值进行分类的阈值
# 第三个参数是当像素高于（有时是小于）阈值时应该被赋予的新的像素值
# 第四个参数 多种阈值方法
# cv2.THRESH_BINARY
# cv2.THRESH_BINARY_INV
# cv2.THRESH_TRUNC
# cv2.THRESH_TOZERO
# cv2.THRESH_TOZERO_INV

# 这个函数有两个返回值，一个是retVal 后面会解释
#                     一个是阈值化后的结果图像
# =============================================================================

#import cv2
#import numpy as np
#from matplotlib import pyplot as plt
#from PIL import Image
#
#img=cv2.imread('jianbian.jpg',0)
#
#ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#ret,thresh2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
#ret,thresh3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
#ret,thresh4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
#ret,thresh5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
#titles=['original','binary','binary_inv','trunc','tozero','tozero_inv']
#images=[img,thresh1,thresh2,thresh3,thresh4,thresh5]
#for i in range(6):
#    plt.subplot(2,3,i+1)
#    plt.imshow(images[i],'gray')  #这里需要加gray  否证会有伪彩色问题
#    plt.title(titles[i])
#    plt.xticks([])
#    plt.yticks([])


#自适应阈值
# =============================================================================
#  简单阈值是全局阈值
#   整副图像一个数作为阈值，不适应所有情况，尤其当一幅图不同部分具有不同亮度时
#   这种情况需要采用自适应阈值，此时的阈值是：############
#    根据图像上的每一个小区域计算其对应的阈值，因此同一副图像的不同区域采用的是不同的
#  值，从而使我们能在亮度不同的情况下得到更好的结果 
#   @ Adaptive Method
#   cv2.ADPTIVE_THRESH_MEAN_C:阈值取自相邻区域的平均值
#   cv2.ADPTIVE_THRESH_GAUSSIAN_C:阈值取自相邻区域的加权和，权重为一个高斯窗口
#   @ Block Size- 邻域大小（用来计算阈值的区域大小）
#   @ C-这就是一个常数，阈值等于的平均值或者加权平均值减去这个常数
# =============================================================================

#import cv2
#import numpy as np
#from matplotlib import pyplot as plt
#img=cv2.imread('jishu.png',0)
##中值滤波
## =============================================================================
## https://blog.csdn.net/qq_27261889/article/details/80822270          各种滤波
## =============================================================================
#cv2.imshow('zhongzhilvbo',img)
#ret,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)#  如果不用Otus二值化，ret就是设定的阈值127
#th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
#th3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#title=['original','global threshold(v=127)','adaptive mean thresholding','adaptive gaussian thresholding']
#images=[img,th1,th2,th3]
#
#for i in range(4):
#    plt.subplot(2,2,i+1)
#    plt.imshow(images[i],'gray')
#    plt.title(title[i])
#    plt.xticks([])
#    plt.yticks([])


#Otsu's二值化
# =============================================================================
# 第一部分，我们提到过retVal，当我们使用Otsu二值化时会用到它
# 在使用全局阈值时，我们就是随便给一个数来作为阈值，那么我们怎么知道我们选取这个数的好坏
# 答案就是不听的尝试，如果是一副双峰图像（简答来说就是图像的直方图存在两个峰）呢，我们岂不是
# 应该在两个峰之间的峰谷选一个值作为阈值？ 这就是Otsu二值化要做的，简单来说，就是对一副
# 双峰图像自动根据其直方图计算出一个阈值，（对于非双峰函数，这种方法得到的结果可能会不理想）

# 这里用到的函数还是 cv2.threshold(),但是需要多传入一个参数（flag）:cv2.THRESH_OTSU
# 这时要把阈值设为0，然后算法会找到最优阈值，这个最优阈值就是返回值retVal.如果不用Otsu
# 二值化，返回的retVal值与设定的阈值相等
# =============================================================================
#下面例子 输入一个带有噪声的图像， 第一种方法，我们设127为全局阈值 第二种方法，直接用Otus二值化
# 第三种方法，我们首先使用一个5x5的高斯核去除噪声，然后再使用Otsu二值化，看看噪声去除对结果影响
# 多大吧~~~~~~
#import cv2
#import numpy as np
#from matplotlib import pyplot as plt
#img=cv2.imread('zaos1.jpg',0)
#
#ret1,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#
#ret2,th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
#blur=cv2.GaussianBlur(img,(5,5),0) #（5，5）为高斯核的大小，0为标准差
#ret3,th3=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #注意将阈值设为0
#
#images=[img,0,th1,img,0,th2,blur,0,th3]
#titles=['original noisy image','histogram','global thresholding','original noisy image','histogram','otus thresholding','gaussian filtered image','histogram','otsus thresholding']
#for i in range(3):
#    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#    plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])
#    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].flatten(),256)
#    plt.title(titles[i*3+1]),plt.xticks([]),plt.yticks([])
#    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#    plt.title(titles[i*3+2]),plt.xticks([]),plt.yticks([])
#plt.show()
    