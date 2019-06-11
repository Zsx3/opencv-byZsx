# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:56:10 2019

@author: dell
"""

#颜色空间转化
#opencv中有超过150种颜色空间转换方法， 但经常用的是BGR<->Gray 和BGR<->HSV
# =============================================================================
# 函数 cv2.cvtColor(input_imgae,flag) flag就是转换类型
#  BGR<->Gray  是cv2.COLOR_BGR2GRAY 
#  BGR<->HSV   是cv2.COLOR_BGR2HSV
# =============================================================================
# HSV格式中，H（色彩/色度）的取值范围是【0，179】
#S(饱和度)取值范围是【0，255】
#V(亮度)取值范围是【0，255】
# https://blog.csdn.net/ColdWindHA/article/details/82080176
#但不同的软件使用的值不同，所有当你需要拿opencv的HSV值与其他软件的HSV值进行对比时，一定要记得归一化

# =============================================================================
# #物体跟踪*******************************************************************
# =============================================================================
#现在我们知道怎么样将一副图像从BGR转换到HSV,我们可以利用这一点来提取某个特定颜色的物体，
#在HSV颜色空间中要比BGR空间中更容易表示一个特点颜色。在我们的程序中，我们要提取的是一个蓝色的
#物体。下面几步：
 #从视频中获取每一帧图片
 #将图像装换到HSV空间
 #设置HSV阈值到蓝色的范围
 #获取蓝色物体，当然我们还可以做其他事，比如在蓝色物体周围画一个圈

#import cv2
#import numpy as np
#video="http://admin:admin@192.168.1.111:8081/"   
#cap =cv2.VideoCapture(video)
#
#while(1):
#    ret,frame=cap.read()
#    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #转换到HSV阈值
#    #设定蓝色阈值
#    lower_blue=np.array([156,43,46])
#    upper_blue=np.array([179,255,255])  #红色
#    #根据阈值构建掩码
#    mask=cv2.inRange(hsv,lower_blue,upper_blue)
## =============================================================================
##     通俗的来讲，这个函数就是判断src中每一个像素是否在[lowerb，upperb]之间，注意集合的开闭。
##     如果结果为是，那么在dst相应像素位置填上255，反之则是0。一般我们把dst当作一个mask来用
## =============================================================================
#    #对原图和掩码进行位运算
#    res=cv2.bitwise_and(frame,frame,mask=mask)
#    #显示图像
#    cv2.imshow('frame',frame)
#    cv2.imshow('mask',mask)
#    cv2.imshow('res',res)
#    k=cv2.waitKey(5)
#    if k==27:
#        break
#cv2.destroyAllWindows()


#怎么样找到要跟踪对象的HSV值（绿色为例）
 #########################
#import cv2
#import numpy as np
#green=np.uint8([[[0,255,0]]])# 不能用【0，255，0】而要用【【【0，255，0】】】
#hsv_green=cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
#print(hsv_green)
##[[[ 60 255 255]]]
#现在则可以用【H-100,100,100】和【H+100,255,255】做为上下阈值





# 把玉米粒灰度化，用于计数的预准备
#import cv2
#import numpy as np
#yumi=cv2.imread('jishu.png')
#hsv=cv2.cvtColor(yumi,cv2.COLOR_BGR2HSV) #转换到HSV阈值
##设定蓝色阈值
#lower_blue=np.array([1,43,46])
#upper_blue=np.array([34,255,255])  #黄白
##根据阈值构建掩码
#mask=cv2.inRange(hsv,lower_blue,upper_blue)
## =============================================================================
##     通俗的来讲，这个函数就是判断src中每一个像素是否在[lowerb，upperb]之间，注意集合的开闭。
##     如果结果为是，那么在dst相应像素位置填上255，反之则是0。一般我们把dst当作一个mask来用
## =============================================================================
#cv2.imshow('mask',mask)
#k=cv2.waitKey(0)
#cv2.destroyAllWindows()