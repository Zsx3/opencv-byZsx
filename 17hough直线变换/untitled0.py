# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:10:00 2019

@author: dell
"""

#opencv中的霍夫变换
#
#上面介绍的整个过程在opencv中都被封装在一个函数里：cv2.HoughLines()
#返回值就是（ρ，θ）。ρ的单位是像素，θ的单位是弧度。这个函数的第一个参数是二值化图像
#，所以进行霍夫变换之前首先要进行二值化，或者进行Canny边缘检测。第二和第三个值分别代表
#ρ和θ的精确值。第四个参数是阈值，只有累加其中的值高于阈值时才被认为是一条直线，也可以把它看成能
#检测的直线的最短长度（以像素点为单位）

import cv2
import numpy as np
img=cv2.imread('2.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,50,150,False)
cv2.imshow('edge',edges)

lines=cv2.HoughLines(edges,0.7,np.pi/180,140)
a,b,c=lines.shape
lines=lines.reshape((a,c))
for rho,theta in lines:
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho
    x1=int(x0+1000*(-b))
    y1=int(y0+1000*(a))
    x2=int(x0-1000*(-b))
    y2=int(y0-1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
cv2.imshow('1',img)
#x=np.array([[1,2],[3,2],[1,2]],np.float32)
#for m,n in x:
#    print(m,n)
#print(x.shape)

#Probabilistic_Hough_Transform
对霍夫变换的一种优化--概率霍夫变换

cv2.HoughLinesP()
minLineLength 线的最短长度
MaxLineGap 两条线段之间的最大间隔，如果小于此值，这两条直线会被看成一条直线

更给力的是，这个函数返回值就是直线的起点和终点，前面的例子只得到了直线的ρ，θ

import cv2
import numpy as np

img=cv2.imread('3.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,50,150,3)
minLineLength=50
maxLineGap=10
lines=cv2.HoughLinesP(edges,1,np.pi/180,minLineLength,maxLineGap)
#a,b,c=lines.shape
#lines=lines.reshape((a,c))
for x1,y1,x2,y2 in lines[:,0,:]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
cv2.imshow('1',img)
    
