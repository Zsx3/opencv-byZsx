# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:59:41 2019

@author: dell
"""

# 图像特征提取与描述

#Harris 角点检测
#学习函数：cv2.cornerHarris(),cv2.cornerSubPix()

# =============================================================================
# cv2.cornerHarris()参数如下：
# img           数据类型为float32的图像
# blockSize     角点检测中要考虑的邻域大小
# ksize-sobel   求导中使用的窗口大小
# k             Harris中的自由参数，取值参数为【0.04-0.06】。  R=det(M)-k(trace(M))^2
# =============================================================================

#import cv2
#import numpy as np
#filename="zhongxin.jpg"
#img=cv2.imread(filename)#uint8
#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#uint8
#gray=np.float32(gray);#uint8->float32
#
#dst=cv2.cornerHarris(gray,2,3,0.04)
##cv2.imshow('1',dst)
#
##img[dst>dst.max()*0.01]=[0,0,255]
#point=[]
#point=dst>dst.max()*0.01 # 把角点存储在point中--point中true的为角点
#rows,cols=np.shape(gray)
#for i in range(rows):#行--y
#    for j in range(cols):#列--x
#        if(point[i,j]==True):
#           cv2.circle(img,(j,i),2,(0,0,255),2)
#cv2.imshow("2",img)

# =============================================================================
#
#  import cv2
#  import numpy as np
#  capture=cv2.VideoCapture(0)
#  while(1):
#      ret,img=capture.read()
#      gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#uint8
#      gray=np.float32(gray);#uint8->float32
#      dst=cv2.cornerHarris(gray,2,3,0.04)
#  #cv2.imshow('1',dst)
#  
#  #img[dst>dst.max()*0.01]=[0,0,255]
#      point=[]
#      point=dst>dst.max()*0.1 # 把角点存储在point中--point中true的为角点
#      rows,cols=np.shape(gray)
#      for i in range(rows):#行--y
#          for j in range(cols):#列--x
#              if(point[i,j]==True):
#                  cv2.circle(img,(j,i),2,(0,0,255),2)
#      cv2.imshow("2",img)
#      k=cv2.waitKey(20)
#      if k==27:
#           break
#  capture.release()
#  cv2.destroyWindow("2")

# =============================================================================
#亚像素级精度的角点
# =============================================================================
# 有时候我们需要最大精度的角点检测，opencv为我们提供了函数cv2.cornerSubPix().它可以
# 提供亚像素级的角点检测，下面是一个例子，首先找到Harris角点，然后将角点的重心传给这个
# 函数进行修正。Harris角点用红色像素标出，绿色像素是修正后的像素，在使用这个函数时我们要定义
# 一个迭代停止条件，当迭代次数达到或者精度条件满足后就会停止迭代。我们同样需要定义进行角点
# 搜索的邻域大小
# =============================================================================


#import cv2
#import numpy as np
#filename="zhongxin.jpg"
#
#img=cv2.imread(filename)#uint8
##cv2.imshow('1',img)
#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#uint8
#gray=np.float32(gray);#uint8->float32
#
#dst=cv2.cornerHarris(gray,2,3,0.04)
#dst=cv2.dilate(dst,None)
#ret,threshold=cv2.threshold(dst,0.01*dst.max(),255,0)
#
#dst=np.uint8(dst)
#ret,labels,stats,centroids=cv2.connectedComponentsWithStats(dst)
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
#corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
#res = np.hstack((centroids,corners)) 
#res = np.int0(res) 
#img[res[:,1],res[:,0]]=[0,0,255] 
#img[res[:,3],res[:,2]] = [0,255,0]
#cv2.imwrite('subpixel5.png',img)

#Shi-Tomasi 角点检测&  适合于跟踪图像特征
对Harris的打分函数做了修改，harris是R=λ1λ2-k(λ1+λ2)²
但shi-Tomasi的打分函数为：
R=min(λ1，λ2)

如果打分超过阈值，我们就认为它是一个角点，我们可以把它绘制到λ1-λ2空间中，如图文件夹中tomasi
从这幅图中，我们可以看出，只有当λ1和λ2都大于最小值时，才被认为是角点

opencv提供了cv2.goodFeaturesToTrack() 这个函数可以帮我们使用shi-tomasi方法获取图像中N个
最好的角点（如果你愿意的话，也可以通过改变参数来使用Harris角点检测算法）。通常情况下，输入的
应该是灰度图像。然后确定你想要检测的 角点数目 。再设置 角点的质量水平（0-1之间）。它代表了角点
的最低质量，低于这个数的所有角点都会被忽略。最后再设置 两个角点之间的最短欧式距离 。 

根据这些信息，函数就能在图像上找到角点。所有低于质量水平的角点都会被忽略。然后再把合格角点按
角点质量进行降序排列，函数会采用角点质量最高的那个点（排序后的第一个），然后将它附件（最小距离之内）
的角点都删掉。按着这样的方式返回N个最佳角点。

import cv2
import numpy as np
from matplotlib import pyplot as plt
capture=cv2.VideoCapture(0)
while(1):
     ret,img=capture.read()
     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#uint8

     corners=cv2.goodFeaturesToTrack(gray,10,0.01,10)#float64
     corners_new=np.int0(corners)#int32
     for i in corners:
         #这里之所以要平铺，因为i返回的是角点是【【600，100】】  含有两层括号的矩阵（2，1）矩阵，所以flatten或者ravel后，变成【600，100】然后赋值给x，y
         x,y=i.ravel()
         cv2.circle(img,(x,y),2,(0,0,255),-1)
     cv2.imshow("2",img)
     k=cv2.waitKey(20)
     if k==27:
          break
capture.release()
cv2.destroyAllWindows()

