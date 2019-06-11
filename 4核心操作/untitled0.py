# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 22:43:10 2019

@author: dell
"""
#import cv2
#import numpy as np
#img=cv2.imread('zhongxin.jpg')
##根据像素的行列获取其像素值，对BGR图像而言，返回为B G R的值，对于灰度图而言，返回的是灰度值（亮度）
#px=img[100,100]
#print(px)
#blue=img[100,100,0]
#print(blue)

# =============================================================================
# [ 73 214 235]
#   73
# =============================================================================
#修改像素值
#import cv2
#import numpy as np
#img=cv2.imread('zhongxin.jpg')
#img[100,100]=[255,255,255]
#cv2.imshow('123',img)

#better way to get a single pixel value
#import cv2
#import numpy as np
#img=cv2.imread('zhongxin.jpg')
#print(img.item(100,100,0))
#img.itemset((100,100,0),100)
#print(img.item(100,100,0))


#获取图属性
#包括 行、列、通道、图像数据类型、像素数目等

##行、列、通道数
#img=cv2.imread('zhongxin.jpg')
#print(img.shape)
#
##像素数目
#print(img.size)
#
##图像的数据类型
#print(img.dtype)

# =============================================================================
# (561, 600, 3)
# 1009800
# uint8
# =============================================================================
# =============================================================================
#在debug时，img.dtype非常重要，因为opencv—python代码中经常出现数据类型不一致的情况 
# =============================================================================


##图像ROI
#import cv2
#import numpy as np
#img=cv2.imread('zhongxin.jpg')
#pic=img[280:340,330:390]  #注意两块区域大小应该一样，否则会报错 60 60 3
#img[273:333,100:160]=pic
#cv2.imshow('123',img)


#拆分及合并图像通道
#有时需要对BGR三个通道分别进行操作，这就需要把BGR分成单个通道，有时需要把独立通道的图片
#合并成一个BGR图像：
# =============================================================================
#  cv2.split()
# =============================================================================
#import cv2
#import numpy as np
#from PIL import *
#img=cv2.imread('zhongxin.jpg')
#b,g,r=cv2.split(img)
#cv2.imshow('blue',b)#出来的是单通道灰度图像
#img=cv2.merge([b,g,r])
#cv2.imshow('new',img)

# =============================================================================
# # https://blog.csdn.net/eric_pycv/article/details/72887758
# #通过创建zeors（img.shape[:2]) 一个单通道全为0的数组，实现单色显示某图（r.g.b）
# 
# import cv2
# import numpy as np
# img=cv2.imread('zhongxin.jpg')
# zeros=np.zeros(img.shape[:2],np.uint8)
# b,g,r=cv2.split(img)
# b=cv2.merge([b,zeros,zeros])
# g=cv2.merge([zeros,g,zeros])
# r=cv2.merge([zeros,zeros,r])
# cv2.imshow('b',b)
# cv2.imshow('g',g)
# cv2.imshow('r',r)
# 
# =============================================================================
#功能同上
#import cv2
#import numpy as np
#img=cv2.imread('zhongxin.jpg')
#img[:,:,:2]=0  #将b,g通道全部置为0，显示图片为红色
#cv2.imshow('r',img)


#为图像扩边
#如果您希望在图像周围创建一个边，就像相框，你可以使用cv2.copyMakeBorder()函数
#这经常在   卷积运算 或 0填充  时被用到，包括如下参数
#src 输入图像
#top bottom left right对应边界的像素数目
#borderType 添加哪种类型的边界，类型如下
#  cv2.BORDER_CONSTANT添加有颜色的常数值边界，还需要下一个参数value
#  cv2.BORDER_REFLECT 边界元素的镜像，比如fedbca|abcdef|hgfedcb
#  cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT 同上，但稍作改动，eg:
#    gfedcb|abcdefgh丨gfedcba
#  cv2.BORDER_REPLICATE 重复最后一个元素，例如 aaaaaa|abcdefgh|hhhhhhh
#  cv2.BORDER_WRAP cdefgh|abcdefgh|abcdefg

#import cv2
#import numpy as np
#from matplotlib import pylab as plt
#BLUE=[255,0,0]
#img1=cv2.imread('logo.jpg')
#replicate=cv2.copyMakeBorder(img1,60,60,60,60,cv2.BORDER_REPLICATE)
#reflect=cv2.copyMakeBorder(img1,60,60,60,60,cv2.BORDER_REFLECT)
#reflect101=cv2.copyMakeBorder(img1,60,60,60,60,cv2.BORDER_REFLECT_101)
#wrap=cv2.copyMakeBorder(img1,60,60,60,60,cv2.BORDER_WRAP)
#constant=cv2.copyMakeBorder(img1,60,60,60,60,cv2.BORDER_CONSTANT)
#
#plt.subplot(231),plt.imshow(img1)
#plt.subplot(232),plt.imshow(replicate,'gray')
#plt.subplot(233),plt.imshow(reflect,'gray')
#plt.subplot(234),plt.imshow(reflect101,'gray')
#plt.subplot(235),plt.imshow(wrap,'gray')
#plt.subplot(236),plt.imshow(constant,'gray')
#plt.show()
#由于是matplotlib 绘制  (rbg)，cv2读入（bgr），所以b,r位置交换了


##图像上的算术运算
# =============================================================================
# cv2.add()   cv2.addWeighted()
# =============================================================================
#图像加法
#opencv的加法是一种饱和操作，numpy加法是一种模操作
#x=np.uint8([250])
#y=np.uint8([10])
#print(cv2.add(x,y))
#print(x+y)
# =============================================================================
# [[255]]    opencv的加法是一种饱和运算，uint8范围是0-255 int范围是-128~127  250+10=260==》255
#   [4]             250+10=260/256=4
#这种差别在对两幅图像进行加法时会更加明显，OpenCV的结果会更好一点，我们尽量使用opencv中的函数
# =============================================================================

#图像混合
#其实也是一种加法，但是不同的是两幅图像的权重不同，这就会给人一种混合或者透明的感觉，公式如下
# g（x）=(1-a)f0(x)+af1(x)
#通过修改a的值，可以实现非常酷的组合
#通过cv2.addWeighted()对图片进行混合操作
#import cv2
#import numpy as np
#img1=cv2.imread('1.jpg')
#img2=cv2.imread('2.jpg')
#dst=cv2.addWeighted(img1,0.7,img2,0.3,0)
#cv2.imshow('dst',dst)
#cv2.waitKey(0)

#按位运算
# AND OR NOT XOR 当我们提取图像的一部分，选择非矩形的ROI时这些操作会很有用（下一章便知）
#我希望将opencv标志放到另外一幅图像。如果使用加法，颜色会改变；如果使用混合，会得到透明效果
#但不希望得到透明效果；如果是矩形，可以用使用ROI（直接img[x:x+w,y:y+h]  )。但如果不是矩形，可以
#通过下面的按位运算实现

import cv2
import numpy as np
img1=cv2.imread('7.jpg')
img2=cv2.imread('logo.jpg')
#希望在5.jpg左上角放logo，所以创建了一个ROI
rows,cols,channels=img2.shape
roi=img1[0:rows,0:cols]
#创建一个logo的掩码
gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret,mask=cv2.threshold(gray,175,255,cv2.THRESH_BINARY)
mask_inv=cv2.bitwise_not(mask)
cv2.imshow('mask',mask)

#cv2.imshow('yanma',mask)
img1_bg=cv2.bitwise_and(roi,roi,mask=mask)  
img1_fg=cv2.bitwise_and(img2,img2,mask=mask_inv)

#将logo放入ROI区域
dst=cv2.add(img1_bg,img1_fg)
img1[0:rows,0:cols]=dst
cv2.imshow('bg',img1_bg)
cv2.imshow('fg',img1_fg)
cv2.imshow('dst',dst)
cv2.imshow('logo+5',img1)


#练习：从一幅图平滑过渡到另外一幅图（cv2.addWeighted）
import cv2
import numpy as np
def nothing(x):
        pass
cv2.namedWindow('image')
cv2.createTrackbar('weight','image',0,100,nothing)#创建轨迹条
img1=cv2.imread('1.jpg')
img2=cv2.imread('2.jpg')
while(1):
    r = cv2.getTrackbarPos('weight', 'image')  #获取当前轨迹条的位置 0-100
    a = float(r)/100.0 #将r变成0-1(分配权值)
    img = cv2.addWeighted(img1,a,img2,1-a,0)
    cv2.imshow('image', img) 
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cv2.imwrite('1+2.jpg',img)
cv2.destroyAllWindows() 

    
    

