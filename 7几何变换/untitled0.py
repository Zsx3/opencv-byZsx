# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:34:41 2019

@author: dell
"""

##拓展缩放
#import cv2
#import numpy as np
#img=cv2.imread('5.jpg')#正常图片是宽度*高度  y*x   600*561   导入为x*y，561*600
##height,width=img.shape[:2]
##res=cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)
##or
#res=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_AREA) #None原本是输出图像尺寸，但因为设置了缩放因子，可以这里是None
## =============================================================================
## https://blog.csdn.net/jningwei/article/details/78822026
## =============================================================================
#while(1):
#    cv2.imshow('res',res)
#    cv2.imshow('img',img)
#    if cv2.waitKey(1)==27:
#        break
#cv2.destroyAllWindows()

##平移
# =============================================================================
# 平移就是对一个对象换一个位置，如果要沿着（x,y）方向移动，移动距离是（tx,ty）,你可以
# 以下面的方式构建移动矩阵
#
#   M=[1 0 tx
#      0 1 ty]
#可以用numpy数组构建这个矩阵，数据类型是np.float32，然后把它传给cv2.warpAffine()
# =============================================================================
#移动（100，50）像素
#import cv2
#import numpy as np
#import matplotlib.pyplot as plt
#
#img = cv2.imread('3.jpg')
#H = np.float32([[1,0,100],[0,1,50]])
#rows,cols = img.shape[:2]
#res = cv2.warpAffine(img,H,(cols,rows)) #需要图像、变换矩阵、变换后的大小
## =============================================================================
## https://blog.csdn.net/qq878594585/article/details/81838260
## =============================================================================
#plt.subplot(121)
#plt.imshow(img)
#plt.subplot(122)
#plt.imshow(res)

#旋转
# =============================================================================
#  对一个图像旋转θ，需要用到下面这个形式的旋转矩阵 
#  M=[cosθ -sinθ
#     sinθ  cosθ]
#  但是openCv允许你在任意地方进行旋转，但是旋转矩阵的形式应该修改为
#   [a β （1-a）*center.x-β*center.y
#   -β a  β*center.x+(1-a)*center.x ]
#   其中 a=scale*cosθ
#        β=scale*sinθ
#为了构建这个旋转矩阵，openCv提供来一个函数
#   cv2.getRotationMatrix2D
#下面是不缩放的情况下，将图像旋转90°
# =============================================================================
#import cv2
#import numpy as np
#img=cv2.imread('3.jpg')
#rows,cols=img.shape[:2]
##构造2维旋转矩阵
#M=cv2.getRotationMatrix2D((cols/2,rows/2),45,1)#第一个参数为旋转中心（注意宽高（1778*1000）用imread导入后边成一个数组（1000行*1778列）），其中心要以1000*1778来计算，即cols/2,rows/2
##第二个参数为旋转角度 #第三个为旋转后的缩放因子
#dst1=cv2.warpAffine(img,M,(cols,rows))
#dst2=cv2.warpAffine(img,M,(2*cols,2*rows))
##第一个为输入图片。第二个为矩阵。第三个为输出的dst的尺寸大小
#while(1):
#    cv2.imshow('1',dst1)
#    cv2.imshow('2',dst2)
#    if cv2.waitKey(1)==27:
#        break
#cv2.destroyAllWindows()


#仿射变换
# =============================================================================
# 在仿射变化中，原图中所有的平行线在结果图像中同样平行，为了创建这个矩阵，我们需要从
#原图中找到三个点以及他们在输出图像中的位置，然后
#   cv2.getAffineTransform会创建一个2*3的矩阵，最后这个矩阵会被传给函数
#   cv2.warpAffine
# =============================================================================
#
#import cv2
#import numpy as np
#from matplotlib import pyplot as plt
#img=cv2.imread('3.jpg')
#rows,cols=img.shape[:2]
#pst1=np.float32([[0,0],[0,rows-1],[cols-1,0]])
#pst2=np.float32([[100,100],[200,900],[1250,100]])
## =============================================================================
## https://www.bilibili.com/video/av28729801/?p=11
## =============================================================================
#M=cv2.getAffineTransform(pst1,pst2)
#
#dst=cv2.warpAffine(img,M,(cols,rows))
#plt.subplot(121),plt.imshow(img),plt.title('input')
#plt.subplot(122),plt.imshow(dst),plt.title('output')

#透视变换
# =============================================================================
# 对于视角变换，我们需要一个3*3变换矩阵，在变换前后直线还是直线，要构建这个变换矩阵，
# 需要在输入图像上找4个点，以及他们在输出图像上对应 的位置，这四个点任意三个点都不能共线
# 这个变换矩阵可以由  cv2.getPerspectiveTransform()构建
# 然后把这个矩阵传给函数cv2.warpPerspective
# =============================================================================
# =============================================================================
# https://www.imooc.com/article/27535  应用实例  
# =============================================================================
# =============================================================================
# =============================================================================
# 哎，搞了一个半小时，一个手动取4个点（左上右上左下右下顺序），然后将二维码摆正或者ppt?
# =============================================================================
import cv2
import numpy as np
from matplotlib import pyplot as plt
global a
a=[]
img = cv2.imread('img2.jpg')
 #print img.shape
 
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
     if event == cv2.EVENT_LBUTTONDOWN:
           xy = "(%d,%d)" % (x, y) #定义xy 为一个字符串“（x，y）”
           print(xy)
           a.append(x)
           a.append(y)
           cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
           cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                       1.0, (0,0,0), thickness = 1)
           cv2.imshow("image", img)
 
cv2.namedWindow("image",0)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
while(True):
   cv2.imshow("image", img)
   if cv2.waitKey(1)==ord('q'):
      break
         
cv2.waitKey(1)
cv2.destroyAllWindows()
pst1=np.array(a)
pst1.resize([4,2])
pst1=np.float32(pst1)

img=cv2.imread('img.jpg')
rows,cols=img.shape[:2]
pst2=np.float32([[0,0],[1036,0],[0,584],[1036,584]])
M=cv2.getPerspectiveTransform(pst1,pst2)  #构建变换矩阵
dst=cv2.warpPerspective(img,M,(cols,rows))
plt.subplot(121),plt.imshow(img),plt.title('input')
plt.subplot(122),plt.imshow(dst),plt.title('output')
cv2.imwrite('changed.jpg',dst)




