# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:57:46 2019

@author: dell
"""


# 目标：
# 图像梯度，图像边界
# 使用到的函数有 cv2.Sobel(),cv2.Schar(),cv2.Laplacian()
# 原理：
# 梯度简单来说就是求导
# opencv提供了三种不同的梯度过滤器，或者说高通过滤器：Sobel,Schar和Laplacian
# Sobel，Scharr其实就是求一阶或者二阶导数。Scharr是对Sobel（使用小的卷积核求解梯度角度时）的优化
# ，Laplacian是求解二阶导数
# 
# Sobel算子和Scharr算子
# Sobel算子是高斯平滑与微分操作的结合体，所以它的抗噪声能力很好。你可以设定求导方向（xorder或yorder）
# 还可以设定使用的卷积核的大小（ksize）.如果ksize=-1，会使用3X3的Scharr滤波器，他的效果比3x3的Sobel滤波器
# 好（而且速度相同，所以在使用3x3滤波器时，应该尽量使用Scharr滤波器）。3x3的Scharr滤波器卷积核如下：
# x方向【-3  0 3       y方向【-3 -10 -3
#       -10 0 10              0   0  0
#       -3  0 3】             3  10  3】

#Laplacian算子(拉普拉斯算子)
# =============================================================================
# 可以使用二阶导数的形式定义，可假设其离散实现类似于二阶Sobel导数，事实上，opencv在计算拉普拉斯算子
# 时直接调用Sobel算子。计算公式如下
# =============================================================================
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('qipan.jpg',0)
laplacian=cv2.Laplacian(img,cv2.CV_64F)#cv2.CV_64F 输出图像的深度（数据类型），可以使用-1，与原图像保持一致，np.uint8
# =============================================================================
#  通过参数 -1 来设定输出图像的深度（数据类型），可与原图像保持一致，但在代码中使用的是 cv2.CV_64F。这是为什么呢？
#  想象一下一个从黑到白的边界的导数是正数，而一个从白到黑的边界点导数却是负数。如果原图像的深度是np.uint8 时，
#  所有的负值都会被截断变成 0，换句话说就是把边界丢失掉。所以如果想同时检测到这两种边界，最好的办法就是将输出的数据类型设置得更高，
#  比如 cv2.CV_16S， cv2.CV_64F 等。取绝对值然后再把它转回到 cv2.CV_8U。下面的示例演示了输出图片的深度不同造成的不同效果。
# =============================================================================
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) #参数1，0为只在x方向求一阶导数，最大可以求二阶导数
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) #参数0，1为只在y方向求一阶导数，最大可以求二阶导数
plt.subplot(221),plt.imshow(img,'gray'),plt.title('ori')
plt.subplot(222),plt.imshow(laplacian,'gray'),plt.title('lap')
plt.subplot(223),plt.imshow(sobelx,'gray'),plt.title('x')
plt.subplot(224),plt.imshow(sobely,'gray'),plt.title('y')

#Canny 边缘检测
# =============================================================================
# 学习cv2.canny()
# 原理：
# 1.噪声去除，由于边缘检测很容易受到噪声影响，所以第一步是使用5x5的高斯滤波器去除噪声
# 
# 2.计算图像梯度，对平滑后的图像使用sobel算子计算水平方向和竖直方向的一阶导数（图像梯度，Gx
# 和Gy）.根据得到的这两幅梯度图找到边界的梯度和方向：
#             edge_gradient(G)=sqrt(Gy²+Gx²)
#             angle(θ)=arctan(Gy/Gx)
# 梯度的方向一般总与边界垂直，梯度方向被归为四类：垂直，水平，和两个对角线
# 
# 3.非极大值抑制,在获得梯度方向和大小后，应该对整副图像做一个扫描，去除那些非边界上的点
# 对每一个像素进行检查，看这个点的梯度是不是具有相同梯度方向的点中最大的
# 
# 4.滞后阈值（目的是为了把边界连起来），现在要确定那些边界才是真正的边界，这时我们需要设置两个阈值：minval和maxval，当
# 图像的灰度梯度高于maxval时被认为是真正的边界，那些低于minval的边界会被抛弃，如果介于两者之间，
# 就需要看这个点的八邻域内是否存在真正的边界点，如果存在，就认为这个点也是边界点，如果不是，就被抛弃
# 
# =============================================================================
#import cv2
#import numpy as np
#from matplotlib import pyplot as plt
#img=cv2.imread('jishu.png',0)
#edge=cv2.Canny(img,50,200,False) #第一个参数是输入图像，第二个和第三个分别是minval和maxval 第四个参数用来设定卷积核的大小 默认为3，最后一个是L2gradient
##用来设定求梯度大小的方程，如果设为TRUE，就会使用我们上面提到的方程，edge_gradient(G)=sqrt(Gx²+Gy²)，否则用edge_gradient(G)=|Gx|²+|Gy|²代替，默认值为False
#plt.subplot(121),plt.imshow(img,'gray')
#plt.subplot(122),plt.imshow(edge,'gray')
