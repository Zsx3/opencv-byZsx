# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:17:52 2019

@author: dell
"""

#学习使用不同的低通滤波器对图像进行模糊
#使用自定义的滤波器来图像进行卷积（2D卷积）

#2D卷积
# =============================================================================
# 
# 低通滤波可以去除噪声，模糊图像
# 高通滤波可以帮我们找到图像边缘
#  
# opencv提供的函数cv2.filter2D()可以让我们对一副图像进行卷积操作，下面我们对一副图像使
# 平均滤波器，下面是一个5x5的平均滤波器核
# 
# 
# K=np.ones([5,5])*(1/25)
# 操作如下：将核放在图像的一个像素A上，求与核对应的图像上25（5x5）个像素的和，再取平均数
# 用这个平均数代替像素A的值.重复以上操作看到将图像的每一个像素都更新一边，代码如下
# =============================================================================
#import cv2
#import numpy as np
#from matplotlib import pylab as plt
#img=cv2.imread('zhongxin.jpg')
#kernel=np.ones([5,5],np.float32)/25
#dst=cv2.filter2D(img,-1,kernel) #(src,dst,kernel,anchor=(-1,1))  dst若为-1  则和原尺存一样
#
#plt.subplot(121),plt.imshow(img),plt.title('ori')
#plt.subplot(122),plt.imshow(dst),plt.title('averaging')
#plt.show()

#图像模糊（图像平滑）
#使用低通滤波可以达到图像模糊的目的，这对去除噪声很有帮助，其实就是去除图像中的高频成分，
# 所以边界也会被模糊一点（当然，也有一些模糊技术不会模糊边界），opencv提供了四种模糊技术

##1 平均
# =============================================================================
# K=np.ones([3,3])/9
# 可以用 函数 cv2.blur()  和 cv2.boxFilter()来完成这个任务
# 
# 如果你不想要使用归一化卷积框，你应该使用cv2.boxFilter(),这时要传入参数normalize=False

#解释：均值滤波是典型的线性滤波算法，它是指在图像上对目标像素给一个模板，该模板包括了其周
#围的临近像素（以目标像素为中心的周围8个像素，构成一个滤波模板，即去掉目标像素本身），再
#用模板中的全体像素的平均值来代替原来像素值。
#
#缺点：由于图像边框上的像素无法被模板覆盖，不做所以处理 
#这当然造成了图像边缘的缺失 

# =============================================================================
#import cv2
#import numpy as np
#from matplotlib import pylab as plt
#img=cv2.imread('zhongxin.jpg')
#blur=cv2.blur(img,(5,5))
#plt.subplot(121),plt.imshow(img),plt.title('ori')
#plt.subplot(122),plt.imshow(blur),plt.title('blurred')
#plt.show()

##2 高斯模糊
# =============================================================================
# 现在把卷积核换成高斯核（简单说，方框不变，将原来每个方框的值是相等的，现在里面的值是符合高斯分布的，
# 方框中心的值最大， 其余方框根据距离中心元素的距离递减，构成一个高斯小山包。原来的求平均数现在变成求加权平均数
# ，这就是全方框的值）。实现的函数的cv2.GaussianBlur()，我们必须指定高斯核的宽和高(必须是奇数)，以及高斯函数
# 沿着X Y方向的标准差。如果我们只指定了X方向的标准差，Y方向也会取相同值，如果两个标准差都是0，那么函数会根据核函数
# 的大小，自己计算。高斯滤波可以有效的从图像中去除高斯噪声
# =============================================================================
#import cv2
#import numpy as np
#from matplotlib import pylab as plt
#img=cv2.imread('zhongxin.jpg')
#blur1=cv2.blur(img,(5,5))
#blur2=cv2.GaussianBlur(img,(5,5),0)
#plt.subplot(131),plt.imshow(img),plt.title('ori')
#plt.subplot(132),plt.imshow(blur1),plt.title('blurred')
#plt.subplot(133),plt.imshow(blur2),plt.title('gauss')
#plt.show()


##3 中值模糊
# =============================================================================
#  解释：中值，中间值，将数据从小到大排序后的中间值，用3×3大小模板进行中值滤波。对模板
#  中的9个数进行从小到大排序：1,1,1,2,2,5,6,6,10。中间值为2所有，中值滤波后（2,2）
#  位置的值变为2.同理对其他像素点。 
# =============================================================================
#import cv2
#import numpy as np
#from matplotlib import pylab as plt
#img=cv2.imread('zhongxin.jpg')
#blur1=cv2.blur(img,(5,5))
#blur2=cv2.GaussianBlur(img,(5,5),0)
#med=cv2.medianBlur(img,5) #中值
#plt.subplot(221),plt.imshow(img),plt.title('ori')
#plt.subplot(222),plt.imshow(blur1),plt.title('blurred')
#plt.subplot(223),plt.imshow(blur2),plt.title('gauss')
#plt.subplot(224),plt.imshow(med),plt.title('med')
#plt.show()

##4 双边滤波                                       #233 美颜效果不错♪(＾∀＾●)ﾉ
# =============================================================================
# 函数cv2.bilateraFilter()能在保持边界清晰的情况下有效去除噪声，但这种操作与其他滤波器相比
# 会比较慢，我们已经知道高斯滤波器是求中心点邻近区像素的高斯加权平均值。这种高斯滤波器只能
# 考虑像素之间的空间关系，而不会考虑像素之间的关系（像素的相似度），所以这种方法不会考虑到
# 一个像素是否位于边界，因此边界也会被模糊掉，而这不是我们想要的
# 
# 双边滤波在同时使用空间高斯权值和灰度值相似高斯权值。空间高斯函数确保只有邻近区域的像素
# 对中心点有影响，灰度相似性高斯函数确保只有与中心像素灰度值相近的才会被用来做模糊运算，（
# 边界灰度变化大），所以可以确保边界不会被模糊掉
# =============================================================================
#import cv2
#import numpy as np
#from matplotlib import pylab as plt
#img=cv2.imread('meiyan.jpg')
#img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#blur1=cv2.blur(img2,(5,5))
#blur2=cv2.bilateralFilter(img2,9,75,75) #9邻域直径 两个75分别是空间高斯函数标准差、灰度值相似性高斯标准差
#plt.subplot(121),plt.imshow(img2),plt.title('yuantu')
#plt.subplot(122),plt.imshow(blur2),plt.title('shuangbian')
