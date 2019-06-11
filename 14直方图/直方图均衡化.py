# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:39:07 2019

@author: dell
"""

目标：
本小节我们要学习直方图均衡化的概念，以及如何使用它来改善图片的对比
原理：
想象一下如果一副图像的大多像素点的像素值都集中在一个像素值的范围之内会怎么样？
例如，如果一副图像整体很亮，那所有像素值都应该很高，但是一副高质量的图像，像素值分布应该很
广泛，所以你应该把它的直方图做一个横向拉伸。这就是直方图均衡化要做的事情，


# =============================================================================
通常这种操作会改变图像的对比度
# =============================================================================

import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('wiki.jpg',0)
hist,bins=np.histogram(img.flatten(),256,[0,256])
#计算累计分布图
cdf=hist.cumsum()
cdf_normalized=cdf*hist.max()/cdf.max()  #归一化

plt.plot(cdf_normalized,color='b') 
plt.hist(img.flatten(),256,[0,256],color='r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'),loc='upper left')
plt.show()
#
#我们可以看出直方图比较集中，我们希望直方图分散能够覆盖整个x轴。所以，我们就需要一个变换函数
#帮助我们把现在的直方图映射到一个广泛分布的直方图中，这就是直方图均衡化要做的事

#构建Numpy掩码数组，cdf为原数组
# =============================================================================
# 在许多情况下，数据集可能不完整或因无效数据的存在而受到污染。例如，传感器可能无法记录数据或记录无效值。该numpy.ma模块通过引入掩码数组提供了一种解决此问题的便捷方法。

# 屏蔽数组是标准numpy.ndarray和掩码的组合。
# =============================================================================
cdf_m=np.ma.masked_equal(cdf,0) #屏蔽一个等于给定值0的数组,将0屏蔽，避免污染
cdf_m1=(cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min()) #归一化 ，cdf_m-cdf_m.min()/max-min范围为（0，1）,乘以255，范围变成（0，255）  #被屏蔽数组不参与计算
cdf_n=np.ma.filled(cdf_m1,0).astype('uint8') #将输入作为数组返回，屏蔽数据替换为填充值0。
plt.figure()
plt.plot(cdf_n,color='g')
img2=cdf_n[img] #img数组中的数字(范围0-255)作为索引 在cdf_n 寻找对应数值  填入img2
cv2.imshow('1',img2)
# =============================================================================
#  网上代码
# =============================================================================
import cv2
import numpy as np
 
image = cv2.imread("wiki.jpg", 0)
#image=cv2.imread('st.png',0)#雕像确实丢失了细节
 
lut = np.zeros(256, dtype = image.dtype )#创建空的查找表
 
hist,bins = np.histogram(image.flatten(),256,[0,256]) 
cdf = hist.cumsum() #计算累积直方图
cdf_m = np.ma.masked_equal(cdf,0) #除去直方图中的0值
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())#等同于前面介绍的lut[i] = int(255.0 *p[i])公式
cdf = np.ma.filled(cdf_m,0).astype('uint8') #将掩模处理掉的元素补为0
 
#计算
result2 = cdf[image]
result = cv2.LUT(image, cdf)
 
cv2.imshow("OpenCVLUT", result)
cv2.imshow("NumPyLUT", result2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#自适应局部均衡化
　　的确在进行完直方图均衡化之后，图片背景的对比度被改变了。但是你再对比一下两幅图像中雕像的面图，
由于太亮我们丢失了很多信息。造成这种结果的根本原因在于这幅图像的直方图并不是集中在某一个区域（试着画出它的直方图，你就明白了）。
    为了解决这个问题，我们需要使用自适应的直方图均衡化。这种情况下，整幅图像会被分成很多小块，这些小块被称为“tiles”
（在 OpenCV 中 tiles 的大小默认是 8x8），然后再对每一个小块分别进行直方图均衡化（跟前面类似）。
所以在每一个的区域中，直方图会集中在某一个小的区域中（除非有噪声干扰）。如果有噪声的话，噪声会被放大。
为了避免这种情况的出现要使用对比度限制。对于每个小块来说，如果直方图中的 bin 超过对比度的上限的话，
就把其中的像素点均匀分散到其他 bins 中，然后在进行直方图均衡化。最后，为了去除每一个小块之间“人造的”
（由于算法造成）边界，再使用双线性差值，对小块进行缝合。
import numpy as np
import cv2
img = cv2.imread('test_6.jpg')
# create a CLAHE object (Arguments are optional).
# 不知道为什么我没好到 createCLAHE 这个模块
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl=np.zeros(img.shape,np.uint8)
for i in range(img.shape[2]):
    cl[:,:,i] = clahe.apply(img[:,:,i])
    
cv2.imshow('1',cl)
#cv2.imwrite('clahe_2.jpg',cl1)
