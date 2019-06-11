# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:37:25 2019

@author: dell
"""

1.1 直方图的计算，绘制并分析
目标：
使用opencv或numpy函数计算直方图
使用opencv或者matplotlib绘制直方图
将要学习的函数有：cv2.calcHist() np.histogram()
原理
什么是直方图呢？通过直方图你可以对整幅图像的灰度发布有一个整体的了解，了解直方图的x轴是灰度值（0到255）,y轴是图片中具有灰度值的点的数目
直方图其实是对图像的另一种解释。以下图为例，通过直方图我们可以对图像的对比度，亮度以及灰度分布有一个直观的认识。几乎所有的图像处理软件都提供了直方图分析功能
要记住直方图是根据灰度图像绘制的，而不是彩色图像。直方图左边的区域显示了暗一点的像素数量，右侧显示了亮一点的像素的数量。

1.2 统计直方图
现在我们知道了什么是直方图，那么怎么样获得一幅图像的直方图呢。首先了解一下相关术语
BINS:
如果你只想知道两像素值之间的像素点数目，举例来说，我们想要知道像素值在0-15之间的像素点的数目，接着是
16-32...... 240到255.我们只需要16个值来绘制直方图，也就是把256个值分成16小组，取每组的总和画直方图，而每一个小组
被称为BIN,我们对这个例子称之有16个BIN.opencv文档中用histSize表示BINS
DIMS:
表示我们收集数据的参数数目，在本例中，我们对收集到的数据只考虑灰度值，所以这里就是 1
RANGE:
就是要统计的灰度值的范围，一般来说为[0,256],也就是所有的灰度值

使用OPNECV统计直方图：函数cv2.calcHist可以帮我们统计一幅图像的直方图
cv2.calcHist(images,channels,mask,histSize,ranges[,hist,[,accumulate]])

images:原图像(图像格式为uint8或者float32)当传入函数时应该用[]括起来，例如：[img]
channels:同样要用中括号括起来，它会告诉函数我们要统计哪幅图像的直方图。如果输入图像是灰度
图，它的值就是[0]；如果是彩色图像的话，传入的参数就可以是[0],[1],[2]他们分别对应着BGR三通道
mask：掩模图像。要统计整幅图像的直方图就把他设为None,但是如果你需要统计图像一部分直方图的话，你就需要
制作一个掩模图像，并且使用它
histSize:BIN的数目，也应该用中括号括起来，例如[256]
range:像素的范围，通常为[0,256]

eg:
import cv2
img=cv2.imread('zhongxin.jpg',0)
hist=cv2.calcHist([img],[0],None,[256],[0,256])


使用numpy统计直方图，np.histogram()也可以帮我们统计直方图。你也可以尝试下下面的代码
hist,bins=np.histogram(img.ravel(),256,[0,256])
*opencv的 函数要比np.histogram快40倍，所以坚持使用opencv函数

1.3 绘制直方图
有两种办法来绘制直方图
1.short Way，使用matplotlib中的绘图函数
2.long way，使用opencv绘图函数

使用matplotlib：matplotlib.pyplot.hist()
eg:
import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('timg.jpg',0)
plt.hist(img.ravel(),256,[0,256]);
plt.show()

同时绘制多通道BGR的直方图，很有用。但是你首先要告诉绘图函数你的直方图数据在哪里。运行一下下面的代码
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('timg.jpg')
color=('b','g','r')
#对一个列表或者数组既要历遍索引又要历遍元素时
#使用内置enumerate函数会更加直接高效
#enumerrate会将数组或列表组成一个索引序列
#使我们再获取索引和索引内容时候更加方便
for i,col in enumerate(color):
    histr=cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color=col)
    plt.xlim([0,256])
plt.show()
    
1.4 使用掩码
要统计图像某个区域的直方图只需要构建一副掩码图像，将要统计的部分设置为白色，其余部分为黑色，就构成来一副掩码图像，然后把这个掩码图像传给函数就可以
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('timg.jpg',0)
mask=np.zeros(img.shape[:2],np.uint8)
mask[200:800,400:1500]=1
masked_img=cv2.bitwise_and(img,img,mask=mask)

hist_full=cv2.calcHist([img],[0],None,[256],[0,256])

hist_mask1=cv2.calcHist([img],[0],mask,[256],[0,256])
plt.plot(hist_full,label='full')
plt.plot(hist_mask1,label='mask')
plt.legend(loc=0,ncol=1)#图例及位置： 1右上角，2 左上角 loc函数可不写 0为最优 ncol为标签有几列
#此处若是不写plt.legend，则不会显示标签
