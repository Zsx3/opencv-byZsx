# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:42:23 2019

@author: dell
"""

目标
1使用模板匹配在一幅图中查找目标
2函数：cv2.matchTemple(),cv2.minMaxLoc()

原理：
模板匹配是用来在一副大图中搜寻模板图像位置的方法，opencv为我们提供了函数，cv2.matchTemple()
。和2D卷积一样，它也是用模板图像在输入图像（大图）上滑动，并在每一个位置对模板图像和其对应的
输入图像的子区域做对比，opencv提供了几种不同的比较方法，返回的结果是一个灰度图像，每一个像素值
表示了此区域与模板匹配程度
如果输入图像大小是（WxH） 模板大小是（wxh）,输出结果大小就是（W-w+1,H-h+1）。当你得到这幅图
后，就可以使用函数cv2.minMaxLoc()来找到其中的最小值和最大值的位置了./在函数完成比较后，
可以使用minMaxLoc（）函数找到最佳匹配作为全局最小值（使用CV_TM_SQDIFF时）或最大值（使用CV_TM_CCORR或CV_TM_CCOEFF时 ）
# 注意：如果使用的比较方法是cv2.TM_SQDIFF,最小值对应的位置才是匹配的区域

opencv中的模板匹配
我们这里有一个例子：我们在梅西的照片中搜索梅西的面部，所以我们要制作下面这样一个模板：
face.jpg 
我们会尝试不用的比较方法，这样我们就可以比较下它们的效果了

import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('messi1.jpg',0)
img2=img.copy()
template=cv2.imread('face.jpg',0)
w,h=template.shape[::-1] #71行58列->58x71 58为宽 71为高

methods=['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']
for meth in methods:
    img=img2.copy()
    method=eval(meth)
# =============================================================================
#     eval是Python的一个内置函数，这个函数的作用是，返回传入字符串的表达式的结果。
#    即变量赋值时，等号右边的表示是写成字符串的格式，返回值就是这个表达式的结果。
# =============================================================================
    res=cv2.matchTemplate(img,template,method)
    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left=min_loc
    else:
        top_left=max_loc   #找对应的左上角的点，然后通过bottom 找到那个框右下角
    bottom_right=(top_left[0]+w,top_left[1]+h)
    cv2.rectangle(img,top_left,bottom_right,255,2)
    plt.figure()
    plt.subplot(121),plt.imshow(res,cmap='gray')
    plt.title('matching result'),plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap='gray')
    plt.title('detected point'),plt.xticks([]),plt.yticks([])
    plt.suptitle(meth)
    
    plt.show()
#cv.TM_CCORR 没找准  其他的都成功找到面部
    
多对象模板匹配
如果匹配对象出现了多次怎么办？函数cv2.imMaxLoc() 只会给出最大值和最小值，此时，我们就要使用阈值了，在
下面的例子中我们要经典游戏Mario的一张截屏图片中找到其中的coin

import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb=cv2.imread('mario.jpg')
img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
template=cv2.imread('coin.jpg',0)
w,h=template.shape[::-1]#

res=cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold=0.8

loc=np.where(res>=threshold) #loc为一个含有两个数组的元祖，第一个数组表示了找到匹配目标的左上角行 第二表示列（其值大于0.8）
for pt in zip(*loc[::-1]):  #之所以要用-1  使得loc元祖上下两行交换位置， 因为loc第一行一开始代表的是匹配坐标对应的行（行代表y），第二行代表所在列（列代表x） 所以
#需要用-1，当第一行变成x，第二行变成y
    cv2.rectangle(img_rgb,pt,(pt[0]+w,pt[1]+h),(0,0,255),1) #左上角坐标
cv2.imshow('1',img_rgb)
 

