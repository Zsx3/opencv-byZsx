# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:15:58 2019

@author: dell
"""
##读入图像并且显示图像
#import numpy as np
#import cv2
#
#im=cv2.imread('test.png',0)
#cv2.imshow('img',im)
#cv2.waitKey(0)#3000=3s
#cv2.destroyAllWindows()
# =============================================================================
# cv2.waitKey() 是一个键盘绑定函数。需要指出的是它的时间尺度是毫秒级。函数等待特定的几毫秒，看是否有键盘输入。特定的几毫秒之内，如果 按下任意键，这个函数会返回按键的 ASCII 码值，
# 程序将会继续运行。如果没 有键盘输入，返回值为 -1，如果我们设置这个函数的参数为 0，那它将会无限 期的等待键盘输入。
# 它也可以被用来检测特定键是否被按下，例如按键 a是否被按下，这个后面我们会接着讨论。 

# cv2.destroyAllWindows() 可以轻易删除任何我们建立的窗口。如果 你想删除特定的窗口可以使用 cv2.destroyWindow()，在括号内输入你想删 除的窗口名。
# =============================================================================


##图像窗口可调整
#import numpy as np
#import cv2
#img=cv2.imread('test.png',0)
#cv2.namedWindow('img',cv2.WINDOW_NORMAL)#第二个默认为cv2.WINDOW_AUTOSIZE 现在这个NOMAL可以调整窗口大小
#cv2.imshow('img',img)
#cv2.waitKey(0)#3000=3s
#cv2.destroyAllWindows()



###保存图像
#cv2.imwrite('graytest.png',img)


##下面程序将会加载一个灰度图，显示图片，按下's'键后保存后退出，或者按下ESC退出不保存
#import numpy as np
#import cv2
#img=cv2.imread('test.png',0)
#cv2.imshow('123',img)
#k=cv2.waitKey(0)
#if k==27:
#    cv2.destroyAllWindows()
#elif k==ord('s'):
#    cv2.imwrite('zongjie.jpg',img)
#    cv2.destroyAllWindows()


#使用Matplotlib
#import numpy as np
#import cv2
#from matplotlib import pyplot as plt
#
#img=cv2.imread('test.png',1)
#cv2.imshow('123',img)#正常颜色 黄色玉米
#plt.imshow(img)# plt显示cv2载入德图片为蓝色玉米  
## =============================================================================
## 由于OpenCV是以BGR模式加载图像，而matplotlib则是以常见的RGB模式显示图像
##  https://blog.csdn.net/qq_37274615/article/details/79893667  有关于RBG和BGR的转换
## =============================================================================
#plt.xticks([]),plt.yticks([])#隐藏xy标刻
    
