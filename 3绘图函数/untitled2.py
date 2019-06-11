# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:05:07 2019

@author: dell
"""
# =============================================================================
# img：你需要绘制图形的那副图像
# color: 形状的颜色，BGR （255，0，0）为蓝色
# thickness: 线条的粗细，如果给一个封闭图像设置为-1，那么图像将会被填充，默认为1
# linetype: 线条的类型，8连接，抗锯齿等，默认情况为8连接。cv2.LINE_AA为抗锯齿，这样看起来会非常平滑
# =============================================================================
##画线
#import numpy as np
#import cv2
#img=np.zeros((512,512,3),np.uint8)
#cv2.line(img,(0,0),(511,511),(0,255,255),5)
#cv2.imshow('123',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


##画矩形
#import numpy as np
#import cv2
#img=np.zeros((512,512,3),np.uint8)
#cv2.rectangle(img,(384,0),(510,128),(0,255,0),-1)#-1填充封闭区域
#cv2.imshow('123',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


##画圆
#import numpy as np
#import cv2
#img=np.zeros((512,512,3),np.uint8)
#cv2.circle(img,(447,63),63,(0,0,255),-1)
#cv2.imshow('123',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#画椭圆
#import numpy as np
#import cv2
#img=np.zeros((512,512,3),np.uint8)
#cv2.ellipse(img,(256,256),(100,50),45,0,360,(0,255,0),1)#(256,256)为中心点坐标 （100，50）
##为长短轴的长度，0椭圆沿逆时针方向旋转的角度，0，180 椭圆弧沿着顺时针方向起始的角度和结束的角度
##如果是0和360就是整个椭圆，上API就是0，180半个椭圆
#cv2.imshow('123',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


##画多边形
#import numpy as np
#import cv2
#img=np.zeros((512,512,3),np.uint8)
#pts=np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
##pts=pts.reshape((-1,1,2)) #可有可无
#cv2.polylines(img,[pts],True,255) #第四个True为闭合，False为起始点和终点没闭合
#cv2.imshow('123',img)


#在图片上添加文字
# =============================================================================
# 参数有：要绘制的文字；绘制的位置；字体类型；字体大小；文字的一般属性如颜色，粗细，线条的类型
# 等，为了更好看一点推荐使用linetype=cv2.LINE_AA
# =============================================================================
#import numpy as np
#import cv2
#img=np.zeros((512,512,3),np.uint8)
#font=cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(img,'OpenCV',(10,500),font,4,(255,255,255),4,cv2.LINE_AA)#文字起始位置（10，500），字体类型font, 字体大小4 ，颜色白色，线条粗细1,cv2.LINE_AA为线条类型
#cv2.imshow('123',img)


#试着绘制OpenCv图标
#import numpy as np
#import cv2
#img=np.ones([512,512,3],np.uint8)*255
#cv2.ellipse(img,(256,120),(60,60),120,0,300,(0,0,255),18)
#cv2.ellipse(img,(181,260),(60,60),0,0,300,(0,255,0),18)
#cv2.ellipse(img,(331,260),(60,60),300,0,300,(255,0,0),18)
#font=cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(img,'OpenCV',(10,450),font,4,(0,0,0),10,cv2.LINE_AA)#文字起始位置（10，500），字体类型font, 字体大小4 ，颜色白色，线条粗细1,cv2.LINE_AA为线条类型
#
#cv2.imshow('img',img)


#鼠标当画笔

#import cv2
#import numpy as np
#def draw_circle(event,x,y,flags,param):
#    if event==cv2.EVENT_MOUSEMOVE: #EVENT_  +  [double click==LBUTTONDBLCL 双击响应]   [LBUTTONDOWN 按下即响应] [LBUTTONUP抬起响应] [MOUSEMOVE一直滑动一直画]
#        cv2.circle(img,(x,y),10,(255,0,0),-1)
#
##创建图像与窗口并将窗口与回调函数绑定
#img=np.zeros([512,512,3],np.uint8)
#cv2.namedWindow('image',0)
#cv2.setMouseCallback('image',draw_circle)
#
#while(1):
#    cv2.imshow('image',img)
#    if cv2.waitKey(20)==27:
#        break
#cv2.destroyAllWindows()



#高级一点的案例(人机交互)
#import cv2
#import numpy as np
#
##当鼠标按下时变成True
#drawing=False
##如果mode为true绘制矩形，按下m变成绘制曲线
#mode=True
#ix,iy=-1,-1
#
##创建回调函数
#def draw_circle(event,x,y,flags,param):
#    global ix,iy,drawing,mode
#    #当按下左键时返回起始位置坐标
#    if event==cv2.EVENT_LBUTTONDOWN:
#        drawing=True
#        ix,iy=x,y
#    elif event==cv2.EVENT_MOUSEMOVE and cv2.EVENT_FLAG_LBUTTON:
#        if drawing==True:
#            if mode==True:
#                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#            else:
#                cv2.circle(img,(x,y),3,(0,0,255),-1)
#    elif event==cv2.EVENT_LBUTTONUP:
#        drawing=False
#
#img=np.zeros([512,512,3],np.uint8)
#cv2.namedWindow('image')
#cv2.setMouseCallback('image',draw_circle)
#while(1):
#    cv2.imshow('image',img)
#    k=cv2.waitKey(1)
#    if k==ord('m'):
#        mode=not mode
#    elif k==27:
#        break
        


