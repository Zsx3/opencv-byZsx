# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:58:49 2019

@author: dell
"""

#import numpy as np
#import cv2
#http="http://admin:admin@192.168.1.113:8081/"
#cap=cv2.VideoCapture(http)
#
#while(True):
#    ret,frame=cap.read()
#    
#    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#    cv2.imshow('123',gray)
#    k=cv2.waitKey(1)#waitKey（）里面不能是0,否则会一直卡在第一张图片，等待按键
#    if k==27:
#        break
#cap.release()
#cv2.destroyAllWindows()


#通过摄像头采集视频，并保存至output。avi
# =============================================================================
#  重点API  fourcc=cv2.VideoWriter_fourcc(*'XVID') 
#           out=cv2.VideoWriter('name',编解码器,播放帧率（影响播放速度）,(1280,640输出视频尺寸))
#           out.write(frame)  #向创建的视频流写入一帧
#           out.release()
# =============================================================================
#import numpy as np
#import cv2
#http="http://admin:admin@192.168.1.113:8081/"
#cap=cv2.VideoCapture(http)
#
#fourcc=cv2.VideoWriter_fourcc(*'XVID') #fourcc为视频 编解码器
#out=cv2.VideoWriter('output.avi',fourcc,20,(1280,640))#20为帧播放速率 设置过高视频播放很快
##创建视频流写入对象，VideoWriter_fourcc为视频编解码器，20为帧播放速率，（640，480）为视频帧大小
#
#while(cap.isOpened()):
#    ret,frame=cap.read()                       ##从摄像头读取一帧
#    if ret==True:
##       frame=cv2.flip(frame,1)                ##图像翻转  0 垂直翻转 1 水平翻转 -1 水平垂直翻转
#       out.write(frame)                        ##向视频文件写入一帧
#       cv2.imshow('frame',frame)
#       if cv2.waitKey(1)==ord('q'):
#           break
#    else:
#        break
#        print('无法正常读取摄像头')
#cap.release()
#out.release()
#cv2.destroyAllWindows()

# =============================================================================
# 自写demo
# =============================================================================
#import numpy as np
#import cv2
#http="http://admin:admin@192.168.1.113:8081/"
#cap=cv2.VideoCapture(http)
#
#fourcc=cv2.VideoWriter_fourcc(*'XVID')
#out=cv2.VideoWriter('demo.avi',fourcc,25,(1280,640))
#
#while (True):
#    ret,frame=cap.read()
#    if ret==True:
#        out.write(frame)
#        cv2.imshow('frame',frame)
#        if cv2.waitKey(10)==ord('q'):
#            break
#    else:
#        break
#cap.release()
#out.release()
#cv2.destroyAllWindows()
    