# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:51:05 2019

@author: dell
"""

import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import spline  

import math as mt

import cv2

 

cap = cv2.VideoCapture(0)  #读入视频

c=1

plt.figure(figsize=(8,8),dpi=80) 

aa =[]

bb =[]

cc =[]

#uing = np.logspace(-3,2,121)

while(cap.isOpened()):  

    ret, frame = cap.read() 

    #分解为一帧一帧图像

    if ret == True: 

        #cv2.imshow("frame"，image) 

        img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #彩色转灰度  

       # print(frame)    

        ret,thresh= cv2.threshold(img,127,255,0)   #二值化  

        contours,hierarchy = cv2.findContours(thresh, 3, 1)  

        img = cv2.medianBlur(img,5) #进行中值滤波

 

        cnt = contours[0]   #选取其中的第一个轮廓,这幅图像只有两个轮廓

        M = cv2.moments(cnt)  

        cX=int(M['m10']/M['m00'])   #计算质心

        cY=int(M['m01']/M['m00'])

         

        cv2.drawContours(img,contours,-1,(0,255,0),2)

        cv2.circle(img,(cX,cY),7,(255,255,255),-1)

        cv2.putText(img,"",(cX-20,cY-20),

        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2) 

         

        cv2.imshow("img",img)

        cv2.imwrite('img/'+str(c) + '.jpg',frame) #存储为图像  

         

       # for u in uing:

        aa.append(cX)

        bb.append(cY)

        cc.append(c)

       # plt.plot(c,cX,'k-') 

         

        #plt.plot(c,cX,color='red',linewidth=2.5,linestyle=':')

       # plt.plot(c,cX,'k^') 

        #plt.plot(c,cY,'yo:')

        c = c+1 

               

    else:

          break 

   # cv2.imshow('frame',gray)  #显示标记后的图像q

      

    if cv2.waitKey(1) & 0xFF == ord('q'):  

         break 

     

cap.release()  

cv2.destroyAllWindows() 

 

c1=np.var(aa)

c2=np.var(bb)

 

c1_1=c1/720*2.3*mt.pi/180

c1_2=c2/512*2.3*mt.pi/180

 

print(c1_1)

print(c1_2)

 

plt.plot(cc,aa) 

plt.show()

plt.plot(cc,bb)

plt.show()