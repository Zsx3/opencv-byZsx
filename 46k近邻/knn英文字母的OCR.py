# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:36:31 2019

@author: dell
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt('letter-recognition.data',dtype='float32',delimiter=',',converters={0:lambda ch:ord(ch)-ord('A')})
train,test=np.vsplit(data,2)
responses,trainData=np.hsplit(train,[1])
labels,testData=np.hsplit(test,[1])

knn=cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
ret,result,neighbours,dist=knn.findNearest(testData,k=5)

correct=np.count_nonzero(result==labels)
accuracy=correct*100/10000
print(accuracy)

