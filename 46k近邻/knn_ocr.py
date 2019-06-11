# -*- coding: utf-8 -*-
"""
Created on Sun May 26 21:44:27 2019

@author: dell
"""

#   digits  具有5000个手写数字，每个数字有500种，每个数字是一个20x20的小图，所以第一步就是将这个图像分隔为
#5000个不同的数字，我们在拆分后，每一个数字的图像重排成一行含有400个像素点的新图像。这就是我们的特征集。我们使用
#每个数字的前250个作为训练集，后250个作为测试集，让我们先准备一下

import numpy as np
import cv2
from matplotlib import pyplot as plt

#img=cv2.imread('digits.png')
#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#cells=[np.hsplit(row,100) for row in np.vsplit(gray,50)]
##pic_1=cells[49][1]#第一位【0-49】对应数字0-9  eg 0 ->0  49->9
##cv2.imshow('1',pic_1)
#
#x=np.array(cells)  #转为数组 （50，100，20，20）
## 准备训练数据和测试数据
#train=x[:,:50].reshape(-1,400).astype(np.float32)  #（2500，400）  每250个是一类数字
#test=x[:,50:100].reshape(-1,400).astype(np.float32)
##创建标签
#k=np.arange(10)
#train_labels=np.repeat(k,250)[:,np.newaxis]###################一个不错的用法，将一维数组（长度为a）变成2维[:,np.newaxis]是变成（a,1）;
##[np.newaxis,:]是变成(1,a)   np.repeat 是重复数字 eg (1,2,3) (1,1,2,2,3,3)   
#test_labels=train_labels.copy()  #（2500，1）
#
#knn=cv2.ml.KNearest_create()
#knn.train(train,cv2.ml.ROW_SAMPLE,train_labels) 
#ret,result,neighbours,dist=knn.findNearest(test,k=5)  #result=(2500,1)
##确认准确度
#matchs=result==test_labels #match 为bool量
#correct=np.count_nonzero(matchs) #统计不为0的个数
#accuracy=correct*100/result.size
#print(accuracy)
##测试自己的数据
#test_5=cv2.imread('5.png',0)
#test_5=cv2.resize(test_5,(20,20)).reshape(-1,400).astype(np.float32)
##cv2.imshow('5',test_5)
#ret_1,result_1,neighbours_1,dist_1=knn.findNearest(test_5,k=5)  










# =============================================================================
# #保留训练好的分类器
# =============================================================================
#np.savez('knn_data.npz',train=train,train_labels=train_labels)
#now  load data
with np.load('knn_data.npz')  as data:
    print(data.files)
    train=data['train']
    train_labels=data['train_labels']
test_5=cv2.imread('5.png',0)
test_5=cv2.resize(test_5,(20,20)).reshape(-1,400).astype(np.float32)
#cv2.imshow('5',test_5)
knn=cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels) 
#ret,result,neighbours,dist=knn.findNearest(test,k=5)  #result=(2500,1)
ret_1,result_1,neighbours_1,dist_1=knn.findNearest(test_5,k=5)  