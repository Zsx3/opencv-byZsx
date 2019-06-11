# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:51:48 2019

@author: dell
"""
#使用SVM进行手写数据OCR
# =============================================================================
# 本节我们还是进行手写数据的OCR,但这次我们使用的是SVM而不是KNN
# 
# 之前KNN使用的是像素的灰度值作为特征向量，这次我们使用方向直方图HOG作为特征向量
# 在计算HOG前，我们使用图片的二阶矩对图片进行 抗扭斜 处理， 所以我们首先要定义一个函数
# deskew() ,它可以对图像进行抗扭斜处理，下面就是deskew()函数
# =============================================================================
import cv2
import numpy as np
def deskew(img):
    m=cv2.moments(img)
    if abs(m['mu02']<1e-2):
        return img.copy()
    skew=m['mu11']/m['mu02']
    M=np.float32([[1,skew,-0.5*SZ*skew],[0,1,0]])
    img=cv2.warpAffine(img,M,(SZ,SZ),flags=affine_flags)
#将图像分为4个小方块（每个10x10），对每个小方块计算他的朝向直方图（16个bin《=》16个朝向）,使用梯度的大小做权重
#这样每个小方块都会得到一个含有16个成员的向量，4个小方块的4个向量就组成了这个图像的特征向量（包含64个成员）

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist


if __name__=='__main__':
    SZ=20
    bin_n=16 #num of bins
    img=cv2.imread('digits.png',0)
    affine_flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
    cells=[np.hsplit(row,100) for row in np.vsplit(img,50)]
    
    train_cells=[i[:50] for i in cells]
    teset_cells=[i[50:] for i in cells]
    
    #train now
    deskwed