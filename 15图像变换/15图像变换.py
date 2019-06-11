# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:24:43 2019

@author: dell
"""
# =============================================================================
# https://www.cnblogs.com/my-love-is-python/p/10406038.html  各函数说明
# =============================================================================
#傅里叶变换
目标：
使用opencv对图像进行傅里叶变换
使用numpy中的FFT(快速傅里叶变换)函数
傅里叶变换的一些用处
我们将要学习的函数有 cv2.dft(),cv2.idft()等

原理：
傅里叶变换经常被用来分析不同滤波器的频率特性。我们可以使用2d离散傅里叶DFT分析图像的频域特性。
实现DFT的一个快速算法被称为快速傅里叶变换(FFT)。

对于一个正弦信号:x(t)=Asin(2πft)，它的频率为f，如果把这个信号转到它的频域表示，我们会在频率f中
看到一个峰值。如果我们的信号是由采样产生的离散信号组成，我们会得到类似的频谱图，只不过前面是连续的
，后面是离散的。你可以把图像想象成沿两个方向采集信号。所以对图像同时进行X,Y方向的傅里叶变换，我们就会
得到这幅图像的频域表示（频谱图）

更直观一点，对于一个正弦信号，如果它的幅度变化非常快，我们可以说他是高频信号，如果变化非常慢，我们称之为
低频信号。你可以把这种想法应用到图像中，图像在哪里幅度变化非常大呢？边界或噪声，所以我们说边界和噪声是图像
的高频分量（注意这里的高频是指变化非常快，而非出现的次数多）。如果没有如此大的幅度变化，我们称之为低频分量

1.1 numpy中的傅里叶变换(高频滤波（保留高频）)
# =============================================================================
# np.fft.fft2()#转到频域
# np.fft.fftshift()# 频移

# np.fft.ifftshift()#逆平移
# np.fft.ifft2()# 逆变换  得到的是复数
# 取绝对值变成实数
# =============================================================================
    Numpy中的FFT包可以帮我们实现快速傅里叶变换。函数np.fft.fft2()可以对信号进行频率转换
，输出结果是一个复杂的数组。本函数第一个参数是输入图像，要求是灰度格式。第二个参数是可选
的，决定输出数组的大小。输出数组的大小和输入图像大小一样。如果输出结果比输入图像大，输入图像
就需要在进行FFT前补0。如果输出结果比输入结果小的话，输入图像就会被切割
    现在我们得到了结果，频率为0的部分（直流分量）在输出图像的左上角。如果想让它（直流分量）在输出图像的中心，
我们还需要将结果沿两个方向平移N/2。函数np.fft.fftshift()可以帮我们实现这一步，（这样更容易分析）
进行完频率变换后，我们就可以构建振幅谱了

import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('messi.jpg',0)
f=np.fft.fft2(img)#2维傅里叶变换
fshift=np.fft.fftshift(f) #频移 将低频从四个角的移动到中心
magnitude_spectrum=20*np.log(np.abs(fshift))#频移后的图片依旧是复数，用abs，相当于求他的绝对值，a+bj ,sqrt(a^2+b^2),又因为求完绝对值
#数值依然很大，所以取个对数

# =============================================================================
# plt.subplot(121),plt.imshow(img,cmap='gray')
# plt.title('input image'),plt.xticks([]),plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum,cmap='gray')
# plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
# plt.show()
# =============================================================================

#我们可以看到输出结果的中心部分更亮了，说明低频成分分量更多
#现在我们可以进行频域变换了，我们可以在频域对图像进行一些操作了，例如高通滤波和重建图像（DFT逆变换）。比如
#我们可以使用一个60x60矩形左上角了，之后使用函数np.ifft2()进行FFT逆变换。同样又得到一堆复杂的数字，我们对他们
#取绝对值：
rows,cols=img.shape
crow,ccol=rows/2,cols/2
crow=np.uint8(crow)
ccol=np.uint8(ccol)
fshift[crow-30:crow+30,ccol-30:ccol+30]=0
f_ishift=np.fft.ifftshift(fshift)
img_back=np.fft.ifft2(f_ishift)
#取绝对值
img_back=np.abs(img_back)
plt.subplot(131),plt.imshow(img,cmap='gray')
plt.title('input image'),plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(img_back,cmap='gray')
plt.title('image after HPF'),plt.xticks([]),plt.yticks([])#高通滤波的图像
plt.subplot(133),plt.imshow(img_back)
plt.title('result in JET'),plt.xticks([]),plt.yticks([])
plt.show()

#上图的结果显示高通滤波（将低频区域都置为0）其实是一种边缘检测操作，这就是我们在前面像梯度那一章看到的，同时
#我们还发现图像的大部分数据集集中在频谱的低频区域。我们现在已经知道如何使用NUMPY进行DFT和
#IDFT了
#如果你仔细观察。尤其时最后一张jet颜色图像，你会看到一些不自然的东西。看上图那里会有一些带状结构，这被称为
#振铃效应（https://blog.csdn.net/fengye2two/article/details/79895542）（个人理解就是，矩形窗口在频域上作为一个
#高通滤波器，将低频成分置0，其形状类似于一维的矩形方波，在频域上乘以矩形方波等于在空域上卷积一个sinc函数
#（sinc函数如图（振铃效应由来.jpg））所以会出现带状结构，两边的余波将对图像产生振铃现象）所以我们一般不用
#矩形窗口进行滤波，利用傅里叶变换，我们发现，若频域滤波函数具有陡峭变化，则傅里叶逆变换得到的空域滤波函数会在外围出现震荡。
#最好选择高斯窗口

1.2 cv中的傅里叶变换(低频滤波（保留低频）)
# =============================================================================
# cv2.dft()
# np.fft.fftshift()

# np.fft.ifftshift()
# cv2.idft()
# magnitude(img_back[:,:,0],img[:,:,1]) 从复数变成实数
# =============================================================================
openCv中相应的函数时cv2.dft()和cv2.idft()。和前面输出的结果一样，但是双通道的。第一个通道是结果的实数
部分，第二个通道是结果的虚数部分。输入图像首先转换成np.float32格式，我们来看看如何操作

import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('messi.jpg',0)
dft=cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT) 
dft_shift=np.fft.fftshift(dft)
magnitude_spectrum=20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))#cv2.magnitude计算梯度/幅度 等价于np.abs  验证确实如此

#plt.subplot(121),plt.imshow(img,cmap='gray')
#plt.title('input image'),plt.xticks([]),plt.yticks([])
#plt.subplot(122),plt.imshow(magnitude_spectrum,cmap='gray')
#plt.title('M s'),plt.xticks([]),plt.yticks([])
#plt.show()
##这里也可以用cv2.cartToPalar()，它会同时返回幅度和相位
#现在我们来做逆变换DFT，在前面的部分，我们实现了一个HPF（高频滤波），现在我们来做LPF(低通
#滤波)将高频部分去除，其实就是对图像进行模糊操作。首先我们需要构建一个掩码，与低频区域对应的地方设置1
#高频区域对于地方设为0
rows,cols=img.shape
crow,ccol=np.uint8(rows/2),np.uint8(cols/2)
#mask
mask=np.zeros([rows,cols,2],np.uint8)
mask[crow-30:crow+30,ccol-30:ccol+30]=1
fshift=dft_shift*mask
f_ishift=np.fft.ifftshift(fshift)
img_back=cv2.idft(f_ishift)
img_back=cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.imshow(img_back,'gray')

1.3 DFT的性能优化
当数组大小为某些值时，DFT的性能会更好。当数组大小是2的指数时，DFT效率最高，当数组大小
是2，3，5的倍数时，效率也会很高。所以如果想高效率允许代码，可以修改输入图像的大小（补0）
对于opencv必须手动补0，但是numpy，你只需要指定FFT（快速傅里叶变换）运算的大小，它会
自动补0
那我们怎么确定最佳大小呢，opencv提供了一个函数：
cv2.getOptimalDFTSize()，它可以同时被cv2.dft()和np.fft.fft2()使用，让我们使用魔法命令
%time来测试一下

import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('op.png',0)
rows,cols=img.shape
print(rows,cols)#519 759
nrows=cv2.getOptimalDFTSize(rows)
ncols=cv2.getOptimalDFTSize(cols)
print(nrows,ncols)#540 768

nimg=np.zeros((nrows,ncols))
nimg[:rows,:cols]=img #多余的行数补0
cv2.imshow('1',nimg)

%timeit fft1=np.fft.fft2(img)  #94.2ms
%timeit fft2=np.fft.fft2(nimg)  #27.2ms
%timeit dft1=cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)#10.9ms
%timeit dft2=cv2.dft(np.float32(nimg),flags=cv2.DFT_COMPLEX_OUTPUT)#5.92ms  nice~

1.4 为什么拉普拉斯算子是高通滤波器
我在论坛遇到一个类似的问题，为什么拉普拉斯算子算是高通滤波器？为什么sobel是HPF？对于第一个
问题的答案我们以傅里叶变换的形式给出。我们一起来对不同的算子进行傅里叶变换并分析它们

import cv2
import numpy as np
from matplotlib import pyplot as plt

mean_filter=np.ones((3,3))#均值滤波器
x=cv2.getGaussianKernel(5,10)
gaussian=x*x.T

#不同的边缘检测滤波器
scharr=np.array([[-3,0,3],[-10,0,10],[-3,0,3]])
sobel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_y=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

laplacian=np.array([[0,1,0],[1,-4,1],[0,1,0]])

filters=[mean_filter,gaussian,laplacian,sobel_x,sobel_y,scharr]
filter_name=['mean_filter','gaussian','laplacian','sobel_x','sobel_y','scharr_x']

fft_filters=[np.fft.fft2(x) for x in filters]
fft_shift=[np.fft.fftshift(y) for y in fft_filters]
mag_spectrum=[np.log(np.abs(z)+1) for z in fft_shift]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i],cmap='gray')
    plt.title(filter_name[i]),plt.xticks([]),plt.yticks([])
plt.show()
    

