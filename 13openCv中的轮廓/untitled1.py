# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 22:07:15 2019

@author: dell
"""

# =============================================================================
# 目标：
# 理解什么是轮廓
# 学习找轮廓，绘制轮廓
# 函数：cv2.findContours(),cv2.drawContours()
# 
# 1.1 什么是轮廓
#     轮廓可以简单认为成将连续的点（连着边界）连在一起的曲线，具有相同的颜色或者灰度，轮廓
# 形状分析和物体检测和识别中很有用
# *为了更加准确，要使用二值化图像，在寻找轮廓前，要进行阈值化处理或者canny边缘检测
# *查找轮廓的函数会修改原始图像，如果你在找到轮廓后还想使用原始图像的话，你应该将原始图像
# 储存到其他变量中
# *在opencv中，查找轮廓就像在黑色背景中寻找超白色物体，你应该记住，要找的物体应该是白色，而背景应该是黑色
# 
# 让我们看看如何在一个二值图像中查到轮廓
# 函数cv2.findContours()有三个参数，第一个是输入图像，第二个是轮廓检索模式，第三个是轮廓近似方法
# =============================================================================
# cv2.RETR_EXTERNAL表示只检测外轮廓
# cv2.RETR_LIST检测的轮廓不建立等级关系
# cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
# cv2.RETR_TREE建立一个等级树结构的轮廓。


#cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
#cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
# =============================================================================
# 返回值也有三个，第一个是图像（python3只有两个返回值），第二个是轮廓，第三个是（轮廓的）层析结构（？？）。第二个返回值（轮廓）
# 是一个python列表，其中储存这图像中所有的轮廓，每一个轮廓都是一个Numpy 数组，包含对象边界点（x，y）坐标
# 注意：我们后面会对第二个和第三个参数，以及层次结构进行详细介绍，在那之前，例子中使用的参数值对所有图像都是
# 使用的
# 1.2 怎么样绘制轮廓
# cv2.drawContours()可以用来绘制轮廓，它可以根据你提供的边界点绘制任何形状，它的第一个参数是原始图像，第二
# 个参数是轮廓，一个python列表，第三个参数是轮廓的索引（在绘制独立轮廓是很有用的，当设置为-1时绘制所有轮廓）
# 接下来的参数是轮廓的颜色和厚度等
# 在一副图像中绘制所有的轮廓：
# =============================================================================
import cv2
import numpy as np
img=cv2.imread('1.jpg')
#img=cv2.pyrDown(img)
#img=cv2.pyrDown(img)#图太大啦可以用这个
img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(img1,127,255,0)
contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#
img2=cv2.drawContours(img,contours,-1,(255,255,0),4) #当地第三个 为-1 时（此时2是绘制第3个轮廓，从0 开始算），绘制所有轮廓，最后两个分别是轮廓线的颜色和粗细
cv2.imshow('123',img2)


## 勉勉强强找到图像边框 待改进
import cv2
import numpy as np
img=cv2.imread('4.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',gray)
ret,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('thresh',thresh)
kernel=np.ones([9,9],np.uint8)
erosion=cv2.erode(thresh,kernel,iterations=4)
dilation=cv2.dilate(thresh,kernel,iterations=4)
#cv2.imshow('final',dilation)
contours,hierachy=cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
c=sorted(contours,key=cv2.contourArea,reverse=True)[0]#找
rect=cv2.minAreaRect(c)                               #框
box_p=cv2.boxPoints(rect)                             #四
box_p=np.int0(box_p)                                  #连
cv2.drawContours(img,[box_p],-1,(255,0,0),4)
cv2.imshow('ai',img)

# =============================================================================
# 使用python opencv返回点集cnt的最小外接矩形，所用函数为 cv2.minAreaRect(cnt) ，cnt是所要求最小外接矩形的点集数组或向量，这个点集不定个数。
# 举例说明：画一个任意四边形的最小外接矩形，其中 cnt 代表该四边形的4个顶点坐标（点集里面有4个点）
# cnt = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) # 必须是array数组的形式
# rect = cv2.minAreaRect(cnt) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
# box = cv2.cv.BoxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点
# box = np.int0(box)
# =============================================================================
## 轮廓特征
目标：
查找轮廓的不同特征，例如面积、周长、重心、边界框等
1.1矩
图像的矩可以帮我们计算图像的质心，面积等。详细学习请查看维基百科Image Moments
cv2.moments()会计算得到的矩以一个字典的形式返回，如下：
import cv2
import numpy as np
img=cv2.imread('ju.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,127,255,0)
contours,hierarchy=cv2.findContours(thresh,1,2)
cv2.drawContours(img,contours,-1,(255,0,0),4)
cv2.imshow('123',img)
cnt=contours[0] ##矩形的轮廓点集合
M=cv2.moments(cnt)
print(M)
cx=int(M['m10']/M['m00'])
cy=int(M['m01']/M['m00']) #根据这些矩的值，可以计算出对象的重心
cv2.circle(img,(cx,cy),2,(0,0,255),2)
cv2.imshow('img',img)

1.2轮廓面积
轮廓的面积可以由函数cv2.contourArea()计算得到，也可以使用矩（0阶矩），M['m00']
area=cv2.contourArea(cnt)  #18625  与m00一样

1.3轮廓周长
可以使用函数cv2.arcLength()计算得到，这个函数的第二参数可以用来指定对象的形状是闭合的（True）还是打开的（一条曲线）
perimeter=cv2.arcLength(cnt,True)  #c=544.8 s=(c/2)^2=18496 与面积相符
1.4轮廓近似
将轮廓形状近似到另一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定，使用的Douglas-Peucker算法
为了帮助理解，假设我们要在一副图像中查找一个矩形，但是由于图像的种种原因，我们不能得到一个完美的矩形，而是一个坏形状
现在你就可以使用这个函数来近似这个形状。这个函数的第二个参数叫epsilon,它是从原始轮廓到近似轮廓的最大距离。它是一个准确度
的参数。选择一个好的epsilon对于得到一个满意的结果非常重要
import cv2
import numpy as np
img=cv2.imread('jinsi.png')
#img=cv2.imread('ploygon.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,127,255,0)
contours,hierarchy=cv2.findContours(thresh,1,2)
cnt=contours[0]
epsilon=0.01*cv2.arcLength(cnt,True) #它是从原始轮廓到近似轮廓的最大距离。它是一个准确度的参数。
approx=cv2.approxPolyDP(cnt,epsilon,True)#0.1 approx返回就是矩形的四个点的坐标
cv2.polylines(img, [approx], True, (0, 0, 255), 4) #画多边形
cv2.imshow('approx',img)

1.5凸包
凸包和轮廓近似类似，但不同，虽然有些情况下他们会给出的结果一样，函数cv2.convexHull()用来检测一个
曲线是否具有凸性缺陷，并能纠正缺陷。一般来说，凸性曲线总是凸出来的，至少是平的。如果有些地方凹进去
了就被叫做凸性缺陷。例如下面中的手，红色曲线显示了手的凸包，凸性缺陷被双箭头标出来了
参数：
points 我们要传入的轮廓
hull 输出，通常不需要
clockwise 方向标志。如果设置为True,输出的凸包是顺时针方向的，否则是逆时针方向的
returnPoints 默认为True,他会返回凸包上的坐标。如果设置为False它就会返回与凸包点对应轮廓的点（轮廓点的索引）
## 方法1  通过轮廓面积大小 找到手势的两个特征轮廓
import cv2
import numpy as np
#img=cv2.imread('ploygon.png')#多边形
img=cv2.imread('shoushi.jpg')#手势
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,235,255,0)
contours,hierarcy=cv2.findContours(thresh,1,2)
c=sorted(contours,key=cv2.contourArea,reverse=True)[1]#找出最大面积的轮廓，这图最大的是图片边框0，所以只能设为1
c1=sorted(contours,key=cv2.contourArea,reverse=True)[2]
#cnt=contours[4]
hull=cv2.convexHull(c)
hull1=cv2.convexHull(c1)
cv2.polylines(img,[hull],True,(0,0,255),4)
cv2.polylines(img,[hull1],True,(0,0,255),4)
cv2.imshow('hull',img)
## 方法2  通过轮廓点的个数>5 找到手势的两个特征轮廓(推荐)
import cv2
import numpy as np
#img=cv2.imread('ploygon.png')#多边形
img=cv2.imread('shoushi.jpg')#手势
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,235,255,0)
contours,hierarcy=cv2.findContours(thresh,1,2)
for cnt in contours:
    hull=cv2.convexHull(cnt)
    length=len(hull)
    if length>5:
        cv2.polylines(img,[hull],True,(0,0,255),3)
cv2.imshow('hull',img)

1.6凸性检测
函数cv2.isContourConvex()可以用来检测一个曲线是否为凸的，它只能返回True或False，没什么用

k=cv2.isContourConvex(c1) #ploygon是false

1.7边界矩形
有两类边界矩形
直角界边界矩形：一个直矩形（就是没有旋转的矩形），它不会考虑对象是否旋转。所以边界矩形的面积不是最小的。可以使用函数cv2.boundingRect()查找得到
（x,y）为矩形左上角的坐标，（w,h）是矩形的宽和高
旋转边界矩形：这个边界矩形是面积最小的，因为它考虑了对象的旋转。用到的函数为cv2.minAreaRect().返回的是一个Box2D结构，其中包含了矩形中心
坐标（x，y）,矩形的宽和高（w，h）以及旋转角度。但是要绘制这个矩形需要矩形的四个角点，可以通过cv2.boxPoints()获得
1.8最小外接圆
函数cv2.minEnclosingCircle()可以帮我们找到一个对象的外接圆。它是所有能够包括对象的圆中面积最小的一个
1.9椭圆拟合
ellipse=cv2.fitEllipse(c) + img=cv2.ellipse(im,ellipse,(255,255,0),2)返回值其实就是旋转边界矩形的内切圆
# =============================================================================
# https://blog.csdn.net/u011854789/article/details/79836242?utm_source=blogxgwz9  有关rect返回值的定义（旋转边界矩形）
# =============================================================================
import cv2
import numpy as np

img = cv2.pyrDown(cv2.imread("bounding.jpg"))

ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) ###########取得是外轮廓

for c in contours:
    # find bounding box coordinates
    x, y, w, h = cv2.boundingRect(c) #没有旋转的矩形
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # find minimum area
    rect = cv2.minAreaRect(c)#rect  包含矩形中心 矩形宽高 和旋转角度
# =============================================================================
#注意：旋转角度θ是水平轴（x轴）逆时针旋转，与碰到的矩形的第一条边的夹角。并且这个边的边长是width，
#另一条边边长是height。也就是说，在这里，width与height不是按照长短来定义的。

#在opencv中，坐标系原点在左上角，相对于x轴，逆时针旋转角度为负，顺时针旋转角度为正。在这里，
#                                                θ∈（-90度，0]。  这里是-81.33°
# =============================================================================
    # 旋转边界矩形
    box = cv2.boxPoints(rect)
    # normalize coordinates to integers
    box = np.int0(box)#向下取整  而且还直接改变了数据类型 从float32->int64
    # draw contours
    cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

    # 最小外接圆
    (a, b), radius = cv2.minEnclosingCircle(c)
    # cast to integers
    center = (int(a), int(b))
    radius = int(radius)
    # draw the circle
    img = cv2.circle(img, center, radius, (0, 255, 0), 2)
    #椭圆拟合
    ellipse=cv2.fitEllipse(c)
    img=cv2.ellipse(img,ellipse,(255,255,0),2)
#裁剪合适的区域(方法1根据最小面积的矩形来裁剪)
Xs=[i[0] for i in box]
Ys=[i[1] for i in box]
Xmin=min(Xs)
Xmax=max(Xs)
Ymin=min(Ys)
Ymax=max(Ys)
#方法2 直接根据外轮廓的边界矩形来裁剪
X_l=Xmax-Xmin
Y_l=Ymax-Ymin
img_new=img[Ymin:Ymax,Xmin:Xmax] #Y是图矩阵的行，X是图像矩阵的列
cv2.imshow('new',img_new)

x=int(x)
y=int(y)
img_new1=img[y:y+h,x:x+w]
cv2.imshow('new1',img_new1)

cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
cv2.imshow("contours", img)

cv2.waitKey()
cv2.destroyAllWindows()


