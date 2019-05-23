# ex2tron's blog:
# http://ex2tron.wang

import cv2
import numpy as np

# 1.腐蚀与膨胀
path = r'D:\TONG\PycharmProjects\ultrasound image\image\5.jpg'

img = cv2.imread(path, 0)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel)  # 腐蚀
dilation = cv2.dilate(img, kernel)  # 膨胀

cv2.imshow('erosion/dilation', img)
cv2.waitKey(0)

cv2.imshow('erosion/dilation', erosion)
cv2.waitKey(0)

cv2.imshow('erosion/dilation', dilation)
cv2.waitKey(0)






# 2.定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
print(kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构
print(kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形结构
print(kernel)

# 3.开运算与闭运算
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素

# 开运算
img = cv2.imread(path, 0)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow('opening', opening)
cv2.waitKey(0)

# 闭运算
img = cv2.imread(path, 0)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('closing', closing)
cv2.waitKey(0)


# 4.形态学梯度
img = cv2.imread(path, 0)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('morphological gradient', gradient)
cv2.waitKey(0)


# 5.顶帽
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('top hat', tophat)
cv2.waitKey(0)


# 6.黑帽
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('black hat', blackhat)
cv2.waitKey(0)
