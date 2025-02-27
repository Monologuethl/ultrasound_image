# import numpy as np
# import cv2
#
#
# class Point(object):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
#     def getX(self):
#         return self.x
#
#     def getY(self):
#         return self.y
#
#
# def getGrayDiff(img, currentPoint, tmpPoint):
#     return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))
#
#
# def selectConnects(p):
#     if p != 0:
#         connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
#                     Point(0, 1), Point(-1, 1), Point(-1, 0)]
#     else:
#         connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
#     return connects
#
#
# def regionGrow(img, seeds, thresh, p=1):
#     height, weight = img.shape
#     seedMark = np.zeros(img.shape)
#     seedList = []
#     for seed in seeds:
#         seedList.append(seed)
#     label = 1
#     connects = selectConnects(p)
#     while (len(seedList) > 0):
#         currentPoint = seedList.pop(0)
#
#         seedMark[currentPoint.x, currentPoint.y] = label
#         for i in range(8):
#             tmpX = currentPoint.x + connects[i].x
#             tmpY = currentPoint.y + connects[i].y
#             if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
#                 continue
#             grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
#             if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
#                 seedMark[tmpX, tmpY] = label
#                 seedList.append(Point(tmpX, tmpY))
#     return seedMark
#
#
# img = cv2.imread(r'D:\TONG\PycharmProjects\ultrasound image\image\5.jpg', 0)
#
# seeds = [Point(10, 10), Point(82, 150), Point(20, 300)]
# binaryImg = regionGrow(img, seeds, 10)
# cv2.imshow(' ', binaryImg)
# cv2.waitKey(0)
# -*- coding:utf-8 -*-
import cv2
import numpy as np

# import Image
path = r'D:\TONG\PycharmProjects\ultrasound image\image\5_1.jpg'
im = cv2.imread(path)
im_shape = im.shape
height = im_shape[0]
width = im_shape[1]

print('the shape of image :', im_shape)


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
            Point(-1, 0)]


# 计算两个点间的欧式距离
def get_dist(seed_location1, seed_location2):
    l1 = im[seed_location1.x, seed_location1.y]
    l2 = im[seed_location2.x, seed_location2.y]
    count = np.sqrt(np.sum(np.square(l1 - l2)))
    return count


# 标记，判断种子是否已经生长
img_mark = np.zeros([height, width])

# 建立空的图像数组,作为一类
img_re = im.copy()
for i in range(height):
    for j in range(width):
        img_re[i, j][0] = 0
        img_re[i, j][1] = 0
        img_re[i, j][2] = 0
# 随即取一点作为种子点
seed_list = [Point(10, 10)]
T = 7  # 阈值
class_k = 1  # 类别
# 生长一个类
while len(seed_list) > 0:
    seed_tmp = seed_list[0]
    # 将以生长的点从一个类的种子点列表中删除
    seed_list.pop(0)

    img_mark[seed_tmp.x, seed_tmp.y] = class_k

    # 遍历8邻域
    for i in range(8):
        tmpX = seed_tmp.x + connects[i].x
        tmpY = seed_tmp.y + connects[i].y

        if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width:
            continue
        dist = get_dist(seed_tmp, Point(tmpX, tmpY))
        # 在种子集合中满足条件的点进行生长
        if dist < T and img_mark[tmpX, tmpY] == 0:
            img_re[tmpX, tmpY][0] = im[tmpX, tmpY][0]
            img_re[tmpX, tmpY][1] = im[tmpX, tmpY][1]
            img_re[tmpX, tmpY][2] = im[tmpX, tmpY][2]
            img_mark[tmpX, tmpY] = class_k
            seed_list.append(Point(tmpX, tmpY))

# 输出图像

cv2.imshow('output', img_re)
cv2.waitKey(0)
