import numpy as np
import cv2

path = r'D:\TONG\PycharmProjects\ultrasound image\image\5_1.jpg'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)  # 膨胀
# Finding sure foreground area

dist_transform = cv2.distanceTransform(opening, 2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.15 * dist_transform.max(), 255, 0)  # 参数改小了，出现不确定区域
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)  # 减去前景

import matplotlib.pyplot as plt


plt.figure()
plt.gray()
plt.imshow(sure_fg)
plt.show()
