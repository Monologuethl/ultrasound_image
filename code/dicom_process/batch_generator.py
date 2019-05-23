import glob
import os
import cv2
import numpy as np
import pydicom
import pylab
from PIL import Image

WSI_MASK_PATH = 'data'
paths_png = glob.glob(os.path.join(WSI_MASK_PATH, '*.png'))
paths_dcm = glob.glob(os.path.join(WSI_MASK_PATH, '*.dcm'))
paths_png.sort()
paths_dcm.sort()
print(paths_png)
print(paths_dcm)

j = 0
for size in range(len(paths_png)):
    ds = pydicom.read_file(paths_dcm[size])
    img = cv2.imread(paths_png[size])
    Grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(Grayimg, 30, 255, cv2.THRESH_BINARY)
    cv2.imwrite("data/" + str(j) + ".jpg", thresh, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    img = Image.open("data/" + str(j) + ".jpg")
    matrix = np.asarray(img)
    arr = matrix.flatten()
    for i in range(262144):
        if arr[i] < 100:
            ds.pixel_array.flat[i] = 0
    ds.PixelData = ds.pixel_array.tostring()
    ds.save_as("data/" + str(j) + ".dcm")
    j = j + 1
    print(j)

# pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
# pylab.show()
