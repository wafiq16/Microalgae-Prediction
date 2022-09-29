from __future__ import print_function
from __future__ import division
import cv2
from cv2 import imwrite
from matplotlib import image
import numpy as np
from numpy.linalg import norm
import csv

path_ref1 = 'image_cam/0500.jpg'
file = '000.jpg'
path_ref2 = 'image_cam/' + file

path_dest = 'hasil_cam/' + file
# path_ref2 = 'image_cam/0150.jpg'

print(path_ref1)
print(path_ref2)

img_test1 = cv2.imread(path_ref1)
# cv2.imshow('gambar input 1', img_test1)
img_test2 = cv2.imread(path_ref2)
# cv2.imshow('gambar input 2', img_test2)
print(img_test1.shape[:2])


def pattern(img):
    rows, cols, _ = img.shape
    patterned_image = cv2.resize(img, (1, rows))
    average = [0, 0, 0]
    column = [cols, cols, cols]
    # print(rows)
    for i in range(rows):
        for j in range(cols):
            k = img[i, j]
            # print(k)
            average += k
        # print(average/column)
        patterned_image[i] = average/column
        average = [0, 0, 0]

    return patterned_image


# cv2.imshow('gambar input 1', img_test1)
img_test1 = cv2.resize(img_test1, (94, 342))
img_test2 = cv2.resize(img_test2, (94, 342))

med1 = cv2.medianBlur(img_test1, 5)
med2 = cv2.medianBlur(img_test2, 5)

image1 = pattern(img_test1)
image2 = pattern(img_test2)
med1 = pattern(med1)

cv2.imwrite(path_dest, image2)
# cv2.imshow('gambar input 1', image1)
# cv2.imshow('gambar input 2', image2)

image_re_1 = cv2.resize(image1, (94, 342))
image_re_2 = cv2.resize(image2, (94, 342))
image_re_med = cv2.resize(med1, (94, 342))


# med1 = cv2.medianBlur(img_test1, 93)
# med2 = cv2.medianBlur(img_test2, 93)

# cv2.imshow('gambar rec 1', image_re_1)
# cv2.imshow('gambar rec 2', image_re_2)

cv2.imshow('non-median blur image 1', image_re_med)
cv2.imshow('median blur image 1', image_re_1)
# med1 = cv2.resize(med1, (1, 342))
# print(image1.shape[:2])
# print(med1.shape[:2])
dist = cv2.norm(med1 - image1, cv2.NORM_L2)
dist2 = np.linalg.norm(med1 - image1)
print(dist)
print(dist2)
# print(image1)
# print(med1)
# cv2.imshow('gambar med 2', med2)
# result = np.dot(image1, image2)/(norm(image1)*norm(image2))
# cv2.imshow('hasil pola 1', image1)
# cv2.imshow('hasil pola 2', image2)

src_base = image1
src_test1 = image2

# src_base = img_test1
# src_test1 = img_test2

hsv_base = cv2.cvtColor(src_base, cv2.COLOR_BGR2HSV)
hsv_test1 = cv2.cvtColor(src_test1, cv2.COLOR_BGR2HSV)

h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
# hue varies from 0 to 179, saturation from 0 to 255
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges  # concat lists
# Use the 0-th and 1-st channels
channels = [0, 1]
hist_base = cv2.calcHist([hsv_base], channels, None,
                         histSize, ranges, accumulate=False)
cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

hist_test1 = cv2.calcHist([hsv_test1], channels, None,
                          histSize, ranges, accumulate=False)
cv2.normalize(hist_test1, hist_test1, alpha=0,
              beta=1, norm_type=cv2.NORM_MINMAX)

for compare_method in range(4):
    # base_base = cv2.compareHist(hist_base, hist_base, compare_method)
    base_test1 = cv2.compareHist(hist_base, hist_test1, compare_method)
    print('Method:', compare_method,
          'Base-Test(1):', base_test1)

# dist = cv2.norm(image1 - image2, cv2.NORM_L2)
# dist2 = np.linalg.norm(image1 - image2)

myFile = 'data/Pewarna_Satuan_MIC.csv'

with open(myFile, 'a', newline='') as outfile:
    writer_reg = csv.writer(outfile)
    writer_reg.writerow([dist, file])

cv2.waitKey()
