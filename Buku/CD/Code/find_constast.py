import cv2
from numpy import argmax
from numpy import argmin

img0 = cv2.imread("0.jpeg", cv2.IMREAD_GRAYSCALE)
# mymin, mymax = 0, 0
min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(img0)
print(min_val, max_val, min_indx, max_indx)
contrast0 = max_val-min_val
print(contrast0)

img0 = cv2.imread("15.jpeg", cv2.IMREAD_GRAYSCALE)
# mymin, mymax = 0, 0
min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(img0)
print(min_val, max_val, min_indx, max_indx)
contrast0 = max_val-min_val
print(contrast0)

img0 = cv2.imread("30.jpeg", cv2.IMREAD_GRAYSCALE)
# mymin, mymax = 0, 0
min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(img0)
print(min_val, max_val, min_indx, max_indx)
contrast0 = max_val-min_val
print(contrast0)

img0 = cv2.imread("45.jpeg", cv2.IMREAD_GRAYSCALE)
# mymin, mymax = 0, 0
min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(img0)
print(min_val, max_val, min_indx, max_indx)
contrast0 = max_val-min_val
print(contrast0)

img0 = cv2.imread("60.jpeg", cv2.IMREAD_GRAYSCALE)
# mymin, mymax = 0, 0
min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(img0)
print(min_val, max_val, min_indx, max_indx)
contrast0 = max_val-min_val
print(contrast0)

img0 = cv2.imread("90.jpeg", cv2.IMREAD_GRAYSCALE)
# mymin, mymax = 0, 0
min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(img0)
print(min_val, max_val, min_indx, max_indx)
contrast0 = max_val-min_val
print(contrast0)
