import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# original
image = cv.imread("36 KARIES.JPG", cv.IMREAD_UNCHANGED)
cv.imshow('Original', image)
plt.subplot(221)
plt.title("Original")
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
# ret1,th1 = cv.threshold(image,127,255,cv.THRESH_BINARY)


# gray
# konversi RGB ke Grayscale menggunakan library opencv
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
cv.imshow('Grayscale', gray)
plt.subplot(222)
plt.title("Grayscale")
plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))


# median
median = cv.medianBlur(gray, 7)
compare = np.concatenate((gray, median), axis=1)  # side by side comparison
cv.imshow('Median', median)
plt.subplot(223)
plt.title("Median")
plt.imshow(cv.cvtColor(median, cv.COLOR_BGR2RGB))


# histo
# histo = cv.equalizeHist(median)
# histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
# cv.imshow("Histogram", histogram)

# plt.subplot(224)
# plt.title("Histogram")
# plt.imshow(histogram)


# find frequency of pixels in range 0-255
color = ('b', 'g', 'r')
histr = cv.calcHist([median], [0], None, [256], [0, 256])
cv.imshow('Histogram', histr)
# show the plotting graph of an image
plt.subplot(224)
plt.plot(histr)
plt.show()

# ret2,th2 = cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
ret1, thresh1 = cv.threshold(median, 120, 255, cv.THRESH_BINARY)
# cv.imshow('Binary Threshold1', thresh1)
ret2, thresh2 = cv.threshold(median, 128, 255, cv.THRESH_BINARY)
# cv.imshow('Binary Threshold2', thresh2)
ret3, thresh3 = cv.threshold(median, 126, 255, cv.THRESH_BINARY)
cv.imshow('Binary Threshold3', thresh3)
# ret1,thresh2= cv.threshold(median,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# cv.imshow('Threshold', thresh2)
# plt.subplot(231)

# plt.title("Threshold")
# plt.imshow(thresh1)


plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
