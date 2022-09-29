# Python program to illustrate
# arithmetic operation of
# subtraction of pixels of two images

# organizing imports
import cv2
import numpy as np

# path to input images are specified and
# images are loaded with imread command
image1 = cv2.imread('R 0 G 0 B 0 .jpg')
image2 = cv2.imread('R 100 G 0 B 0 .jpg')

cv2.imshow('image 1', image1)
cv2.imshow('image 2', image2)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

print(image1.shape[:2])
print(image2.shape[:2])

# cv2.subtract is applied over the
# image inputs with applied parameters
sub = cv2.subtract(image1, image2)
# sub = abs(image1 - image2)

# the window showing output image
# with the subtracted image
# print(sub)
cv2.imshow('Subtracted Image', sub)

src = sub
if src is None:
    print('Could not open or find the image:')
    exit(0)
bgr_planes = cv2.split(src)
histSize = 256
histRange = (0, 256)  # the upper boundary is exclusive
accumulate = False
b_hist = cv2.calcHist(bgr_planes, [0], None, [
    histSize], histRange, accumulate=accumulate)
g_hist = cv2.calcHist(bgr_planes, [1], None, [
    histSize], histRange, accumulate=accumulate)
r_hist = cv2.calcHist(bgr_planes, [2], None, [
    histSize], histRange, accumulate=accumulate)
hist_w = 512
hist_h = 400
bin_w = int(round(hist_w/histSize))
histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
for i in range(1, histSize):
    cv2.line(histImage, (bin_w*(i-1), hist_h - int(b_hist[i-1])),
             (bin_w*(i), hist_h - int(b_hist[i])),
             (255, 0, 0), thickness=2)
    cv2.line(histImage, (bin_w*(i-1), hist_h - int(g_hist[i-1])),
             (bin_w*(i), hist_h - int(g_hist[i])),
             (0, 255, 0), thickness=2)
    cv2.line(histImage, (bin_w*(i-1), hist_h - int(r_hist[i-1])),
             (bin_w*(i), hist_h - int(r_hist[i])),
             (0, 0, 255), thickness=2)

cv2.imshow('Source image', src)
cv2.imshow('calcHist Demo', histImage)
cv2.waitKey()

img = np.reshape(sub, (np.prod(sub.shape[:2]), 3))
res = np.unique(img, axis=0, return_counts=True)
dom_color = res[0][np.argmax(res[1]), :]
print(dom_color)

# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
