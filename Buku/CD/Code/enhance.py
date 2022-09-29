import cv2
import os

from cv2 import imwrite

fname = '30.jpg'
img = cv2.imread(fname)
dst = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
imwrite('30_enh.jpeg', dst)

fname = 'h.jpeg'
img = cv2.imread(fname)
dst = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
imwrite('h_enh.jpeg', dst)
