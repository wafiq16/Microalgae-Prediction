import numpy as np
import cv2

img3 = cv2.imread('rgb_led/white.jpeg')
# img3 = img3[100:(100+50), 100:(100+50)]
# cv2.imshow('img3 pre', img3)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
# cv2.imshow('img3 aft', img3)
hue, sat, val = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
# c = np.mean(img3[:, :, 2])
hue = np.mean(hue)
sat = np.mean(sat)
val = np.mean(val)

print("white")
print(int(hue))
print(int(sat))
print(int(val))
# cv2.waitKey(0)

img3 = cv2.imread('rgb_led/purple.jpeg')
# img3 = img3[100:(100+50), 100:(100+50)]
# cv2.imshow('img3 pre', img3)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
# cv2.imshow('img3 aft', img3)
hue, sat, val = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
# c = np.mean(img3[:, :, 2])
hue = np.mean(hue)
sat = np.mean(sat)
val = np.mean(val)

print("purple")
print(int(hue))
print(int(sat))
print(int(val))

img3 = cv2.imread('rgb_led/magenta.jpeg')
# img3 = img3[100:(100+50), 100:(100+50)]
# cv2.imshow('img3 pre', img3)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
# cv2.imshow('img3 aft', img3)
hue, sat, val = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
# c = np.mean(img3[:, :, 2])
hue = np.mean(hue)
sat = np.mean(sat)
val = np.mean(val)

print("magenta")
print(int(hue))
print(int(sat))
print(int(val))

img3 = cv2.imread('rgb_led/blue.jpeg')
# img3 = img3[100:(100+50), 100:(100+50)]
# cv2.imshow('img3 pre', img3)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
# cv2.imshow('img3 aft', img3)
hue, sat, val = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
# c = np.mean(img3[:, :, 2])
hue = np.mean(hue)
sat = np.mean(sat)
val = np.mean(val)

print("blue")
print(int(hue))
print(int(sat))
print(int(val))

img3 = cv2.imread('rgb_led/green.jpeg')
# img3 = img3[100:(100+50), 100:(100+50)]
# cv2.imshow('img3 pre', img3)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
# cv2.imshow('img3 aft', img3)
hue, sat, val = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
# c = np.mean(img3[:, :, 2])
hue = np.mean(hue)
sat = np.mean(sat)
val = np.mean(val)

print("green")
print(int(hue))
print(int(sat))
print(int(val))

img3 = cv2.imread('rgb_led/yellow.jpeg')
# img3 = img3[100:(100+50), 100:(100+50)]
# cv2.imshow('img3 pre', img3)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
# cv2.imshow('img3 aft', img3)
hue, sat, val = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
# c = np.mean(img3[:, :, 2])
hue = np.mean(hue)
sat = np.mean(sat)
val = np.mean(val)

print("yellow")
print(int(hue))
print(int(sat))
print(int(val))

img3 = cv2.imread('rgb_led/orange.jpeg')
# img3 = img3[100:(100+50), 100:(100+50)]
# cv2.imshow('img3 pre', img3)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
# cv2.imshow('img3 aft', img3)
hue, sat, val = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
# c = np.mean(img3[:, :, 2])
hue = np.mean(hue)
sat = np.mean(sat)
val = np.mean(val)

print("orange")
print(int(hue))
print(int(sat))
print(int(val))

img3 = cv2.imread('rgb_led/red.jpeg')
# img3 = img3[100:(100+50), 100:(100+50)]
# cv2.imshow('img3 pre', img3)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
# cv2.imshow('img3 aft', img3)
hue, sat, val = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
# c = np.mean(img3[:, :, 2])
hue = np.mean(hue)
sat = np.mean(sat)
val = np.mean(val)

print("red")
print(int(hue))
print(int(sat))
print(int(val))
