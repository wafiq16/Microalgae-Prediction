import numpy as np
import cv2

img3 = cv2.imread('sens/1.jpg')
# img3 = img3[100:(100+50), 100:(100+50)]
# cv2.imshow('img3 pre', img3)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
# cv2.imshow('img3 aft', img3)
hue, sat, val = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
# c = np.mean(img3[:, :, 2])
hue = np.mean(hue)
sat = np.mean(sat)
val = np.mean(val)

print("1")
print(int(hue))
print(int(sat))
print(int(val))
# cv2.waitKey(0)

img3 = cv2.imread('sens/2.jpg')
# img3 = img3[100:(100+50), 100:(100+50)]
# cv2.imshow('img3 pre', img3)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
# cv2.imshow('img3 aft', img3)
hue, sat, val = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
# c = np.mean(img3[:, :, 2])
hue = np.mean(hue)
sat = np.mean(sat)
val = np.mean(val)

print("2")
print(int(hue))
print(int(sat))
print(int(val))

img3 = cv2.imread('sens/3.jpg')
# img3 = img3[100:(100+50), 100:(100+50)]
# cv2.imshow('img3 pre', img3)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
# cv2.imshow('img3 aft', img3)
hue, sat, val = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
# c = np.mean(img3[:, :, 2])
hue = np.mean(hue)
sat = np.mean(sat)
val = np.mean(val)

print("3")
print(int(hue))
print(int(sat))
print(int(val))

img3 = cv2.imread('sens/4.jpg')
# img3 = img3[100:(100+50), 100:(100+50)]
# cv2.imshow('img3 pre', img3)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
# cv2.imshow('img3 aft', img3)
hue, sat, val = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
# c = np.mean(img3[:, :, 2])
hue = np.mean(hue)
sat = np.mean(sat)
val = np.mean(val)

print("4")
print(int(hue))
print(int(sat))
print(int(val))

img3 = cv2.imread('sens/5.jpg')
# img3 = img3[100:(100+50), 100:(100+50)]
# cv2.imshow('img3 pre', img3)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
# cv2.imshow('img3 aft', img3)
hue, sat, val = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
# c = np.mean(img3[:, :, 2])
hue = np.mean(hue)
sat = np.mean(sat)
val = np.mean(val)

print("5")
print(int(hue))
print(int(sat))
print(int(val))
