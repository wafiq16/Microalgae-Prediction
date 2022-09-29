import cv2
import os

from cv2 import imwrite


def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    print(len(channels))
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


fname = '30.jpg'
img = cv2.imread(fname)

cv2.imshow('img', img)
img2 = hisEqulColor(img)
cv2.imshow('img2', img2)
# imwrite('30_equ.jpeg', img2)

# median = cv2.medianBlur(img, 107)


def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # lim = 255 - value
    # s[s > lim] = 255
    # s[s <= lim] += value

    lim = 255 - value
    v[v < lim] = value
    v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


frame = increase_brightness(img2, value=255)

imwrite('30_equ.jpg', frame)

# fname = 'waf.jpg'
# img = cv2.imread(fname)

# # cv2.imshow('img', img)
# # img2 = hisEqulColor(img)
# # cv2.imshow('img2', img2)
# median = cv2.medianBlur(img, 107)


# def increase_brightness(img, value):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)

#     lim = 255 - value
#     s[s > lim] = 255
#     s[s <= lim] += value

#     lim = 255 - value
#     v[v > lim] = 255
#     v[v <= lim] += value

#     final_hsv = cv2.merge((h, s, v))
#     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     return img


# frame = increase_brightness(median, value=40)

# imwrite('waf_equ.jpg', frame)
