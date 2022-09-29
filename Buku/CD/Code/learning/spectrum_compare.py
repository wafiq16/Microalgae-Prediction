import cv2
import numpy as np
import glob
import statistics

filenames = [img for img in glob.glob("image_cam/*.jpg")]

filenames.sort()  # ADD THIS LINE

file = '000.jpg'
path = 'image_cam/' + file
image_sample = cv2.imread(path)


def pattern(img, mean=True):
    rows, cols, _ = img.shape
    patterned_image = cv2.resize(img, (1, rows))
    average = [0, 0, 0]
    column = [cols, cols, cols]
    # print(rows)
    if(mean):
        for i in range(rows):
            for j in range(cols):
                k = img[i, j]
                # print(k)
                average += k
            # print(average/column)
            patterned_image[i] = average/column
            average = [0, 0, 0]
    else:
        for i in range(rows):
            # for j in range(cols):
            k = np.median(img[i], axis=0)
            # print(k)
            # average += k
            # print(average/column)
            patterned_image[i] = k
            # average = [0, 0, 0]
            # print(i)
    return patterned_image


def find_similarities(img1, img2):
    dist = cv2.norm(img1 - img2, cv2.NORM_L2)
    # dist2 = np.linalg.norm(img1 - img2)
    print("cv2 norm = " + str(dist))
    # print("numpy linear = " + str(dist2))
    return dist


images = []
similarity = {}
for img in filenames:
    n = cv2.imread(img)

    # image_sample = pattern(image_sample, mean=False)
    image_sample = pattern(image_sample, mean=True)

    # n = pattern(n, mean=False)
    n = pattern(n, mean=True)

    # similar = find_similarities(dst1, dst2)
    similar = find_similarities(n, image_sample)
    similarity[img] = similar
    # images.appednd(n)
    # print(img)

cv2.imshow('sample', image_sample)
# print(image_sample.shape[:2])
similarity = dict(sorted(similarity.items(), key=lambda item: item[1]))
print(similarity)
print("image sample = " + str(path))
key = list(similarity)
print("most similar image to sample = " + str(key[2]))
cv2.waitKey()
# print(key[2])
