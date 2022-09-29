from numpy import array
from numpy.random import uniform
from numpy import hstack
from pandas import read_csv
import numpy as np
from sklearn import datasets


def find_similarities(img1, img2):
    # dist = cv2.norm(img1 - img2, cv2.NORM_L2)
    # print(img1)
    # print(img2)
    dist2 = np.linalg.norm(np.array(img1) - np.array(img2))
    # print("cv2 norm = " + str(dist))
    # print("numpy linear = " + str(dist2))
    return dist2


path = 'data_train.csv'
df = read_csv(path, header=None)
# scaler = StandardScaler()
# transform data
# scaled = scaler.fit_transform(df)
# split into input and output columns
x, y = df.values[:, :-3].tolist(), df.values[:, -3:].tolist()
# x, y = x.astype('float'), y.astype('float')
# print(x.shape[:2])
# print(y.shape[:2])
# plt.plot(y)
# plt.show()
# x = x.reshape(x.shape[0], x.shape[1], 1)
# print("x:", x.shape, "y:", y.shape)

# in_dim = (x.shape[1], x.shape[2])
# out_dim = y.shape[1]
# print(in_dim)
# print(out_dim)

path = 'Sensor_test.csv'
df = read_csv(path, header=None)
# scaler = StandardScaler()
# transform data
# scaled = scaler.fit_transform(df)
# split into input and output columns
row, inpt = df.values[:, :-3].tolist(), df.values[:, -3:].tolist()

dataset = {}
j = 0

# iterating through the elements of list
for i in x:
    # print(y[j])
    dataset[str(i)] = y[j]
    j += 1

print(dataset)

datatest = {}
j = 0
# iterating through the elements of list
for i in row:
    datatest[str(i)] = inpt[j]
    j += 1

# print(dataset)
# print(datatest)

similarity = {}

# print(row)
# print(x)
for i in row:
    for j in x:
        similar = find_similarities(i, j)
        similarity[str(datatest[str(i)]) + '||' +
                   str(dataset[str(j)])] = similar
        # print(i, j)
        # print(similar)
    similarity = dict(sorted(similarity.items(), key=lambda item: item[1]))
    print(' ')
    # print(similarity)
    key = list(similarity)
    print("most similar sensor to sample = " + str(key[0]))
    print("distance  = " + str(similarity[key[0]]))
    similarity = {}
# print(row)
# print("image sample = " + str(path))
