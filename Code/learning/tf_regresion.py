# mlp for regression
from pickletools import optimize
from statistics import mode
from numpy import sqrt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from matplotlib import pyplot
from numpy import argmax
import tensorflow as tf
from keras.models import load_model
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.layers import GaussianNoise
import csv
import math
import time
import json
import datetime

# myData = open('data/data_alga.csv', 'w')
# myData = open('data/pewarna_15_tetes.csv', 'w')

# myFile = 'data/tumbas_bebek_r.csv'
# import noise layer
# load the dataset
path = 'data_train.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[:, :-3], df.values[:, -3:]
X, y = X.astype('float'), y.astype('float')

# X *= 255
# print(X)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
print(n_features)
layer = tf.keras.layers.experimental.preprocessing.Normalization()
layer.adapt(X)
# define model
# print(len(X_train))
# print(len(X_test))
# model = Sequential()
# model.add(Dense(512, input_dim=n_features, activation='relu',
#           kernel_initializer='he_normal'))
# # model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(12, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(9, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(7, activation='relu', kernel_initializer='he_normal'))
# # model.add(Dense(3, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(3, activation='linear'))

# coba
# grid search
# model 1
# model = Sequential()
# model.add(Dense(256, input_dim=n_features, activation='relu',
#           kernel_initializer='he_normal'))
# model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(3, activation='linear'))

# model 2
# model = Sequential()
# model.add(Dense(20, input_dim=n_features, activation='relu',
#           kernel_initializer='he_normal'))
# model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(3, activation='linear'))

# model 3
# model = Sequential()
# model.add(Dense(64, input_dim=n_features, activation='relu',
#           kernel_initializer='he_normal'))
# model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(3, activation='linear'))

# model 4
# model = Sequential()
# model.add(Dense(20, input_dim=n_features, activation='relu',
#           kernel_initializer='he_normal'))
# model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(3, activation='linear'))

# print(X.shape[:2])
# in_dim = X.shape[:2]

# model 5
model = Sequential()
model.add(layer)
model = Sequential()
model.add(Dense(64, input_dim=3,
          kernel_initializer='he_uniform', activation='relu'))
# model.add(Dense(500,
#           kernel_initializer='he_uniform', activation='relu'))
# model.add(Dense(400,
#           kernel_initializer='he_uniform', activation='relu'))
# model.add(Dense(300,
#           kernel_initializer='he_uniform', activation='relu'))
# model.add(Dense(200,
#           kernel_initializer='he_uniform', activation='relu'))
# model.add(Dense(100,
#           kernel_initializer='he_uniform', activation='relu'))
# model.add(Dense(20,
# kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(10,
          kernel_initializer='he_uniform', activation='relu'))
# model.add(Dense(5,
#   kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(3))
model.compile(loss='mae', optimizer='adam')
# model.add(GaussianNoise(0.1))
model.summary()

# plot_model(model, 'model.png', show_shapes=True)

# optimizer = tf.keras.optimizers.Adam()
# optimizer = tf.keras.optimizers.Adam(learning_rate=2e-07)
# print("x train")
# print(X_train)
# print("x test")
# print(X_test)
# optimizer = tf.keras.optimizers.SGD()
# compile the model
# model.compile(optimizer, loss='mae')
# fit the model
history = model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=300, verbose=1)
# evaluate the model
# pyplot.title('Learning Curves')
# pyplot.xlabel('Epoch')
# pyplot.ylabel('mse')
# pyplot.plot(history.history['loss'], label='train')
# # pyplot.plot(history.history['mse'], label='val')
# pyplot.legend()
# pyplot.show()

# print(model.score(X_test, y_test))
print("error train")
error = model.evaluate(X_train, y_train, verbose=1, batch_size=1)
print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))

print("error test")
error = model.evaluate(X_test, y_test, verbose=1, batch_size=1)
print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))

# make a prediction


# 40,50,100
# row = [0.466, 0.086, 0.448, 0.636, 0.636, 0.636, 0.496,
#    0.19, 0.314, 0.616, 0.033, 0.351, 0.55, 0.127, 0.323]
# yhat = model.predict([row])

# print("[[40, 50, 100]]")
# print(yhat)
# error = mean_absolute_error(y_test, yhat)
# print('MAE: %.3f' % error)
# print('Predicted: %s' % (yhat))
model.save('model.h5')
print("Predict Saja")

model = model

# inpt = [
#     [0, 0, 0],
#     [0, 0, 100],
#     [100, 0, 0],
#     [0, 100, 0],
#     [20, 80, 0],
#     [40, 60, 0],
#     [60, 40, 0],
#     [80, 20, 0],
#     [0, 20, 80],
#     [0, 40, 60],
#     [0, 60, 40],
#     [0, 80, 20],
#     [20, 0, 80],
#     [40, 0, 60],
#     [60, 0, 40],
#     [80, 0, 20],
#     [20, 60, 20],
#     [20, 40, 40],
#     [20, 20, 60],
#     [40, 40, 20],
#     [40, 20, 40],
#     [60, 20, 20],
#     [33.3, 33.3, 33.3],
#     [10, 0, 0],
#     [20, 0, 0],
#     [50, 0, 0],
#     [0, 10,	0],
#     [0, 20,	0],
#     [0, 50,	0],
#     [0, 0, 10],
#     [0, 0, 20],
#     [0, 0, 50]
# ]

# row = [
#     [202, 208, 205, 206, 209, 209, 99, 93, 77, 99, 93, 77, 87, 85, 87],
#     [0, 214, 231, 0, 224, 232, 52, 77, 53, 52, 77, 53, 38, 112, 112],
#     [254, 138, 101, 255, 143, 103, 92, 65, 46, 92, 65, 46, 134, 44, 81],
#     [0, 246, 65, 0, 248, 66, 51, 73, 33, 51, 73, 33, 64, 116, 72],
#     [22, 227, 65, 24, 235, 69, 86, 91, 51, 86, 91, 51, 72, 106, 81],
#     [92, 140, 41, 92, 140, 41, 104, 100, 61, 104, 100, 61, 85, 89, 81],
#     [245, 195, 97, 255, 202, 103, 105, 97, 60, 105, 97, 60, 90, 82, 86],
#     [254, 135, 89, 255, 143, 88, 109, 97, 63, 109, 97, 63, 98, 78, 86],
#     [0, 244, 164, 0, 250, 168, 71, 110, 64, 71, 110, 64, 48, 114, 98],
#     [0, 244, 106, 0, 251, 111, 79, 109, 62, 79, 109, 62, 49, 119, 90],
#     [0, 248, 146, 0, 255, 177, 76, 108, 61, 76, 108, 61, 50, 120, 89],
#     [0, 241, 100, 0, 255, 125, 81, 110, 61, 81, 110, 61, 56, 121, 81],
#     [0, 167, 253, 0, 186, 255, 85, 103, 70, 85, 103, 70, 46, 97, 115],
#     [52, 141, 253, 66, 171, 255, 101, 101, 70, 101, 101, 70, 69, 80, 109],
#     [56, 114, 251, 69, 130, 255, 103, 99, 70, 103, 99, 70, 75, 79, 109],
#     [251, 132, 253, 255, 147, 255, 117, 102, 76, 117, 102, 76, 96, 64, 99],
#     [0, 188, 60, 0, 214, 72, 109, 119, 85, 109, 119, 85, 69, 107, 83],
#     [0, 172, 105, 0, 174, 105, 86, 103, 66, 86, 103, 66, 58, 103, 98],
#     [0, 147, 115, 0, 176, 137, 87, 105, 63, 87, 105, 63, 59, 102, 98],
#     [0, 153, 75, 0, 166, 84, 95, 103, 62, 95, 103, 62, 75, 94, 90],
#     [0, 133, 132, 0, 133, 136, 83, 99, 60, 83, 99, 60, 62, 98, 98],
#     [108, 112, 110, 118, 123, 121, 99, 99, 61, 99, 99, 61, 86, 82, 91],
#     [28, 141, 156, 34, 150, 167, 91, 96, 60, 91, 96, 60, 76, 89, 93],
#     [254, 108, 164, 255, 112, 173, 142, 134, 111, 145, 178, 87, 100, 74, 87],
#     [254, 131, 218, 255, 136, 233, 142, 128, 104, 147, 171, 87, 103, 72, 83],
#     [254, 130, 229, 255, 132, 228, 139, 122, 102, 148, 158, 89, 109, 63, 87],
#     [60, 190, 100, 65, 187, 106, 117, 124, 89, 127, 169, 63, 80, 95, 76],
#     [95, 215, 129, 95, 218, 132, 118, 124, 89, 128, 170, 61, 88, 90, 82],
#     [0, 249, 111, 0, 255, 119, 81, 112, 60, 92, 156, 31, 69, 112, 77],
#     [0, 237, 252, 0, 255, 255, 83, 106, 80, 93, 149, 47, 51, 100, 106],
#     [0, 181, 236, 0, 204, 255, 79, 107, 79, 78, 149, 39, 53, 103, 106],
#     [0, 157, 244, 0, 161, 255, 62, 97, 75, 73, 141, 43, 36, 111, 111],
# ]
# y = 0
# for i in row:
#     yhat = model.predict([i])
#     print("input " + str(y) + " = " + str(inpt[y]))
#     # print()
#     print("hasil " + " = " + str(yhat[0]))
#     # print()
#     y += 1

path = 'Sensor_test.csv'
df = read_csv(path, header=None)
# split into input and output columns
row, inpt = df.values[:, :-3].tolist(), df.values[:, -3:].tolist()
# row, inpt = row.astype('float'), inpt.astype('float')

# print(row)
# print(inpt)
# inpt = [
#     [0, 0, 0],
#     [0, 0, 100],
#     [100, 0, 0],
#     [0, 100, 0],
#     # [20, 80, 0],
#     # [40, 60, 0],
#     # [60, 40, 0],
#     # [80, 20, 0],
#     # [0, 20, 80],
#     # [0, 40, 60],
#     # [0, 60, 40],
#     # [0, 80, 20],
#     # [20, 0, 80],
#     # [40, 0, 60],
#     # [60, 0, 40],
#     # [80, 0, 20],
#     # [20, 60, 20],
#     # [20, 40, 40],
#     # [20, 20, 60],
#     # [40, 40, 20],
#     # [40, 20, 40],
#     # [60, 20, 20],
#     # [33.3, 33.3, 33.3],
#     [10, 0, 0],
#     [20, 0, 0],
#     [50, 0, 0],
#     [0, 10,	0],
#     [0, 20,	0],
#     [0, 50,	0],
#     [0, 0, 10],
#     [0, 0, 20],
#     [0, 0, 50]
# ]

# row = [
#     [202, 208, 205, 206, 209, 209, 99, 93, 77, 99, 93, 77, 87, 85, 87],
#     [0, 214, 231, 0, 224, 232, 52, 77, 53, 52, 77, 53, 38, 112, 112],
#     [254, 138, 101, 255, 143, 103, 92, 65, 46, 92, 65, 46, 134, 44, 81],
#     [0, 246, 65, 0, 248, 66, 51, 73, 33, 51, 73, 33, 64, 116, 72],
#     # [22, 227, 65, 24, 235, 69, 86, 91, 51, 86, 91, 51, 72, 106, 81],
#     # [92, 140, 41, 92, 140, 41, 104, 100, 61, 104, 100, 61, 85, 89, 81],
#     # [245, 195, 97, 255, 202, 103, 105, 97, 60, 105, 97, 60, 90, 82, 86],
#     # [254, 135, 89, 255, 143, 88, 109, 97, 63, 109, 97, 63, 98, 78, 86],
#     # [0, 244, 164, 0, 250, 168, 71, 110, 64, 71, 110, 64, 48, 114, 98],
#     # [0, 244, 106, 0, 251, 111, 79, 109, 62, 79, 109, 62, 49, 119, 90],
#     # [0, 248, 146, 0, 255, 177, 76, 108, 61, 76, 108, 61, 50, 120, 89],
#     # [0, 241, 100, 0, 255, 125, 81, 110, 61, 81, 110, 61, 56, 121, 81],
#     # [0, 167, 253, 0, 186, 255, 85, 103, 70, 85, 103, 70, 46, 97, 115],
#     # [52, 141, 253, 66, 171, 255, 101, 101, 70, 101, 101, 70, 69, 80, 109],
#     # [56, 114, 251, 69, 130, 255, 103, 99, 70, 103, 99, 70, 75, 79, 109],
#     # [251, 132, 253, 255, 147, 255, 117, 102, 76, 117, 102, 76, 96, 64, 99],
#     # [0, 188, 60, 0, 214, 72, 109, 119, 85, 109, 119, 85, 69, 107, 83],
#     # [0, 172, 105, 0, 174, 105, 86, 103, 66, 86, 103, 66, 58, 103, 98],
#     # [0, 147, 115, 0, 176, 137, 87, 105, 63, 87, 105, 63, 59, 102, 98],
#     # [0, 153, 75, 0, 166, 84, 95, 103, 62, 95, 103, 62, 75, 94, 90],
#     # [0, 133, 132, 0, 133, 136, 83, 99, 60, 83, 99, 60, 62, 98, 98],
#     # [108, 112, 110, 118, 123, 121, 99, 99, 61, 99, 99, 61, 86, 82, 91],
#     # [28, 141, 156, 34, 150, 167, 91, 96, 60, 91, 96, 60, 76, 89, 93],
#     [254, 108, 164, 255, 112, 173, 142, 134, 111, 145, 178, 87, 100, 74, 87],
#     [254, 131, 218, 255, 136, 233, 142, 128, 104, 147, 171, 87, 103, 72, 83],
#     [254, 130, 229, 255, 132, 228, 139, 122, 102, 148, 158, 89, 109, 63, 87],
#     [60, 190, 100, 65, 187, 106, 117, 124, 89, 127, 169, 63, 80, 95, 76],
#     [95, 215, 129, 95, 218, 132, 118, 124, 89, 128, 170, 61, 88, 90, 82],
#     [0, 249, 111, 0, 255, 119, 81, 112, 60, 92, 156, 31, 69, 112, 77],
#     [0, 237, 252, 0, 255, 255, 83, 106, 80, 93, 149, 47, 51, 100, 106],
#     [0, 181, 236, 0, 204, 255, 79, 107, 79, 78, 149, 39, 53, 103, 106],
#     [0, 157, 244, 0, 161, 255, 62, 97, 75, 73, 141, 43, 36, 111, 111],
# ]
y = 0
# print(row[0])
for i in row:
    # print(i)
    yhat = model.predict([i])
    print("input " + str(y) + " = " + str(inpt[y]))
    # print()
    print("hasil " + " = " + str(yhat[0]))
    # print()

    err_R = abs(yhat[0][0] - inpt[y][0])
    err_G = abs(yhat[0][1] - inpt[y][1])
    err_B = abs(yhat[0][2] - inpt[y][2])
    mean_err = (err_R + err_G + err_B)/3
    r = [inpt[y][0], inpt[y][1], inpt[y][2],
         yhat[0][0], yhat[0][1], yhat[0][2], err_R, err_G, err_B, mean_err]
    print(r)
    # with open('paper.csv', 'a', newline='') as outfile:
    # writer = csv.writer(outfile)
    # writer = csv.writer(f)
    # writer_reg = csv.writer(outfile)
    # writer_reg.writerow(r)
    y += 1
