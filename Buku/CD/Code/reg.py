# mlp for regression
from numpy import argmax
from numpy import sqrt
from pandas import read_csv
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot

import csv

f_uji = open('data/data_hasil_reg.csv', 'w')
# writer     = csv.writer(f)
writer_uji = csv.writer(f_uji)

# load the dataset
path = 'data/dummy_data_reg.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
print(n_features)
# define model
# model = Sequential()
# model.add(Dense(64, activation='relu',
#   kernel_initializer='he_normal', input_shape=(n_features,)))
# model.add(Dense(1024, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(32))
# model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(4, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(1))
# normalizer = tf.keras.layers.Normalization(axis=-1)
# model = Sequential()
# # model.add(normalizer)
# model.add(Dense(64, activation='relu',
#                 kernel_initializer='he_normal', input_shape=(n_features,)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

model = keras.models.Sequential([

    keras.layers.Dense(256, input_dim=X_train.shape[1], activation='relu'),
    keras.layers.Dense(256, input_dim=X_train.shape[1], activation='relu'),
    # keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=1, activation="linear"),
])
model.summary()

# model.compile(loss='mean_absolute_error',
#   optimizer=tf.keras.optimizers.Adam(0.001))
# model = build_and_compile_model(normalizer)
# dnn_model.summary()

plot_model(model, 'model_reg.png', show_shapes=True)

# compile the model
model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=5e-4))
# fit the model
# history = model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0)
history = model.fit(
    X_train,
    y_train,
    validation_split=0.3,
    verbose=1,
    epochs=500,
    batch_size=1024)
# evaluate the model
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('mse')
pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['mse'], label='val')
pyplot.legend()
pyplot.show()

error = model.evaluate(X_test, y_test, verbose=1)
print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))
# make a prediction

# red
data_1_real = 60
data_1 = [0.598854317, 0.025323781, 0.375821902, 0.610946971, 0.002072538, 0.386980491, 0.637447009,
          0.052611506, 0.309941485, 0.698569121, 0.000100512, 0.301330366, 0.45742565, 0.020472709, 0.522101641]
# green
data_2_real = 80
data_2 = [0.312043034, 0.683236095, 0.004720871, 0.30547876, 0.691220125, 0.003301115, 0.522626275,
          0.458304144, 0.019069581, 0.324226214, 0.63765771, 0.038116076, 0.342910939, 0.653577518, 0.003511544]
# blue
data_3_real = 20
data_3 = [0.388522969, 0.157471367, 0.454005663, 0.351297611, 0.028966666, 0.619735723, 0.397162807,
          0.098067591, 0.504769602, 0.305144959, 0.051090285, 0.643764756, 0.407238817, 0.065499115, 0.527262068]
# # blue
# data_4_real = 15
# data_4 = [0.322222935, 0.106270295, 0.57150677, 0.306516779, 0.024510988, 0.668972234, 0.398457113,
#           0.129808646, 0.471734241, 0.298992319, 0.102468094, 0.598539588, 0.477228914, 0.078241746, 0.44452934]
# # red
# data_5_real = 10
# data_5 = [0.461602342, 0.220763714, 0.317633944, 0.467878272, 0.19212971, 0.339992018, 0.552750118,
#           0.107630933, 0.339618948, 0.644301531, 0.010229279, 0.34546919, 0.609029355, 0.08578563, 0.305185014]
# # blue
# data_6_real = 15
# data_6 = [0.376088467, 0.095166585, 0.528744948, 0.309565045, 0.146627363, 0.543807591, 0.306093558,
#           9.27E-05, 0.6938137, 0.299377898, 0.004058689, 0.696563413, 0.347978501, 0.008462074, 0.643559424]

# # green
# data_7_real = 10
# data_7 = [0.509907801, 0.470426748, 0.019665451, 0.311677384, 0.527114055, 0.161208561, 0.369441723,
#           0.420288304, 0.210269973, 0.287552048, 0.577571297, 0.134876655, 0.322167564, 0.547748256, 0.13008418]

# # green
# data_8_real = 30
# data_8 = [0.31762137, 0.670424989, 0.011953641, 0.308911101, 0.686229545, 0.004859354, 0.336333421,
#           0.641776244, 0.021890335, 0.295885412, 0.661203934, 0.042910654, 0.380837831, 0.47235254, 0.146809628]

# # red
# data_9_real = 20
# data_9 = [0.514185679, 0.172642152, 0.313172169, 0.594338456, 0.097587397, 0.308074147, 0.631666774,
#           0.006229753, 0.362103473, 0.694278217, 0.017164497, 0.288557286, 0.566423738, 0.103922729, 0.329653533]

# # blue
# data_10_real = 15
# data_10 = [0.320600431, 0.100645978, 0.57875359, 0.333668624, 0.056478575, 0.609852801, 0.317884459,
#            0.049057469, 0.633058072, 0.259122354, 0.051102631, 0.689775015, 0.311835654, 0.014794821, 0.673369525]

yhat1 = model.predict([data_1])
yhat2 = model.predict([data_2])
yhat3 = model.predict([data_3])
# yhat4 = model.predict([data_4])
# yhat5 = model.predict([data_5])
# yhat6 = model.predict([data_6])
# yhat7 = model.predict([data_7])
# yhat8 = model.predict([data_8])
# yhat9 = model.predict([data_9])
# yhat10 = model.predict([data_10])

row = [data_1, yhat1]
writer_uji.writerow(row)
row = [data_2, yhat2]
writer_uji.writerow(row)
row = [data_3, yhat3]
writer_uji.writerow(row)
# row = [data_4, yhat4]
# writer_uji.writerow(row)
# row = [data_5, yhat5]
# writer_uji.writerow(row)
# row = [data_6, yhat6]
# writer_uji.writerow(row)
# row = [data_7, yhat7]
# writer_uji.writerow(row)
# row = [data_8, yhat8]
# writer_uji.writerow(row)
# row = [data_9, yhat9]
# writer_uji.writerow(row)
# row = [data_10, yhat10]
# writer_uji.writerow(row)

print('Predicted: %.3f ' % yhat1)
print('Predicted: %.3f ' % yhat2)
print('Predicted: %.3f ' % yhat3)
# print('Predicted: %.3f ' % yhat4)
# print('Predicted: %.3f ' % yhat5)
# print('Predicted: %.3f ' % yhat6)
# print('Predicted: %.3f ' % yhat7)
# print('Predicted: %.3f ' % yhat8)
# print('Predicted: %.3f ' % yhat9)
# print('Predicted: %.3f ' % yhat10)

model.save('model.h5')
