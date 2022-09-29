from keras.models import Sequential
from keras.layers import Dense
from numpy import array
from numpy.random import uniform
from numpy import hstack
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas import read_csv
import keras


def create_data(n):
    path = 'data/data_train.csv'
    df = read_csv(path, header=None)
    # split into input and output columns
    X, y = df.values[:, :-3], df.values[:, -3:]
    X, y = X.astype('float'), y.astype('float')
    return X, y


X, Y = create_data(n=450)

plt.plot(Y)
plt.show()

print("X:", X.shape, "Y:", Y.shape)
in_dim = X.shape[1]
out_dim = Y.shape[1]

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.4)
print("xtrain:", xtrain.shape, "ytrian:", ytrain.shape)

model = Sequential()
model = Sequential()
model.add(Dense(10000, input_dim=15, activation='relu',
                kernel_initializer='he_uniform'))
model.add(Dense(1000, activation='relu',
                kernel_initializer='he_uniform'))
model.add(Dense(250, activation='relu',
                kernel_initializer='he_uniform'))
model.add(Dense(3, activation='linear',
                kernel_initializer='he_uniform'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(xtrain, ytrain, epochs=100, batch_size=12, verbose=1)

ypred = model.predict(xtest)
print("y1 MSE:%.4f" % mean_squared_error(ytest[:, 0], ypred[:, 0]))
print("y2 MSE:%.4f" % mean_squared_error(ytest[:, 1], ypred[:, 1]))

x_ax = range(len(xtest))
plt.scatter(x_ax, ytest[:, 0],  s=6, label="y1-test")
plt.plot(x_ax, ypred[:, 0], label="y1-pred")
plt.scatter(x_ax, ytest[:, 1],  s=6, label="y2-test")
plt.plot(x_ax, ypred[:, 1], label="y2-pred")
plt.legend()
plt.show()
