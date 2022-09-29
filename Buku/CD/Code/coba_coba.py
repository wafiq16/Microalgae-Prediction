
# mlp for multi-output regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from pandas import read_csv
import keras
import csv
from sklearn.preprocessing import StandardScaler
# get the dataset


def get_dataset():
    path = 'data/Pewarna_Satuan.csv'
    df = read_csv(path, header=None)
    # split into input and output columns
    X, y = df.values[:, :-3], df.values[:, -3:]
    X, y = X.astype('float'), y.astype('float')
    # X, y = make_regression(n_samples=1000, n_features=10,
    #    n_informative=5, n_targets=3, random_state=2)
    return X, y

# get the model


def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(500, input_dim=n_inputs,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(140,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(130,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(120,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(110,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(100,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(90,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(80,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(70,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(60,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(50,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(40,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(30,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(20,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(10,
              kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    return model

# evaluate a model using repeated k-fold cross-validation


def evaluate_model(X, y):
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)

    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        model.fit(X_train, y_train, validation_data=(
            X_test, y_test), verbose=1, epochs=300)
        # evaluate model on test set
        mae = model.evaluate(X_test, y_test, verbose=1)
        # store result
        print('>%.3f' % mae)
        model.save('coba2.h5')
        results.append(mae)
    return results


# load dataset
X, y = get_dataset()
# evaluate model
results = evaluate_model(X, y)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(results), std(results)))

model = keras.models.load_model("coba2.h5")

inpt = [
    [0, 0, 0],
    [0, 0, 100],
    [100, 0, 0],
    [0, 100, 0],
    # [20, 80, 0],
    # [40, 60, 0],
    # [60, 40, 0],
    # [80, 20, 0],
    # [0, 20, 80],
    # [0, 40, 60],
    # [0, 60, 40],
    # [0, 80, 20],
    # [20, 0, 80],
    # [40, 0, 60],
    # [60, 0, 40],
    # [80, 0, 20],
    # [20, 60, 20],
    # [20, 40, 40],
    # [20, 20, 60],
    # [40, 40, 20],
    # [40, 20, 40],
    # [60, 20, 20],
    # [33.3, 33.3, 33.3],
    [10, 0, 0],
    [20, 0, 0],
    [50, 0, 0],
    [0, 10,	0],
    [0, 20,	0],
    [0, 50,	0],
    [0, 0, 10],
    [0, 0, 20],
    [0, 0, 50]
]

row = [
    [202, 208, 205, 206, 209, 209, 99, 93, 77, 99, 93, 77, 87, 85, 87],
    [0, 214, 231, 0, 224, 232, 52, 77, 53, 52, 77, 53, 38, 112, 112],
    [254, 138, 101, 255, 143, 103, 92, 65, 46, 92, 65, 46, 134, 44, 81],
    [0, 246, 65, 0, 248, 66, 51, 73, 33, 51, 73, 33, 64, 116, 72],
    # [22, 227, 65, 24, 235, 69, 86, 91, 51, 86, 91, 51, 72, 106, 81],
    # [92, 140, 41, 92, 140, 41, 104, 100, 61, 104, 100, 61, 85, 89, 81],
    # [245, 195, 97, 255, 202, 103, 105, 97, 60, 105, 97, 60, 90, 82, 86],
    # [254, 135, 89, 255, 143, 88, 109, 97, 63, 109, 97, 63, 98, 78, 86],
    # [0, 244, 164, 0, 250, 168, 71, 110, 64, 71, 110, 64, 48, 114, 98],
    # [0, 244, 106, 0, 251, 111, 79, 109, 62, 79, 109, 62, 49, 119, 90],
    # [0, 248, 146, 0, 255, 177, 76, 108, 61, 76, 108, 61, 50, 120, 89],
    # [0, 241, 100, 0, 255, 125, 81, 110, 61, 81, 110, 61, 56, 121, 81],
    # [0, 167, 253, 0, 186, 255, 85, 103, 70, 85, 103, 70, 46, 97, 115],
    # [52, 141, 253, 66, 171, 255, 101, 101, 70, 101, 101, 70, 69, 80, 109],
    # [56, 114, 251, 69, 130, 255, 103, 99, 70, 103, 99, 70, 75, 79, 109],
    # [251, 132, 253, 255, 147, 255, 117, 102, 76, 117, 102, 76, 96, 64, 99],
    # [0, 188, 60, 0, 214, 72, 109, 119, 85, 109, 119, 85, 69, 107, 83],
    # [0, 172, 105, 0, 174, 105, 86, 103, 66, 86, 103, 66, 58, 103, 98],
    # [0, 147, 115, 0, 176, 137, 87, 105, 63, 87, 105, 63, 59, 102, 98],
    # [0, 153, 75, 0, 166, 84, 95, 103, 62, 95, 103, 62, 75, 94, 90],
    # [0, 133, 132, 0, 133, 136, 83, 99, 60, 83, 99, 60, 62, 98, 98],
    # [108, 112, 110, 118, 123, 121, 99, 99, 61, 99, 99, 61, 86, 82, 91],
    # [28, 141, 156, 34, 150, 167, 91, 96, 60, 91, 96, 60, 76, 89, 93],
    [254, 108, 164, 255, 112, 173, 142, 134, 111, 145, 178, 87, 100, 74, 87],
    [254, 131, 218, 255, 136, 233, 142, 128, 104, 147, 171, 87, 103, 72, 83],
    [254, 130, 229, 255, 132, 228, 139, 122, 102, 148, 158, 89, 109, 63, 87],
    [60, 190, 100, 65, 187, 106, 117, 124, 89, 127, 169, 63, 80, 95, 76],
    [95, 215, 129, 95, 218, 132, 118, 124, 89, 128, 170, 61, 88, 90, 82],
    [0, 249, 111, 0, 255, 119, 81, 112, 60, 92, 156, 31, 69, 112, 77],
    [0, 237, 252, 0, 255, 255, 83, 106, 80, 93, 149, 47, 51, 100, 106],
    [0, 181, 236, 0, 204, 255, 79, 107, 79, 78, 149, 39, 53, 103, 106],
    [0, 157, 244, 0, 161, 255, 62, 97, 75, 73, 141, 43, 36, 111, 111],
]
y = 0
for i in row:
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
    with open('dnn.csv', 'a', newline='') as outfile:
        # writer = csv.writer(outfile)
        # writer = csv.writer(f)
        writer_reg = csv.writer(outfile)
        writer_reg.writerow(r)
    y += 1
