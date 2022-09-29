# mlp for multiclass classification
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
import csv

f_uji = open('data/data_hasil_class.csv', 'w')
# writer = csv.writer(f)
writer_uji = csv.writer(f_uji)

# load the dataset
path = 'data/dummy_data_cla.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
print(y)
y = LabelEncoder().fit_transform(y)
print(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(n_features, activation='relu',
          kernel_initializer='he_normal', input_shape=(n_features,)))
# model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))

model.summary()

plot_model(model, 'model_class.png', show_shapes=True)

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit the model
history = model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0)
# evaluate the model
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('mse')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.show()
# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
# make a prediction

# green
data_1 = [0.325367934, 0.549562174, 0.125069892, 0.316590403, 0.552327294, 0.131082303, 0.340601434,
          0.496970823, 0.162427743, 0.230646304, 0.674498959, 0.094854737, 0.404955566, 0.441357797, 0.153686637]
# blue
data_2 = [0.363646191, 0.05357212, 0.582781689, 0.387450142, 0.022098326, 0.590451533, 0.470418187,
          0.095637208, 0.433944604, 0.233451872, 0.248419231, 0.518128897, 0.309510471, 0.100113868, 0.590375661]
# red
data_3 = [0.412821549, 0.240270311, 0.34690814, 0.47376371, 0.083849109, 0.442387181, 0.476996386,
          0.042415954, 0.48058766, 0.64017596, 0.215863847, 0.143960193, 0.696470077, 0.001488438, 0.302041484]
# blue
data_4 = [0.322222935, 0.106270295, 0.57150677, 0.306516779, 0.024510988, 0.668972234, 0.398457113,
          0.129808646, 0.471734241, 0.298992319, 0.102468094, 0.598539588, 0.477228914, 0.078241746, 0.44452934]
# red
data_5 = [0.461602342, 0.220763714, 0.317633944, 0.467878272, 0.19212971, 0.339992018, 0.552750118,
          0.107630933, 0.339618948, 0.644301531, 0.010229279, 0.34546919, 0.609029355, 0.08578563, 0.305185014]
# blue
data_6 = [0.376088467, 0.095166585, 0.528744948, 0.309565045, 0.146627363, 0.543807591, 0.306093558,
          9.27E-05, 0.6938137, 0.299377898, 0.004058689, 0.696563413, 0.347978501, 0.008462074, 0.643559424]

# green
data_7 = [0.509907801, 0.470426748, 0.019665451, 0.311677384, 0.527114055, 0.161208561, 0.369441723,
          0.420288304, 0.210269973, 0.287552048, 0.577571297, 0.134876655, 0.322167564, 0.547748256, 0.13008418]

# green
data_8 = [0.31762137, 0.670424989, 0.011953641, 0.308911101, 0.686229545, 0.004859354, 0.336333421,
          0.641776244, 0.021890335, 0.295885412, 0.661203934, 0.042910654, 0.380837831, 0.47235254, 0.146809628]

# red
data_9 = [0.514185679, 0.172642152, 0.313172169, 0.594338456, 0.097587397, 0.308074147, 0.631666774,
          0.006229753, 0.362103473, 0.694278217, 0.017164497, 0.288557286, 0.566423738, 0.103922729, 0.329653533]

# # blue
# data_10 = [0.320600431, 0.100645978, 0.57875359, 0.333668624, 0.056478575, 0.609852801, 0.317884459,
#            0.049057469, 0.633058072, 0.259122354, 0.051102631, 0.689775015, 0.311835654, 0.014794821, 0.673369525]

yhat1 = model.predict([data_1])
yhat2 = model.predict([data_2])
yhat3 = model.predict([data_3])
yhat4 = model.predict([data_4])
yhat5 = model.predict([data_5])
yhat6 = model.predict([data_6])
yhat7 = model.predict([data_7])
yhat8 = model.predict([data_8])
yhat9 = model.predict([data_9])
# yhat10 = model.predict([data_10])

row = data_1, yhat1
writer_uji.writerow(row)
row = data_2, yhat2
writer_uji.writerow(row)
row = data_3, yhat3
writer_uji.writerow(row)
row = data_4, yhat4
writer_uji.writerow(row)
row = data_5, yhat5
writer_uji.writerow(row)
row = data_6, yhat6
writer_uji.writerow(row)
row = data_7, yhat7
writer_uji.writerow(row)
row = data_8, yhat8
writer_uji.writerow(row)
row = data_9, yhat9
writer_uji.writerow(row)
# row = data_10, yhat10
# writer_uji.writerow(row)

print('Predicted: %s (class=%d)' % (yhat1, yhat1.argmax(axis=-1)))
print('Predicted: %s (class=%d)' % (yhat2,  yhat2.argmax(axis=-1)))
print('Predicted: %s (class=%d)' % (yhat3,  yhat3.argmax(axis=-1)))
print('Predicted: %s (class=%d)' % (yhat4,  yhat4.argmax(axis=-1)))
print('Predicted: %s (class=%d)' % (yhat5,  yhat5.argmax(axis=-1)))
print('Predicted: %s (class=%d)' % (yhat6,  yhat6.argmax(axis=-1)))
print('Predicted: %s (class=%d)' % (yhat7,  yhat7.argmax(axis=-1)))
print('Predicted: %s (class=%d)' % (yhat8,  yhat8.argmax(axis=-1)))
print('Predicted: %s (class=%d)' % (yhat9,  yhat9.argmax(axis=-1)))
# print('Predicted: %s (class=%d)' % (yhat10, yhat10.argmax(axis=-1)))
