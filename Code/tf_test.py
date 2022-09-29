from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset
dataframe = read_csv("data/data_train.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, :-3]
y = dataset[:, -3:]
# define wider model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)


def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(10000, input_dim=15, activation='relu',
                    kernel_initializer='random_uniform'))
    model.add(Dense(1000, activation='relu',
              kernel_initializer='random_uniform'))
    model.add(Dense(250, activation='relu',
              kernel_initializer='random_uniform'))
    model.add(Dense(3, activation='linear',
              kernel_initializer='random_uniform'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(
    model=wider_model, epochs=100, batch_size=32, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, Y, cv=kfold,
                          scoring='neg_mean_squared_error')
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
