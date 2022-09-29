# import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# impor CSV ke dataset
path = 'data/dummy_data_cla.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[1:, :-1], df.values[1:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# menggunakan SVM library untuk membuat SVM classifier
classifier = svm.SVC(kernel='linear')

# memasukkan training data kedalam classifier
classifier.fit(X_train, y_train)

# memasukkan testing data ke variabel y_predict
y_predict = classifier.predict(X_test)

# menampilkan classification report
print(classification_report(y_test, y_predict))

# mengimplementasikan testing data dan hasil prediksi dalam confusion matrix
cm = confusion_matrix(y_test, y_predict)

# membuat plotting confusion matrix
# matplotlib inline
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
