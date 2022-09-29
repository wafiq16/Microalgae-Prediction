# get the libraries
from sklearn.svm import SVR  # import the model
# import the StandardScalar class from preprocessing model;sklerarn
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# get the dataset
dataset = pd.read_csv('data/dummy_data_reg.csv')
# our dataset in this implementation is small and thus we can print it all instead of viewing only the end
print("Dataset:")
print(dataset)

# split the data into featutes and target variable seperately
X_l = dataset.iloc[1:, :-1].values  # features set
y_p = dataset.iloc[1:, -1].values  # set of study variable
print("x_p:")
print(X_l)
print("y_p:")
print(y_p)

# change the dimension of y_p to 2D
y_p = y_p.reshape(-1, 1)
# print to see if y_p was really reshaped
print("Reshape y_p:")
print(y_p)

# scale up x_l and y_p seperately to same dimension
StdS_X = StandardScaler()
StdS_y = StandardScaler()
X_l = StdS_X.fit_transform(X_l)
y_p = StdS_y.fit_transform(y_p)
# print scaled variables
print("Scaled X_l:")
print(X_l)
print("Scaled y_p:")
print(y_p)

print(len(X_l))
print(len(y_p))

# # plot the data to have an idea about the relationship between varaibles
# plt.scatter(X_l, y_p, color='red')  # plotting the training set
# plt.title('Scatter Plot')  # adding a tittle to our plot
# plt.xlabel('Levels')  # adds a label to the x-axis
# plt.ylabel('Salary')  # adds a label to the y-axis
# print("Scatter-plot:")
# plt.show()  # prints

# implement SVR
regressor = SVR(kernel='rbf')  # create the model object
regressor.fit(X_l, y_p)  # fit the model on the data

# pediction with SVR
A = regressor.predict(StdS_X.transform([[0.35725933494358664, 0.17347244372148493, 0.46926822133492846, 0.4752566242251104, 0.03302623041869935, 0.49171714535619027, 0.30230817095188195,
                      0.021330065131543636, 0.6763617639165744, 0.30376510458717454, 0.006575824468033709, 0.6896590709447917, 0.3409548711309326, 0.08224088940101869, 0.5768042394680487]]))
print(A)
# Convert A to 2D
A = A.reshape(-1, 1)
print("A")
print(A)
# Taking the inverse of the scaled value
A_pred = StdS_y.inverse_transform(A)
print("A_pred")
print(A_pred)
# this transformation can be performed implicitly as follows
B_pred = StdS_y.inverse_transform(regressor.predict(
    StdS_X.transform([[0.35725933494358664, 0.17347244372148493, 0.46926822133492846, 0.4752566242251104, 0.03302623041869935, 0.49171714535619027, 0.30230817095188195,
                      0.021330065131543636, 0.6763617639165744, 0.30376510458717454, 0.006575824468033709, 0.6896590709447917, 0.3409548711309326, 0.08224088940101869, 0.5768042394680487]])).reshape(-1, 1))
print("B_pred")
print(B_pred)

# Plotting SVR curve
# inverse the transformation to go back to the initial scale

# plt.scatter(StdS_X.inverse_transform(X_l),
#             StdS_y.inverse_transform(y_p), color='red')
plt.plot(StdS_X.inverse_transform(X_l), StdS_y.inverse_transform(
    regressor.predict(X_l).reshape(-1, 1)), color='blue')
# add the title to the plot
plt.title('Support Vector Regression Model')
# label x axis
plt.xlabel('Position')
# label y axis
plt.ylabel('Salary Level')
# print the plot
print("SVR Plot:")
plt.show()
