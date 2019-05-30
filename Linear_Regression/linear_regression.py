# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 20:35:29 2019

@author: SonyTF
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset =pd.read_csv("Salary.csv")

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

from sklearn.linear_model import LinearRegression
regressor =LinearRegression()

regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

#visualisation of the training data
plt.figure()
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Exp (training set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()

# visualisation of the test data
plt.figure()
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Exp (testing set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()


# predicting the test set results
y_pred=regressor.predict(X_test)
print('Coefficients =', regressor.coef_)
print('Intercept =' , regressor.intercept_)
print('Mean Squared error =  %.2f' % np.mean((y_pred-y_test)**2))
print('Variance score: %.2f' % regressor.score(X_test,y_test))


from sklearn.metrics import mean_squared_error
mse= mean_squared_error(y_pred,y_test)