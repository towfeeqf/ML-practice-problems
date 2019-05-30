# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:06:51 2019

@author: SonyTF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

#import data
dataset=pd.read_csv("Admission_Predict.csv")
#dataset2=pd.read_csv("Admission_Predict_Ver1.1.csv")
dataset.isna().sum() # no empty values


#
#### data exploration
plt.figure()
plt.scatter(dataset.iloc[:,1].values,dataset.iloc[:,-1].values)
# the relation is linear so admission is dependant on GRE score

plt.figure()
plt.scatter(dataset.iloc[:,2].values,dataset.iloc[:,-1].values)
# the relation is linear so admission is dependant on TOEFL score



####################
plt.figure()
plt.scatter(dataset.iloc[:,3].values,dataset.iloc[:,-1].values)
# university rating doesn't much impact admission

plt.figure()
plt.scatter(dataset.iloc[:,4].values,dataset.iloc[:,-1].values)
# SOP doesn't much impact admission

########################
plt.figure()
plt.scatter(dataset.iloc[:,6].values,dataset.iloc[:,-1].values)
# Admission does depend on CGPA

########################
plt.figure()
plt.scatter(dataset.iloc[:,7].values,dataset.iloc[:,-1].values)
# RESEARCH doesn't much impact admission



dataset.drop(dataset.columns[[0,3,4,7]], axis=1, inplace=True)

X=dataset.iloc[:,:-1].values 
y=dataset.iloc[:,4].values



# splitting training and testing data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test=sc_y.fit_transform(y_test)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)


# add column of 1's into the column so that b0 has dependant variable attached 
# so as to make y =b0x0 +b1x1 +b1x2 .... so on
# numpy.append(arr,values,axis)

# building the optimal model using backward elimination
import statsmodels.formula.api as sm

X=np.append(arr=np.ones((400,1)).astype(int),values=X,axis=1)

X_opt=X[:,[0,1,2,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


y_pred=regressor.predict(X_test)
print('Coefficients : ',regressor.coef_)
print('Intercept: ', regressor.intercept_)

print('Variance score: %.2f' % regressor.score(X_test,y_test))
