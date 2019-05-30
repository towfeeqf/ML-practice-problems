# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:05:16 2019

@author: SonyTF
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
dataset=pd.read_csv("50_Startups.csv")

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values


#pre processing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()

X[:,3]= labelencoder_X.fit_transform(X[:,3])

onehotencoder =OneHotEncoder(categorical_features=[3])

X=onehotencoder.fit_transform(X).toarray()

X=X[:,1:]

#splitting  and partitioning

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

# add column of 1's into the column so that b0 has dependant variable attached 
# so as to make y =b0x0 +b1x1 + .... so on


# numpy.append(arr,values,axis)

# building the optimal model using backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

