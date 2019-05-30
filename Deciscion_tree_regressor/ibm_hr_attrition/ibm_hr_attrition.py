# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:55:20 2019

@author: SonyTF
"""
import numpy as np
import pandas as pd
import seaborn as sns
import os

import matplotlib.pyplot as plt
# %matplotlib inline

df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

df=df.drop(['EmployeeCount','EmployeeNumber','StandardHours','Over18'],axis=1)

# =============================================================================
# # categorical columns 
# """
# ['Attrition', 'BusinessTravel','Department','EducationField','Gender','JobRole',
# 'MaritalStatus','OverTime']
# """
# =============================================================================
df_new=pd.get_dummies(df,columns=['Attrition', 'BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'],drop_first=True)

df_X=df_new.drop(['Attrition_Yes'],axis=1)
df_y=df_new[['Attrition_Yes']]
 
X=df_X.iloc[:,0:44].values
y=df_y.iloc[:,0].values
 
# =============================================================================
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 789)

# =============================================================================

# =============================================================================
# =============================================================================
# # # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler(with_mean=True,with_std=True)
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler(with_mean=True,with_std=True)
# y_train = sc_y.fit_transform(y_train)
# 
# =============================================================================
# =============================================================================
# feature scaling

from sklearn.preprocessing import MinMaxScaler
sc_X =MinMaxScaler()#(feature_range=(0,1),copy=True)
X_train=sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#sc_y=MinMaxScaler(feature_range=(0,1),copy=True)
#y_train = sc_y.fit_transform(y_train)
# =============================================================================
# 
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion='mse',min_samples_split=4)
#regressor = DecisionTreeRegressor()

regressor.fit(X, y)

y_pred=regressor.predict(X_test).astype('int64')

from sklearn.metrics import confusion_matrix, classification_report

cm=confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))

# =============================================================================
# 
#from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression()
#clf.fit(np.array(X),y)
# 
#from sklearn.metrics import accuracy_score
# 
#pred_y = clf.predict(X_test)
# 
#accuracy = accuracy_score(y_test,y_pred, normalize=True, sample_weight=None)
#accuracy
## 
# =============================================================================




# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


