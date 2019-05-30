u# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 23:01:00 2019

@author: SonyTF
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("MFG10YearTerminationData.csv")
df.isna().sum()

# ============================================================================= 
# df_new=df.drop(['EmployeeID',],axis=1)
# # =============================================================================

df_new=df[['age','length_of_service','gender_full','STATUS_YEAR','BUSINESS_UNIT','STATUS']].copy()

X=df_new.iloc[:,:-1].values
y=df_new.iloc[:,-1].values

# =============================================================================
# #df_new['BUSINESS_UNIT'].unique()
# #->array(['HEADOFFICE', 'STORES'], dtype=object)
# df_new['STATUS_YEAR'].unique()
# # -> array([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015],
#      dtype=int64)
# =============================================================================

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
labelencoder_y1 = LabelEncoder()

X[:,2]= labelencoder_x1.fit_transform(X[:,2])
X[:,3]=labelencoder_x1.fit_transform(X[:,3])

X[:,4]=labelencoder_x1.fit_transform(X[:,4])

y=labelencoder_y1.fit_transform(y)

onehotencoder=OneHotEncoder(categorical_features=[3]) # change the years column into numerical
X=onehotencoder.fit_transform(X).toarray()


# =============================================================================
# X[:, 3] = labelencoder.fit_transform(X[:, 3])
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()
# =============================================================================


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# 
# =============================================================================
  # Feature Scaling
  from sklearn.preprocessing import StandardScaler
  sc_X = StandardScaler()
  X_train = sc_X.fit_transform(X_train)
  X_test = sc_X.transform(X_test)
  sc_y = StandardScaler()
  y_train = sc_y.fit_transform(y_train)
# =============================================================================
# 
# =============================================================================
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, y)

y_pred = regressor.predict(X_test)


# Predicting a new result

y_pred = regressor.predict(6.5)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()






