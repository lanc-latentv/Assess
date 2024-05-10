# -*- coding: utf-8 -*-
"""lab1_reg.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14fcmhaWEpSJEp7Wa3Zli3DX9UFdfgSnW
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score , mean_squared_error, mean_absolute_error

df= pd.read_csv('/content/Fare prediction.csv')

df.head()

df.shape

df.isnull().sum()

df.dtypes

df.columns

df.duplicated().sum()



"""UNIVARIATE ANALYSIS"""

sns.histplot(df['fare_amount'])

#df=df.drop(columns='key')
#df

"""BIVARIATE ANALYSIS"""

#SCATTER PLOT

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for i in range(len(numerical_columns)):
    for j in range(i + 1, len(numerical_columns)):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=numerical_columns[i], y=numerical_columns[j])
        plt.title(f'Scatter Plot between {numerical_columns[i]} and {numerical_columns[j]}')
        plt.show()

#BOX PLOT
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()

df.columns

list = ['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count']
for column in list:
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lb = Q1-1.5*IQR
  ub = Q3+1.5*IQR
  df = df[(df[column] > lb) & (df[column] < ub)]

"""COREALATION ANALYSIS"""

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numerical_columns].corr()
print("Correlation matrix:\n", correlation_matrix)

plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='inferno')
plt.title('Heatmap of Correlation Matrix')
plt.show()

df.head()

df.dtypes

df.columns



X= df[['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count']]
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

r2 = r2_score(y_test,y_pred_rf)
mse = mean_squared_error(y_test,y_pred_rf)
mae = mean_absolute_error(y_test,y_pred_rf)

print("R2 Score: ",r2)
print("MSE: ",mse)
print("MAE: ",mae)