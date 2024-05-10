# -*- coding: utf-8 -*-
"""lab3_cluster.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fkUBHKVSaaA0MFBVsy2bNvwxrnB6hauu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

df = pd.read_csv("/content/customer_segmentation.csv")
df.head()

df.dtypes



df.shape

df.info()

df.describe(include='all')

df.isna().sum()

df.columns

df.dropna(inplace=True)

#df['Area']=df['Area'].fillna(df['Area'].median())

df.isna().sum()

df.duplicated().sum()

#BAR PLOT
for column in df.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(10, 5))
    df[column].value_counts().plot(kind='bar')
    plt.title(f'Bar Chart of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

df.head()

#SCATTER PLOT

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for i in range(len(numerical_columns)):
    for j in range(i + 1, len(numerical_columns)):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=numerical_columns[i], y=numerical_columns[j])
        plt.title(f'Scatter Plot between {numerical_columns[i]} and {numerical_columns[j]}')
        plt.show()

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()

list = ['ID', 'Year_Birth', 'Income', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumWebVisitsMonth']
for column in list:
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lb = Q1-1.5*IQR
  ub = Q3+1.5*IQR
  df = df[(df[column] > lb) & (df[column] < ub)]

df.columns

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numerical_columns].corr()
print("Correlation matrix:\n", correlation_matrix)

plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='inferno')
plt.title('Heatmap of Correlation Matrix')
plt.show()

df.columns

scaler = MinMaxScaler()
scaler.fit(df[['Recency']])
df['Recency'] = scaler.transform(df[['Recency']])
scaler.fit(df[['ID']])
df['ID'] = scaler.transform(df[['ID']])
print(df.head())
plt.scatter(df['ID'],df['Recency'])

df.columns

sse = [] # The sum of Squared Errors =SSE
k_rng = range(1,10)
for k in k_rng:
   km = KMeans(n_clusters=k)
   km.fit(df[['ID','Recency']])
   sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

df.columns

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['ID','Recency']])
#y_predicted
df['cluster']=y_predicted
#df.head(25)
print(km.cluster_centers_)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1['ID'],df1['Recency'],color='green')
plt.scatter(df2['ID'],df2['Recency'],color='red')
plt.scatter(df3['ID'],df3['Recency'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('ID')
plt.ylabel('Recency')
plt.legend()

silhouette_score(df[['ID','Recency']], km.fit_predict(df[['ID','Recency']]))

