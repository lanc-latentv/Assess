# -*- coding: utf-8 -*-
"""KMeans_lancy.ipynb

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

df = pd.read_csv("/content/Mall_Customers.csv")
df.head()

df.dtypes

df.shape

df.info()

df.describe(include='all')

df.isna().sum()

df.columns

df['Annual Income (k$)']=df['Annual Income (k$)'].fillna(df['Annual Income (k$)'].median())

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

#SCATTER PLOT

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for i in range(len(numerical_columns)):
    for j in range(i + 1, len(numerical_columns)):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=numerical_columns[i], y=numerical_columns[j])
        plt.title(f'Scatter Plot between {numerical_columns[i]} and {numerical_columns[j]}')
        plt.show()

sns.pairplot(df,corner=True)

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()

# Calculate Q1, Q3, and IQR
q1 = np.percentile(df["Annual Income (k$)"]	, 25)
q3 = np.percentile(df["Annual Income (k$)"]	, 75)
iqr = q3 - q1

# Calculate outlier bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower Bound (Outlier):", lower_bound)
print("Upper Bound (Outlier):", upper_bound)

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
scaler.fit(df[['Spending Score (1-100)']])
df['Spending Score (1-100)'] = scaler.transform(df[['Spending Score (1-100)']])
scaler.fit(df[['Annual Income (k$)']])
df['Annual Income (k$)'] = scaler.transform(df[['Annual Income (k$)']])
print(df.head())
plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'])

df.columns

sse = [] # The sum of Squared Errors =SSE
k_rng = range(1,10)
for k in k_rng:
   km = KMeans(n_clusters=k)
   km.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
   sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

df.columns

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Annual Income (k$)','Spending Score (1-100)']])
#y_predicted
df['cluster']=y_predicted
#df.head(25)
print(km.cluster_centers_)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1['Annual Income (k$)'],df1['Spending Score (1-100)'],color='green')
plt.scatter(df2['Annual Income (k$)'],df2['Spending Score (1-100)'],color='red')
plt.scatter(df3['Annual Income (k$)'],df3['Spending Score (1-100)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

silhouette_score(df[['Annual Income (k$)','Spending Score (1-100)']], km.fit_predict(df[['Annual Income (k$)','Spending Score (1-100)']]))

