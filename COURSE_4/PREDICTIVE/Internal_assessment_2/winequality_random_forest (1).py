# -*- coding: utf-8 -*-
"""Winequality_random_forest.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1B7oco39RyScObesLKAYXkR7IJ73mlD1i
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

df = pd.read_csv("/content/winequality-red.csv")
df.head()

df.shape

df.info()

df.describe(include='all')

df.isnull().sum()

df.dtypes

df.columns

df['fixed acidity']=df['fixed acidity'].fillna(df['fixed acidity'].median())
df['volatile acidity']=df['volatile acidity'].fillna(df['volatile acidity'].median())
df['citric acid']=df['citric acid'].fillna(df['citric acid'].median())
df['residual sugar']=df['residual sugar'].fillna(df['residual sugar'].median())
df['chlorides']=df['chlorides'].fillna(df['chlorides'].median())
df['free sulfur dioxide']=df['free sulfur dioxide'].fillna(df['free sulfur dioxide'].median())
df['sulphates']=df['sulphates'].fillna(df['sulphates'].median())

df.isna().sum()

df.duplicated().sum()

#treat duplicates
df = df.drop_duplicates()

#BOX PLOT
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()

# Calculate Q1, Q3, and IQR
q1 = np.percentile(df["fixed acidity"]	, 25)
q3 = np.percentile(df["fixed acidity"]	, 75)
iqr = q3 - q1

# Calculate outlier bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower Bound (Outlier):", lower_bound)
print("Upper Bound (Outlier):", upper_bound)

# Map the 'quality' feature into two classes
def map_quality(quality):
    if quality >= 3 and quality <= 6:
        return 0
    elif quality >= 7 and quality <= 8:
        return 1
    else:
        return None

df['quality'] = df['quality'].apply(map_quality)

# Check the distribution of wine quality
print("Wine quality distribution:")
print(df['quality'].value_counts())

# Visualise the distribution of wine quality
plt.figure(figsize=(8, 6))
sns.countplot(x='quality', data=df, palette='viridis')
plt.title('Wine Quality Distribution')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Separate features and target variable
X = df.drop(columns=['quality'])
y = df['quality']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a RF classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")

# Generate a classification report
print(classification_report(y_test, y_pred))

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt="d", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

