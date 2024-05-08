"""QUESTION 2 - LOGISTIC REGRESSION"""

df1 = pd.read_csv("/content/booking.csv")
df1.head()

df1.shape

df1.info()

df1.describe(include='all')

df1.isnull().sum()

df1.duplicated().sum()

"""UNIVARIATE ANALYSIS"""

#HISTOGRAM
for column in df1.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df1[column])
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

#BAR PLOT
for column in df1.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(10, 5))
    df1[column].value_counts().plot(kind='bar')
    plt.title(f'Bar Chart of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

df1.columns

df1['average price']=df1['average price'].fillna(df1['average price'].median())
df1['room type']=df1['room type'].fillna(df1['room type'].mode())

"""BIVARIATE ANALYSIS"""

# Generate scatter plots for pairs of numerical variables
numerical_columns = df1.select_dtypes(include=['float64', 'int64']).columns
for i in range(len(numerical_columns)):
    for j in range(i + 1, len(numerical_columns)):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df1, x=numerical_columns[i], y=numerical_columns[j])
        plt.title(f'Scatter Plot between {numerical_columns[i]} and {numerical_columns[j]}')
        plt.show()

#box plot
numerical_columns = df1.select_dtypes(include=['float64', 'int64']).columns

for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df1[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()

df1.columns

# Calculate Q1, Q3, and IQR
q1 = np.percentile(df1["number of weekend nights"]	, 25)
q3 = np.percentile(df1["number of weekend nights"]	, 75)
iqr = q3 - q1

# Calculate outlier bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower Bound (Outlier):", lower_bound)
print("Upper Bound (Outlier):", upper_bound)

# Calculate Q1, Q3, and IQR
q1 = np.percentile(df1["lead time"]	, 25)
q3 = np.percentile(df1["lead time"]	, 75)
iqr = q3 - q1

# Calculate outlier bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower Bound (Outlier):", lower_bound)
print("Upper Bound (Outlier):", upper_bound)

# Calculate Q1, Q3, and IQR
q1 = np.percentile(df1["average price"]	, 25)
q3 = np.percentile(df1["average price"]	, 75)
iqr = q3 - q1

# Calculate outlier bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower Bound (Outlier):", lower_bound)
print("Upper Bound (Outlier):", upper_bound)

# Calculate Q1, Q3, and IQR
q1 = np.percentile(df1["special requests"]	, 25)
q3 = np.percentile(df1["special requests"]	, 75)
iqr = q3 - q1

# Calculate outlier bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower Bound (Outlier):", lower_bound)
print("Upper Bound (Outlier):", upper_bound)

numerical_columns = df1.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df1[numerical_columns].corr()
print("Correlation matrix:\n", correlation_matrix)

plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='inferno')
plt.title('Heatmap of Correlation Matrix')
plt.show()

from sklearn.preprocessing import LabelEncoder
import pandas as pd

label_encoder = LabelEncoder()
categories = df1.select_dtypes(include=['object']).columns

# Fit and transform the target column

for column in categories:
    df1[column] = label_encoder.fit_transform(df1[column])
df1

df1.columns

X = df1.drop(columns = ['booking status'])
y = df1[['booking status']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# F1-score
f1_score = f1_score(y_test, y_pred)
print("F1-score:", f1_score)