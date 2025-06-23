import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# from sklearn.impute import KNNImputer # Removed: Not used in this code

# --- 1. Import the dataset and explore basic info (nulls, data types). ---

# You'll need to download the 'titanic.csv' file and place it in the same directory
# as your Python script, or provide the full path to the file.
# Link to download: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
try:
    df = pd.read_csv('titanic.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'titanic.csv' not found. Please download it from the provided link and place it in the correct directory.")
    # Exit if the dataset is not found, as subsequent steps will fail
    exit()

print("\n--- Basic Info ---")
df.info()

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values Count ---")
print(df.isnull().sum())

print("\n--- Data Types ---")
print(df.dtypes)

# --- 2. Handle missing values using mean/median/imputation. ---

# 'Age': Numerical, some missing values. Median is generally preferred for Age as it's less sensitive to outliers.
df['Age'].fillna(df['Age'].median(), inplace=True)
print("\nMissing values in 'Age' filled with median.")

# 'Embarked': Categorical, a few missing values. Fill with the mode (most frequent value).
# .mode()[0] is used because .mode() can return multiple values if there's a tie.
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
print("Missing values in 'Embarked' filled with mode.")

# 'Cabin': Has a very large number of missing values (approx 77% missing).
# Dropping this column is often the most practical approach unless there's a specific domain
# knowledge reason to try to impute or create a 'has_cabin' feature.
df.drop('Cabin', axis=1, inplace=True)
print("Dropped 'Cabin' column due to high number of missing values.")

# Verify no more missing values (except potentially 'Cabin' if it wasn't dropped but should be)
print("\n--- Missing Values After Handling ---")
print(df.isnull().sum())


# --- 3. Convert categorical features into numerical using encoding. ---

# Drop 'Name' and 'Ticket' as they are typically not useful features for ML models directly.
# 'PassengerId' is just an identifier, also not useful for training a model.
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
print("\nDropped 'Name', 'Ticket', and 'PassengerId' columns.")

# 'Sex': Binary categorical (Male/Female). One-Hot Encoding is good here.
# 'Embarked': Multi-category nominal (S, C, Q). One-Hot Encoding is appropriate.
# drop_first=True is used to avoid multicollinearity (dummy variable trap).
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print("\n--- Data after One-Hot Encoding 'Sex' and 'Embarked' ---")
print(df.head())
print(df.info())


# --- 4. Normalize/standardize the numerical features. ---

# Identify numerical features that need scaling.
# 'Survived' is the target variable.
# 'Sex_male', 'Embarked_Q', 'Embarked_S' are binary (0/1) from one-hot encoding and usually
# do not require standardization as their scale is already fixed and small.
numerical_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\n--- Data after Standardization of numerical features ---")
print(df.head())
print(df.describe()) # Check min, max, mean, std dev to confirm standardization


# --- 5. Visualize outliers using boxplots and remove them. ---

print("\n--- Visualizing Outliers with Boxplots (Before Removal) ---")
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=df[feature])
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# --- Outlier Removal using IQR method for 'Fare' (example) ---
# Note: Outlier removal should be done carefully and based on domain knowledge.
# Removing outliers can sometimes remove valuable information.
# For demonstration, we'll remove outliers in 'Fare'.

# Calculate Q1, Q3, and IQR for 'Fare'
Q1_fare = df['Fare'].quantile(0.25)
Q3_fare = df['Fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare

# Define outlier bounds
lower_bound_fare = Q1_fare - 1.5 * IQR_fare
upper_bound_fare = Q3_fare + 1.5 * IQR_fare

# Filter out the outliers for 'Fare'
# Using .copy() to explicitly create a new DataFrame and avoid SettingWithCopyWarning
df_cleaned = df[(df['Fare'] >= lower_bound_fare) & (df['Fare'] <= upper_bound_fare)].copy()

print(f"\nOriginal DataFrame Shape: {df.shape}")
print(f"DataFrame Shape after Outlier Removal (Fare): {df_cleaned.shape}")

# You can also visualize after removal to confirm
print("\n--- Visualizing Outliers with Boxplots (After Fare Outlier Removal) ---")
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=df_cleaned[feature])
    plt.title(f'Boxplot of {feature} (After Fare Outlier Removal)')
plt.tight_layout()
plt.show()

print("\n--- Final Preprocessed Data Info ---")
print(df_cleaned.info())
print("\n--- Final Preprocessed Data Head ---")
print(df_cleaned.head())