# Titanic Dataset - Data Cleaning & Preprocessing

This project is part of an AI & ML Internship task to clean and preprocess the Titanic dataset to make it suitable for machine learning.

## 📁 Dataset
The dataset used is the Titanic survival data (`titanic.csv`), which includes passenger details such as age, sex, class, fare, etc.

## 🔍 Steps Performed

### 1. Data Exploration
- Loaded the dataset using Pandas
- Checked data types, null values, and basic statistics

### 2. Handling Missing Values
- Filled missing values in `Age` with the mean
- Filled missing values in `Embarked` with the mode
- Dropped `Cabin` column due to too many missing values

### 3. Encoding Categorical Features
- Used one-hot encoding for `Sex` and `Embarked`
- Used Label Encoding for `Pclass` (optional step)

### 4. Feature Scaling
- Standardized `Age` and `Fare` using `StandardScaler` to bring them to mean 0 and unit variance

### 5. Outlier Detection & Removal
- Visualized outliers using boxplots
- Removed outliers in `Fare` column using the IQR method

## ⚙️ Tools Used
- Python
- Google Colab
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

## 📦 Output
A cleaned and preprocessed version of the Titanic dataset stored as `titanic_cleaned.csv`, ready to be used for machine learning tasks.

## 🔗 Project Structure
├── titanic.csv # Original dataset (uploaded manually in Colab)
├── Titanic_Preprocessing.ipynb # Main notebook with all preprocessing steps
├── titanic_cleaned.csv # Output CSV after preprocessing
└── README.md # Project overview and explanation
