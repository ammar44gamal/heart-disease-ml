import pandas as pd
import numpy as np

# ---------------------------------------
# 1. Load the raw heart disease dataset
# ---------------------------------------

# Column names based on the UCI dataset documentation
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'target'
]

# Load the CSV file (after renaming from .data to .csv)
df = pd.read_csv("heart_disease.csv", names=columns)

print("✅ Loaded dataset successfully")
print(df.head())  # Show first 5 rows

# ---------------------------------------
# 2. Clean the data
# ---------------------------------------

# Replace missing values represented by '?' with NaN
df = df.replace('?', np.nan)

# Convert 'ca' and 'thal' columns to numeric (they contain '?')
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')

# Drop rows with any missing values
df = df.dropna()

# ---------------------------------------
# 3. Inspect the cleaned data
# ---------------------------------------

print("\n📊 Dataset Info After Cleaning:")
print(df.info())  # Data types and non-null counts

print("\n📈 Descriptive Statistics:")
print(df.describe())  # Statistical summary

print(f"\n🧼 Final number of rows: {df.shape[0]}")

# ---------------------------------------
# 4. Save the cleaned dataset to a new file
# ---------------------------------------

df.to_csv("heart_disease_cleaned.csv", index=False)
print("\n💾 Cleaned dataset saved as 'heart_disease_cleaned.csv'")
