import pandas as pd

# ---------------------------------------
# 1. Load the raw heart disease dataset
# ---------------------------------------

# Define column names (based on UCI documentation)
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'target'
]

# Path to raw dataset
data_path = "data/heart_disease.csv"

# Load the CSV file (replace '?' with NaN)
df = pd.read_csv(data_path, names=columns, na_values='?')

print("✅ Loaded dataset successfully")
print(df.head())

# ---------------------------------------
# 2. Clean the data
# ---------------------------------------

# Convert problematic columns to numeric just in case
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')

# Check missing values
print("\n🕵️ Missing values before drop:")
print(df.isnull().sum())

# Drop rows with any missing values
df = df.dropna()

# ---------------------------------------
# 3. Inspect the cleaned data
# ---------------------------------------

print("\n📊 Dataset Info After Cleaning:")
print(df.info())

print("\n📈 Descriptive Statistics:")
print(df.describe())

print(f"\n🧼 Final number of rows: {df.shape[0]}")

# ---------------------------------------
# 4. Save the cleaned dataset to a new file
# ---------------------------------------

cleaned_path = "data/heart_disease_cleaned.csv"
df.to_csv(cleaned_path, index=False)

print(f"\n💾 Cleaned dataset saved as: {cleaned_path}")