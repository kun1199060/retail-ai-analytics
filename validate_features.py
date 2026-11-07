import pandas as pd
import os

print("=" * 50)
print("FEATURE ENGINEERING - VALIDATION")
print("=" * 50)

file_path = "data/processed/retail_features.csv"

df = pd.read_csv(file_path)

# 1. Dataset information
print("\nDataset Shape:", df.shape)

# 2. Check columns
print("\nAll Columns:")
print(df.columns.tolist())

# 3. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# 5. Check feature statistics
features = [
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30"
]

print("\nFeature Statistics:")
print(df[features].describe())

# 6. Check sample records
print("\nSample Records:")
print(df[
    ["date", "store_nbr", "family", "sales"] + features
].iloc[35:40])

print("\nVALIDATION COMPLETED!")