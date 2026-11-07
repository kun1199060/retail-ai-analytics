import pandas as pd
import numpy as np
import os

print("=" * 55)
print("RETAIL AI PROJECT - FEATURE ENGINEERING")
print("=" * 55)

# Create processed folder
os.makedirs("data/processed", exist_ok=True)

# Load training dataset
print("\nLoading dataset...")
df = pd.read_csv("data/raw/train.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Sort data correctly
df = df.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)

print("Dataset loaded:", df.shape)

# 1. Date-based features
print("\nCreating date features...")

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int16")
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype("int8")

# 2. Lag features
print("Creating lag features...")

group = df.groupby(
    ["store_nbr", "family"],
    sort=False
)["sales"]

df["lag_7"] = group.shift(7)
df["lag_14"] = group.shift(14)
df["lag_30"] = group.shift(30)

# 3. Rolling mean features
print("Creating rolling mean features...")

df["rolling_mean_7"] = (
    df.groupby(["store_nbr", "family"], sort=False)["sales"]
    .transform(
        lambda x: x.shift(1).rolling(7, min_periods=7).mean()
    )
)

df["rolling_mean_30"] = (
    df.groupby(["store_nbr", "family"], sort=False)["sales"]
    .transform(
        lambda x: x.shift(1).rolling(30, min_periods=30).mean()
    )
)

# 4. Fill missing historical features
print("Handling missing feature values...")

feature_columns = [
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30"
]

df[feature_columns] = df[feature_columns].fillna(0)

# 5. Save processed dataset
print("\nSaving processed dataset...")

output_path = "data/processed/retail_features.csv"

df.to_csv(output_path, index=False)

print("\n" + "=" * 55)
print("FEATURE ENGINEERING COMPLETED!")
print("=" * 55)

print("\nFinal Dataset Shape:", df.shape)

print("\nNew Features:")
print(feature_columns)

print("\nFirst 5 Records:")
print(df.head())

print("\nSaved at:", output_path)