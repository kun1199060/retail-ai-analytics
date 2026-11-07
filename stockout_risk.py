import pandas as pd
import numpy as np
import os

print("=" * 60)
print("RETAIL AI - STOCKOUT RISK DATA GENERATION")
print("=" * 60)

# Load processed retail data
df = pd.read_csv(
    "data/processed/retail_features.csv"
)

print("\nOriginal Dataset Shape:", df.shape)

# Use a manageable sample
df = df.sample(
    n=200000,
    random_state=42
).copy()

# Simulate inventory using historical demand
np.random.seed(42)

df["daily_demand"] = df["rolling_mean_7"].clip(lower=0)

# Simulated available inventory
df["available_inventory"] = (
    df["daily_demand"] *
    np.random.uniform(0.5, 2.0, size=len(df))
)

# Simulated lead time
df["lead_time"] = np.random.randint(
    1, 8, size=len(df)
)

# Estimated demand during lead time
df["lead_time_demand"] = (
    df["daily_demand"] * df["lead_time"]
)

# Calculate inventory coverage
df["inventory_coverage"] = (
    df["available_inventory"] /
    (df["daily_demand"] + 1)
)

# Create simulated stockout-risk labels
df["stockout_risk"] = (
    df["available_inventory"] <
    df["lead_time_demand"]
).astype(int)

# Save dataset
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/stockout_dataset.csv",
    index=False
)

print("\nStockout Risk Distribution:")
print(df["stockout_risk"].value_counts())

print("\nStockout Risk Percentage:")
print(
    df["stockout_risk"].value_counts(normalize=True) * 100
)

print("\nDataset Shape:", df.shape)

print("\nDataset saved successfully:")
print("data/processed/stockout_dataset.csv")

print("\nSTOCKOUT DATA GENERATION COMPLETED!")