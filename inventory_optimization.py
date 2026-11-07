import pandas as pd
import numpy as np
import os

print("=" * 60)
print("RETAIL AI - INVENTORY OPTIMIZATION")
print("=" * 60)

# Load simulated inventory dataset
df = pd.read_csv(
    "data/processed/stockout_dataset.csv"
)

print("\nDataset Loaded Successfully!")

print("\nDataset Shape:", df.shape)

print("\nAvailable Columns:")
print(df.columns.tolist())

print("\nSample Inventory Records:")
print(
    df[
        [
            "store_nbr",
            "family",
            "daily_demand",
            "available_inventory",
            "lead_time",
            "lead_time_demand",
            "stockout_risk"
        ]
    ].head()
)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDATA INSPECTION COMPLETED!")


# ==========================================
# STEP 2: INVENTORY OPTIMIZATION CALCULATIONS
# ==========================================

print("\n" + "=" * 60)
print("INVENTORY OPTIMIZATION CALCULATIONS")
print("=" * 60)

# Average Daily Demand
df["average_daily_demand"] = df["rolling_mean_30"]

# Demand variability
df["demand_std"] = df["rolling_mean_30"] * 0.20

# Service level assumption: approximately 95%
service_factor = 1.65

# Safety Stock
df["safety_stock"] = (
    service_factor *
    df["demand_std"] *
    np.sqrt(df["lead_time"])
)

# Reorder Point
df["reorder_point"] = (
    df["average_daily_demand"] * df["lead_time"]
    + df["safety_stock"]
)

# Recommended Order Quantity
df["recommended_order_quantity"] = np.maximum(
    0,
    np.ceil(
        df["reorder_point"]
        - df["available_inventory"]
    )
)

df["inventory_status"] = np.where(
    df["average_daily_demand"] <= 0,
    "No Demand",
    np.where(
        df["available_inventory"] <= df["reorder_point"],
        "Reorder Required",
        "Stock Sufficient"
    )
)

print("\nInventory Optimization Results:")

print(
    df[
        [
            "store_nbr",
            "family",
            "average_daily_demand",
            "available_inventory",
            "safety_stock",
            "reorder_point",
            "recommended_order_quantity",
            "inventory_status"
        ]
    ].head(10).to_string(index=False)
)

print("\nInventory Status Distribution:")
print(df["inventory_status"].value_counts())

print("\nSTEP 2 COMPLETED!")

# ==========================================
# STEP 3: SAVE OPTIMIZED INVENTORY DATA
# ==========================================

import os

os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports", exist_ok=True)

output_path = "data/processed/optimized_inventory.csv"

df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("OPTIMIZED INVENTORY DATA SAVED")
print("=" * 60)

print("File:", output_path)
print("Total Records:", len(df))

print("\nFinal Inventory Status:")
print(df["inventory_status"].value_counts())

print("\nSTEP 3 COMPLETED!")


# ==========================================
# STEP 4: INVENTORY ANALYSIS VISUALIZATION
# ==========================================

import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

# Chart 1: Inventory Status Distribution
status_counts = df["inventory_status"].value_counts()

plt.figure(figsize=(9, 6))
status_counts.plot(kind="bar")

plt.title("Inventory Status Distribution")
plt.xlabel("Inventory Status")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/inventory_status_distribution.png")
plt.close()


# Chart 2: Top 10 Product Categories by Order Quantity
category_orders = (
    df.groupby("family")["recommended_order_quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(11, 6))
category_orders.plot(kind="bar")

plt.title("Top 10 Categories by Recommended Order Quantity")
plt.xlabel("Product Category")
plt.ylabel("Recommended Order Quantity")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("outputs/top_10_reorder_categories.png")
plt.close()


print("\nCHARTS GENERATED SUCCESSFULLY!")
print("1. outputs/inventory_status_distribution.png")
print("2. outputs/top_10_reorder_categories.png")

print("\nSTEP 4 COMPLETED!")


# ==========================================
# STEP 5: BUSINESS INSIGHTS
# ==========================================

print("\n" + "=" * 60)
print("RETAIL INVENTORY BUSINESS INSIGHTS")
print("=" * 60)

# 1. Overall Inventory Summary
total_records = len(df)

status_summary = (
    df["inventory_status"]
    .value_counts()
    .to_frame("record_count")
)

status_summary["percentage"] = (
    status_summary["record_count"] / total_records * 100
)

print("\n1. INVENTORY STATUS SUMMARY")
print(status_summary.round(2))

# 2. Top 10 Categories Requiring Inventory
top_categories = (
    df.groupby("family")["recommended_order_quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n2. TOP 10 CATEGORIES REQUIRING REORDER")
print(top_categories)

# 3. Top 10 Stores Requiring Inventory
top_stores = (
    df.groupby("store_nbr")["recommended_order_quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n3. TOP 10 STORES REQUIRING REORDER")
print(top_stores)

# 4. Total Recommended Order Quantity
total_order_quantity = df["recommended_order_quantity"].sum()

print("\n4. TOTAL RECOMMENDED ORDER QUANTITY")
print(round(total_order_quantity, 2))

# 5. Save Business Insights
status_summary.to_csv(
    "reports/inventory_status_summary.csv"
)

top_categories.to_csv(
    "reports/top_reorder_categories.csv"
)

top_stores.to_csv(
    "reports/top_reorder_stores.csv"
)

print("\nBUSINESS INSIGHT REPORTS SAVED!")
print("1. reports/inventory_status_summary.csv")
print("2. reports/top_reorder_categories.csv")
print("3. reports/top_reorder_stores.csv")

print("\nSTEP 5 COMPLETED!")