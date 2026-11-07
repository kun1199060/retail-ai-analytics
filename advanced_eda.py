import pandas as pd
import matplotlib.pyplot as plt
import os

print("=" * 55)
print("RETAIL AI PROJECT - ADVANCED EDA")
print("=" * 55)

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load datasets
df = pd.read_csv("data/raw/train.csv")
stores = pd.read_csv("data/raw/stores.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Merge store information
df = df.merge(stores, on="store_nbr", how="left")

# 1. Monthly Sales Trend
monthly_sales = (
    df.groupby(df["date"].dt.to_period("M"))["sales"]
    .sum()
)

plt.figure(figsize=(12, 5))
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("outputs/monthly_sales_trend.png")
plt.close()

# 2. Weekday-wise Sales
df["weekday"] = df["date"].dt.day_name()

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_sales = (
    df.groupby("weekday")["sales"]
    .mean()
    .reindex(weekday_order)
)

plt.figure(figsize=(10, 5))
weekday_sales.plot(kind="bar")
plt.title("Average Sales by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Average Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/weekday_sales.png")
plt.close()

# 3. Top 10 Stores
store_sales = (
    df.groupby("store_nbr")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))
store_sales.plot(kind="bar")
plt.title("Top 10 Stores by Revenue")
plt.xlabel("Store Number")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("outputs/top_10_stores.png")
plt.close()

# 4. Promotion Analysis
promotion_sales = (
    df.groupby("onpromotion")["sales"]
    .mean()
)

print("\nAVERAGE SALES BY PROMOTION STATUS")
print(promotion_sales)

# 5. Product Category Analysis
category_sales = (
    df.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTOP 10 PRODUCT CATEGORIES")
print(category_sales.head(10))

# 6. Store Type Analysis
store_type_sales = (
    df.groupby("type")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSALES BY STORE TYPE")
print(store_type_sales)

print("\nADVANCED EDA COMPLETED SUCCESSFULLY!")
print("Charts saved in outputs folder.")