import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("RETAIL AI PROJECT - EXPLORATORY DATA ANALYSIS")

# Load data
df = pd.read_csv("data/raw/train.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Basic information
print("\nDataset Shape:", df.shape)
print("\nDate Range:")
print(df["date"].min(), "to", df["date"].max())

# Total sales
print("\nTotal Sales:", df["sales"].sum())

# Sales by product family
family_sales = (
    df.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 Product Categories:")
print(family_sales.head(10))

# Daily sales trend
daily_sales = df.groupby("date")["sales"].sum()

plt.figure(figsize=(14, 6))
daily_sales.plot()
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("outputs/daily_sales_trend.png")
plt.close()

# Top 10 product categories chart
plt.figure(figsize=(12, 6))
family_sales.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Product Categories by Sales")
plt.xlabel("Total Sales")
plt.tight_layout()
plt.savefig("outputs/top_product_categories.png")
plt.close()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Charts saved in outputs folder.")