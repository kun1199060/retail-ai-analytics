# ============================================================
# RETAIL AI ANALYTICS
# STEP 6: AI BUSINESS RECOMMENDATION ENGINE
# ============================================================

import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. LOAD OPTIMIZED INVENTORY DATA
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("RETAIL AI - BUSINESS RECOMMENDATION ENGINE")
print("=" * 65)

input_path = "data/processed/optimized_inventory.csv"

if not os.path.exists(input_path):
    raise FileNotFoundError(
        f"Dataset not found: {input_path}"
    )

df = pd.read_csv(input_path)

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# Required columns
required_columns = [
    "store_nbr",
    "family",
    "daily_demand",
    "available_inventory",
    "lead_time",
    "reorder_point",
    "recommended_order_quantity",
    "inventory_status",
    "stockout_risk"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ------------------------------------------------------------
# 2. PREPARE OUTPUT FOLDERS
# ------------------------------------------------------------

os.makedirs("reports", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

recommendations = []

# ------------------------------------------------------------
# 3. CATEGORY-LEVEL BUSINESS RECOMMENDATIONS
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("CATEGORY ANALYSIS")
print("-" * 65)

category_analysis = (
    df.groupby("family")
    .agg(
        total_demand=("daily_demand", "sum"),
        total_inventory=("available_inventory", "sum"),
        total_order_quantity=(
            "recommended_order_quantity", "sum"
        ),
        reorder_records=(
            "inventory_status",
            lambda x: (x == "Reorder Required").sum()
        )
    )
    .sort_values(
        "total_order_quantity",
        ascending=False
    )
)

print("\nTop 10 Categories:")
print(category_analysis.head(10).round(2))

for category, row in category_analysis.head(10).iterrows():

    if row["total_order_quantity"] > 0:
        recommendations.append({
            "recommendation_type": "Category",
            "priority": "High",
            "entity": category,
            "recommended_order_quantity":
                row["total_order_quantity"],
            "recommendation":
                f"Prioritize replenishment of {category}. "
                "It has a high cumulative recommended order quantity."
        })

# ------------------------------------------------------------
# 4. STORE-LEVEL BUSINESS RECOMMENDATIONS
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("STORE ANALYSIS")
print("-" * 65)

store_analysis = (
    df.groupby("store_nbr")
    .agg(
        total_demand=("daily_demand", "sum"),
        total_inventory=("available_inventory", "sum"),
        total_order_quantity=(
            "recommended_order_quantity", "sum"
        ),
        reorder_records=(
            "inventory_status",
            lambda x: (x == "Reorder Required").sum()
        )
    )
    .sort_values(
        "total_order_quantity",
        ascending=False
    )
)

print("\nTop 10 Stores:")
print(store_analysis.head(10).round(2))

for store, row in store_analysis.head(10).iterrows():

    if row["total_order_quantity"] > 0:
        recommendations.append({
            "recommendation_type": "Store",
            "priority": "High",
            "entity": f"Store {store}",
            "recommended_order_quantity":
                row["total_order_quantity"],
            "recommendation":
                f"Review replenishment requirements for Store {store}. "
                "It has one of the highest cumulative reorder quantities."
        })

# ------------------------------------------------------------
# 5. STOCKOUT RISK RECOMMENDATIONS
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("STOCKOUT RISK ANALYSIS")
print("-" * 65)

stockout_records = df[
    (df["stockout_risk"] == 1) &
    (df["daily_demand"] > 0)
].copy()

print(
    "\nRecords flagged with simulated stockout risk:",
    len(stockout_records)
)

print("\nTop 10 Stockout Risk Records:")

stockout_records = stockout_records.sort_values(
    "recommended_order_quantity",
    ascending=False
)

print(
    stockout_records[
        [
            "store_nbr",
            "family",
            "daily_demand",
            "available_inventory",
            "lead_time",
            "recommended_order_quantity"
        ]
    ].head(10).round(2).to_string(index=False)
)

for _, row in stockout_records.head(20).iterrows():

    recommendations.append({
        "recommendation_type": "Stockout Risk",
        "priority": "Critical",
        "entity":
            f"Store {row['store_nbr']} - {row['family']}",
        "recommended_order_quantity":
            row["recommended_order_quantity"],
        "recommendation":
            "Review inventory availability and lead-time demand. "
            "This record is flagged by the simulated stockout rule."
    })

# ------------------------------------------------------------
# 6. ZERO-DEMAND RECOMMENDATIONS
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("ZERO-DEMAND ANALYSIS")
print("-" * 65)

zero_demand_records = df[
    df["inventory_status"] == "No Demand"
]

print(
    "\nRecords with No Demand:",
    len(zero_demand_records)
)

recommendations.append({
    "recommendation_type": "Inventory Management",
    "priority": "Medium",
    "entity": "Zero-Demand Products",
    "recommended_order_quantity": 0,
    "recommendation":
        "Avoid automatic replenishment for zero-demand records. "
        "Review historical demand before placing new orders."
})

# ------------------------------------------------------------
# 7. INVENTORY SHORTAGE RECOMMENDATIONS
# ------------------------------------------------------------

reorder_records = df[
    df["inventory_status"] == "Reorder Required"
]

print(
    "\nTotal Records Requiring Reorder:",
    len(reorder_records)
)

recommendations.append({
    "recommendation_type": "Inventory Management",
    "priority": "High",
    "entity": "Reorder Required Records",
    "recommended_order_quantity":
        reorder_records["recommended_order_quantity"].sum(),
    "recommendation":
        "Review reorder recommendations and prioritize records "
        "with the largest replenishment requirements."
})

# ------------------------------------------------------------
# 8. CREATE FINAL RECOMMENDATION REPORT
# ------------------------------------------------------------

recommendation_df = pd.DataFrame(recommendations)

recommendation_df = recommendation_df.sort_values(
    by="recommended_order_quantity",
    ascending=False
)

print("\n" + "=" * 65)
print("FINAL BUSINESS RECOMMENDATIONS")
print("=" * 65)

print("\nTotal Recommendations:", len(recommendation_df))

print(
    recommendation_df.head(20).to_string(index=False)
)

# ------------------------------------------------------------
# 9. SAVE RECOMMENDATION REPORTS
# ------------------------------------------------------------

recommendation_path = (
    "reports/business_recommendations.csv"
)

category_path = (
    "reports/category_business_analysis.csv"
)

store_path = (
    "reports/store_business_analysis.csv"
)

stockout_path = (
    "reports/stockout_risk_records.csv"
)

recommendation_df.to_csv(
    recommendation_path,
    index=False
)

category_analysis.to_csv(category_path)

store_analysis.to_csv(store_path)

stockout_records.to_csv(
    stockout_path,
    index=False
)

# ------------------------------------------------------------
# 10. GENERATE EXECUTIVE SUMMARY
# ------------------------------------------------------------

total_records = len(df)

reorder_count = (
    df["inventory_status"] == "Reorder Required"
).sum()

no_demand_count = (
    df["inventory_status"] == "No Demand"
).sum()

sufficient_count = (
    df["inventory_status"] == "Stock Sufficient"
).sum()

total_order_quantity = (
    df["recommended_order_quantity"].sum()
)

summary = f"""
============================================================
RETAIL AI ANALYTICS - BUSINESS RECOMMENDATION REPORT
============================================================

DATASET SUMMARY

Total Records: {total_records}

INVENTORY STATUS

Reorder Required: {reorder_count}
No Demand: {no_demand_count}
Stock Sufficient: {sufficient_count}

REORDER PERCENTAGE

{(reorder_count / total_records) * 100:.2f}%

TOTAL RECOMMENDED ORDER QUANTITY

{total_order_quantity:,.2f}

TOP 5 CATEGORIES

{category_analysis.head(5).to_string()}

TOP 5 STORES

{store_analysis.head(5).to_string()}

STOCKOUT RISK RECORDS

{len(stockout_records)}

BUSINESS RECOMMENDATIONS

1. Prioritize high-order-quantity product categories.
2. Review stores with high replenishment requirements.
3. Monitor records flagged with simulated stockout risk.
4. Avoid automatic replenishment for zero-demand records.
5. Review reorder quantities before placing purchase orders.

IMPORTANT DATA LIMITATION

Inventory quantities, lead times, demand variability, and
stockout labels are simulated or assumption-based.

These recommendations are analytical demonstrations and
should not be treated as actual purchase orders.

============================================================
END OF REPORT
============================================================
"""

summary_path = "reports/executive_business_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write(summary)

print("\n" + "=" * 65)
print("FILES SAVED SUCCESSFULLY")
print("=" * 65)

print("1.", recommendation_path)
print("2.", category_path)
print("3.", store_path)
print("4.", stockout_path)
print("5.", summary_path)

print("\n" + "=" * 65)
print("STEP 6 COMPLETED SUCCESSFULLY!")
print("=" * 65)