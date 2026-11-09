"""
Retail Analytics & AI-Powered Sales Forecasting System
Phase 3: Advanced EDA + Anomaly Detection + Seasonal Trends + Growth Drivers

Run from project root:
    python src/advanced_eda.py

Author: Kunal | Project: Unlox Capstone
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_CSV, OUTPUTS_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Global Plot Style ─────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
PRIMARY = "#1a1a2e"
ACCENT = "#e94560"
ACCENT2 = "#0f3460"
ACCENT3 = "#16213e"
PALETTE = ["#e94560", "#0f3460", "#16213e", "#533483", "#e94560", "#0f3460", "#16213e", "#533483"]
sns.set_palette(PALETTE)

# Dashboard-quality figure defaults
plt.rcParams.update({
    "figure.figsize": (14, 7),
    "figure.dpi": 120,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def save_fig(fig, name, tight=True):
    """Save figure to outputs/ folder."""
    if tight:
        fig.tight_layout()
    path = OUTPUTS_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    log.info("Chart saved: %s", path.name)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_overview(df):
    """Compute high-level business KPIs and save to report."""
    log.info("Computing overview statistics ...")

    overview = {
        "Total Revenue": f"${df['Revenue'].sum():,.2f}",
        "Total Sales (Total_Sales)": f"${df['Total_Sales'].sum():,.2f}",
        "Total Units Sold": f"{df['Units_Sold'].sum():,}",
        "Total Transactions": f"{len(df):,}",
        "Average Revenue per Transaction": f"${df['Revenue'].mean():,.2f}",
        "Average Units per Transaction": f"{df['Units_Sold'].mean():.1f}",
        "Average Discount %": f"{df['Discount_Percentage'].mean():.1f}%",
        "Average Store Rating": f"{df['Store_Rating'].mean():.2f}",
        "Number of Stores": df["Store_ID"].nunique(),
        "Number of Products": df["Product_ID"].nunique(),
        "Number of Categories": df["Product_Category"].nunique(),
        "Number of Brands": df["Brand"].nunique(),
        "Number of Regions": df["Region"].nunique(),
        "Date Range": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        "Holiday Transactions": f"{df['Holiday_Flag'].sum():,} ({df['Holiday_Flag'].mean()*100:.1f}%)",
        "Promotion Applied %": f"{df['Promotion_Applied'].astype(str).str.lower().isin(['yes','true','1']).mean()*100:.1f}%",
    }

    # Save overview report
    report_path = REPORTS_DIR / "overview_statistics.txt"
    lines = ["=" * 60, "RETAIL ANALYTICS - BUSINESS OVERVIEW", "=" * 60, ""]
    for key, val in overview.items():
        lines.append(f"  {key:.<40} {val}")
    lines += ["", "=" * 60]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Print to console
    print("\n" + "=" * 60)
    print("  BUSINESS OVERVIEW")
    print("=" * 60)
    for key, val in overview.items():
        print(f"  {key:.<40} {val}")
    print("=" * 60)

    return overview


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SALES & REVENUE TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_sales_trends(df):
    """Daily, monthly, and quarterly revenue trends."""
    log.info("Plotting sales & revenue trends ...")

    # ── Daily Revenue Trend ───────────────────────────────────────────────────
    daily = df.groupby("Date").agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
    ).reset_index()

    # 7-day moving average
    daily["Revenue_MA7"] = daily["Revenue"].rolling(window=7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(daily["Date"], daily["Revenue"], alpha=0.3, color=ACCENT2, linewidth=0.8, label="Daily Revenue")
    ax.plot(daily["Date"], daily["Revenue_MA7"], color=ACCENT, linewidth=2.5, label="7-Day Moving Average")
    ax.set_title("Daily Revenue Trend with 7-Day Moving Average", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.legend(fontsize=11)
    ax.fill_between(daily["Date"], daily["Revenue_MA7"], alpha=0.1, color=ACCENT)
    save_fig(fig, "01_daily_revenue_trend")

    # ── Monthly Revenue Trend ────────────────────────────────────────────────
    monthly = df.groupby("YearMonth").agg(
        Revenue=("Revenue", "sum"),
        Total_Sales=("Total_Sales", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Transactions=("Date", "count"),
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Revenue bars with line overlay
    colors = [ACCENT if i == monthly["Revenue"].idxmax() else ACCENT2 for i in range(len(monthly))]
    axes[0].bar(monthly["YearMonth"], monthly["Revenue"], color=colors, alpha=0.85, edgecolor="white")
    axes[0].set_title("Monthly Revenue", fontsize=16, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    axes[0].tick_params(axis="x", rotation=45)

    # Annotate highest month
    max_idx = monthly["Revenue"].idxmax()
    max_month = monthly.loc[max_idx, "YearMonth"]
    max_rev = monthly.loc[max_idx, "Revenue"]
    axes[0].annotate(f"Peak: {max_month}\n${max_rev:,.0f}",
                     xy=(max_idx, max_rev), xytext=(max_idx - 2, max_rev * 1.05),
                     fontsize=10, fontweight="bold", color=ACCENT,
                     arrowprops=dict(arrowstyle="->", color=ACCENT))

    # Units sold
    axes[1].plot(monthly["YearMonth"], monthly["Units_Sold"], marker="o", color=ACCENT2, linewidth=2.5, markersize=8)
    axes[1].fill_between(range(len(monthly)), monthly["Units_Sold"], alpha=0.15, color=ACCENT2)
    axes[1].set_title("Monthly Units Sold", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Units Sold")
    axes[1].tick_params(axis="x", rotation=45)

    save_fig(fig, "02_monthly_revenue_trend")

    # ── Quarterly Comparison ─────────────────────────────────────────────────
    quarterly = df.groupby(["Year", "Quarter"]).agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
    ).reset_index()
    quarterly["Quarter_Label"] = quarterly["Year"].astype(str) + "-Q" + quarterly["Quarter"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors_q = [ACCENT2, ACCENT, ACCENT3, "#533483"]
    axes[0].bar(quarterly["Quarter_Label"], quarterly["Revenue"], color=colors_q * 2, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Quarterly Revenue", fontsize=14, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(quarterly["Quarter_Label"], quarterly["Units_Sold"], color=colors_q * 2, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Quarterly Units Sold", fontsize=14, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)

    save_fig(fig, "03_quarterly_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CATEGORY & PRODUCT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_category_analysis(df):
    """Revenue and sales by category, subcategory, and brand."""
    log.info("Plotting category & product analysis ...")

    # ── Category-wise Revenue ─────────────────────────────────────────────────
    cat_rev = df.groupby("Product_Category").agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Avg_Discount=("Discount_Percentage", "mean"),
    ).sort_values("Revenue", ascending=False).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Revenue bar
    colors = sns.color_palette("RdYlBu_r", len(cat_rev))
    bars = axes[0].barh(cat_rev["Product_Category"], cat_rev["Revenue"], color=colors, edgecolor="white")
    axes[0].set_title("Revenue by Product Category", fontsize=14, fontweight="bold")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    axes[0].invert_yaxis()

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        axes[0].text(width * 1.01, bar.get_y() + bar.get_height() / 2,
                     f"${width:,.0f}", va="center", fontsize=10, fontweight="bold")

    # Units sold pie
    wedges, texts, autotexts = axes[1].pie(
        cat_rev["Units_Sold"], labels=cat_rev["Product_Category"],
        autopct="%1.1f%%", colors=colors, startangle=140,
        textprops={"fontsize": 11}, pctdistance=0.8,
    )
    for t in autotexts:
        t.set_fontweight("bold")
    axes[1].set_title("Units Sold by Category", fontsize=14, fontweight="bold")

    save_fig(fig, "04_category_revenue_analysis")

    # ── Top 10 Subcategories ─────────────────────────────────────────────────
    sub_rev = df.groupby("Product_Subcategory").agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
    ).sort_values("Revenue", ascending=False).head(10).reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = sns.color_palette("viridis", 10)
    bars = ax.barh(sub_rev["Product_Subcategory"], sub_rev["Revenue"], color=colors, edgecolor="white")
    ax.set_title("Top 10 Product Subcategories by Revenue", fontsize=16, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.01, bar.get_y() + bar.get_height() / 2,
                f"${width:,.0f}", va="center", fontsize=10, fontweight="bold")
    save_fig(fig, "05_top10_subcategories")

    # ── Brand Performance ────────────────────────────────────────────────────
    brand_rev = df.groupby("Brand").agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Avg_Price=("Unit_Price", "mean"),
    ).sort_values("Revenue", ascending=False).head(15).reset_index()

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = sns.color_palette("magma", 15)
    bars = ax.bar(brand_rev["Brand"], brand_rev["Revenue"], color=colors, edgecolor="white")
    ax.set_title("Top 15 Brands by Revenue", fontsize=16, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, "06_top15_brands")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STORE & REGION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_store_region_analysis(df):
    """Store-wise and region-wise performance."""
    log.info("Plotting store & region analysis ...")

    # ── Store-wise Performance ───────────────────────────────────────────────
    store_rev = df.groupby(["Store_ID", "Store_Location"]).agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Transactions=("Date", "count"),
        Avg_Rating=("Store_Rating", "mean"),
        Avg_Discount=("Discount_Percentage", "mean"),
    ).sort_values("Revenue", ascending=False).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Revenue by store
    colors = sns.color_palette("RdYlGn", len(store_rev))
    bars = axes[0].barh(store_rev["Store_ID"] + " (" + store_rev["Store_Location"] + ")",
                        store_rev["Revenue"], color=colors, edgecolor="white")
    axes[0].set_title("Revenue by Store", fontsize=14, fontweight="bold")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    axes[0].invert_yaxis()

    # Store rating
    axes[1].scatter(store_rev["Avg_Rating"], store_rev["Revenue"],
                    s=store_rev["Transactions"] / 10, c=colors, alpha=0.7, edgecolors="white", linewidth=2)
    for _, row in store_rev.iterrows():
        axes[1].annotate(row["Store_ID"], (row["Avg_Rating"], row["Revenue"]),
                         fontsize=10, fontweight="bold", ha="center", va="bottom")
    axes[1].set_title("Store Rating vs Revenue (bubble = transactions)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Average Store Rating")
    axes[1].set_ylabel("Revenue ($)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    save_fig(fig, "07_store_performance")

    # ── Region-wise Analysis ─────────────────────────────────────────────────
    region_rev = df.groupby("Region").agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Transactions=("Date", "count"),
        Avg_Discount=("Discount_Percentage", "mean"),
    ).sort_values("Revenue", ascending=False).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors_r = sns.color_palette("Set2", len(region_rev))
    axes[0].bar(region_rev["Region"], region_rev["Revenue"], color=colors_r, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Revenue by Region", fontsize=14, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    axes[1].pie(region_rev["Revenue"], labels=region_rev["Region"],
                autopct="%1.1f%%", colors=colors_r, startangle=90,
                textprops={"fontsize": 12}, pctdistance=0.8)
    for t in axes[1].texts:
        t.set_fontweight("bold")
    axes[1].set_title("Revenue Share by Region", fontsize=14, fontweight="bold")

    save_fig(fig, "08_region_analysis")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CUSTOMER & PAYMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_customer_analysis(df):
    """Customer type and payment mode analysis."""
    log.info("Plotting customer & payment analysis ...")

    # ── Customer Type ─────────────────────────────────────────────────────────
    cust_rev = df.groupby("Customer_Type").agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Transactions=("Date", "count"),
        Avg_Discount=("Discount_Percentage", "mean"),
    ).sort_values("Revenue", ascending=False).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors_c = sns.color_palette("coolwarm", len(cust_rev))
    axes[0].bar(cust_rev["Customer_Type"], cust_rev["Revenue"], color=colors_c, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Revenue by Customer Type", fontsize=14, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Payment mode
    pay_rev = df.groupby("Payment_Mode").agg(
        Revenue=("Revenue", "sum"),
        Transactions=("Date", "count"),
    ).sort_values("Revenue", ascending=False).reset_index()

    colors_p = sns.color_palette("YlOrRd", len(pay_rev))
    axes[1].bar(pay_rev["Payment_Mode"], pay_rev["Revenue"], color=colors_p, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Revenue by Payment Mode", fontsize=14, fontweight="bold")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    axes[1].tick_params(axis="x", rotation=45)

    save_fig(fig, "09_customer_payment_analysis")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PROMOTION & DISCOUNT EFFECTIVENESS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_promotion_analysis(df):
    """Promotion effectiveness and discount impact."""
    log.info("Plotting promotion & discount analysis ...")

    # ── Promotion vs No Promotion ────────────────────────────────────────────
    df["Promo_Status"] = df["Promotion_Applied"].astype(str).str.lower().map(
        {"yes": "Promotion Applied", "true": "Promotion Applied", "1": "Promotion Applied"}
    ).fillna("No Promotion")

    promo_rev = df.groupby("Promo_Status").agg(
        Revenue=("Revenue", "sum"),
        Avg_Revenue=("Revenue", "mean"),
        Units_Sold=("Units_Sold", "sum"),
        Transactions=("Date", "count"),
        Avg_Discount=("Discount_Percentage", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors_promo = [ACCENT, ACCENT2]
    axes[0].bar(promo_rev["Promo_Status"], promo_rev["Avg_Revenue"], color=colors_promo[:len(promo_rev)],
                edgecolor="white", linewidth=1.5)
    axes[0].set_title("Average Revenue: Promotion vs No Promotion", fontsize=14, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.2f}"))

    # Discount bucket analysis
    disc_rev = df.groupby("Discount_Bucket").agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Transactions=("Date", "count"),
    ).reset_index()

    colors_disc = sns.color_palette("YlOrRd", len(disc_rev))
    axes[1].bar(disc_rev["Discount_Bucket"].astype(str), disc_rev["Units_Sold"],
                color=colors_disc, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Units Sold by Discount Level", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Discount Bucket")
    axes[1].set_ylabel("Units Sold")
    axes[1].tick_params(axis="x", rotation=45)

    save_fig(fig, "10_promotion_discount_analysis")

    # ── Discount vs Units Sold Scatter ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    sample = df.sample(min(2000, len(df)), random_state=42)
    scatter = ax.scatter(sample["Discount_Percentage"], sample["Units_Sold"],
                         c=sample["Revenue"], cmap="RdYlGn", alpha=0.6, s=30, edgecolors="white", linewidth=0.5)
    plt.colorbar(scatter, label="Revenue ($)", ax=ax)
    ax.set_title("Discount % vs Units Sold (color = Revenue)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Discount Percentage (%)")
    ax.set_ylabel("Units Sold")
    save_fig(fig, "11_discount_vs_units_scatter")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SEASONALITY & TIME PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_seasonality(df):
    """Weekday, monthly, holiday, and seasonal patterns."""
    log.info("Plotting seasonality & time patterns ...")

    # ── Weekday Pattern ──────────────────────────────────────────────────────
    weekday_rev = df.groupby(["DayOfWeek", "DayName"]).agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Transactions=("Date", "count"),
    ).reset_index().sort_values("DayOfWeek")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors_w = sns.color_palette("coolwarm", 7)
    axes[0].bar(weekday_rev["DayName"], weekday_rev["Revenue"], color=colors_w, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Revenue by Day of Week", fontsize=14, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(weekday_rev["DayName"], weekday_rev["Transactions"], color=colors_w, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Transactions by Day of Week", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Number of Transactions")
    axes[1].tick_params(axis="x", rotation=45)

    save_fig(fig, "12_weekday_pattern")

    # ── Holiday Impact ───────────────────────────────────────────────────────
    holiday_rev = df.groupby("Holiday_Flag").agg(
        Revenue=("Revenue", "sum"),
        Avg_Revenue=("Revenue", "mean"),
        Units_Sold=("Units_Sold", "sum"),
        Transactions=("Date", "count"),
    ).reset_index()
    holiday_rev["Holiday_Flag"] = holiday_rev["Holiday_Flag"].map({0: "Non-Holiday", 1: "Holiday"})

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors_h = [ACCENT2, ACCENT]
    axes[0].bar(holiday_rev["Holiday_Flag"], holiday_rev["Avg_Revenue"], color=colors_h[:len(holiday_rev)],
                edgecolor="white", linewidth=1.5)
    axes[0].set_title("Avg Revenue: Holiday vs Non-Holiday", fontsize=14, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.2f}"))

    axes[1].bar(holiday_rev["Holiday_Flag"], holiday_rev["Transactions"], color=colors_h[:len(holiday_rev)],
                edgecolor="white", linewidth=1.5)
    axes[1].set_title("Transactions: Holiday vs Non-Holiday", fontsize=14, fontweight="bold")

    save_fig(fig, "13_holiday_impact")

    # ── Monthly Seasonality Heatmap ──────────────────────────────────────────
    monthly_heatmap = df.groupby(["Year", "Month"]).agg(
        Revenue=("Revenue", "sum"),
    ).reset_index()
    pivot = monthly_heatmap.pivot_table(index="Year", columns="Month", values="Revenue", aggfunc="sum")

    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="YlOrRd", linewidths=1, ax=ax,
                annot_kws={"fontsize": 10, "fontweight": "bold"})
    ax.set_title("Monthly Revenue Heatmap (Year x Month)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    save_fig(fig, "14_monthly_heatmap")

    # ── Category x Month Seasonality ─────────────────────────────────────────
    cat_month = df.groupby(["Product_Category", "Month"]).agg(
        Revenue=("Revenue", "sum"),
    ).reset_index()
    pivot_cm = cat_month.pivot_table(index="Product_Category", columns="Month", values="Revenue", aggfunc="sum")

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot_cm, annot=True, fmt=",.0f", cmap="viridis", linewidths=0.5, ax=ax,
                annot_kws={"fontsize": 8})
    ax.set_title("Category x Month Revenue Heatmap", fontsize=16, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Product Category")
    save_fig(fig, "15_category_month_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def detect_anomalies(df):
    """
    Detect sales anomalies using:
    1. Z-Score method (statistical)
    2. IQR method (robust)
    3. Isolation Forest (ML-based)
    """
    log.info("Running anomaly detection ...")

    daily = df.groupby("Date").agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
    ).reset_index()

    # ── Method 1: Z-Score ────────────────────────────────────────────────────
    daily["Revenue_ZScore"] = stats.zscore(daily["Revenue"])
    daily["ZScore_Anomaly"] = (abs(daily["Revenue_ZScore"]) > 2.5).astype(int)

    # ── Method 2: IQR ────────────────────────────────────────────────────────
    Q1 = daily["Revenue"].quantile(0.25)
    Q3 = daily["Revenue"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    daily["IQR_Anomaly"] = ((daily["Revenue"] < lower_bound) | (daily["Revenue"] > upper_bound)).astype(int)

    # ── Method 3: Isolation Forest ───────────────────────────────────────────
    try:
        from sklearn.ensemble import IsolationForest

        features = daily[["Revenue", "Units_Sold"]].values
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        daily["IsoForest_Anomaly"] = (iso_forest.fit_predict(features) == -1).astype(int)
        log.info("Isolation Forest anomalies: %d", daily["IsoForest_Anomaly"].sum())
    except ImportError:
        log.warning("sklearn not available for Isolation Forest, using Z-Score + IQR only")
        daily["IsoForest_Anomaly"] = 0

    # ── Combined Anomaly Score ───────────────────────────────────────────────
    daily["Anomaly_Score"] = daily["ZScore_Anomaly"] + daily["IQR_Anomaly"] + daily["IsoForest_Anomaly"]
    daily["Is_Anomaly"] = (daily["Anomaly_Score"] >= 2).astype(int)

    anomaly_count = daily["Is_Anomaly"].sum()
    log.info("Total anomalies detected: %d out of %d days", anomaly_count, len(daily))

    # ── Plot Anomalies ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 7))

    # Normal points
    normal = daily[daily["Is_Anomaly"] == 0]
    anomalies = daily[daily["Is_Anomaly"] == 1]

    ax.plot(daily["Date"], daily["Revenue"], color=ACCENT2, alpha=0.5, linewidth=1, label="Daily Revenue")
    ax.scatter(normal["Date"], normal["Revenue"], color=ACCENT2, s=15, alpha=0.6, label="Normal Days")

    # Anomaly points
    ax.scatter(anomalies["Date"], anomalies["Revenue"], color=ACCENT, s=120,
               zorder=5, edgecolors="black", linewidth=1.5, label=f"Anomalies ({anomaly_count})")

    # Annotate top anomalies
    top_anomalies = anomalies.nlargest(5, "Revenue")
    for _, row in top_anomalies.iterrows():
        ax.annotate(f"${row['Revenue']:,.0f}",
                    xy=(row["Date"], row["Revenue"]),
                    xytext=(0, 15), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=ACCENT,
                    ha="center", arrowprops=dict(arrowstyle="->", color=ACCENT))

    # Bounds
    ax.axhline(y=upper_bound, color="red", linestyle="--", alpha=0.5, label=f"Upper Bound (${upper_bound:,.0f})")
    ax.axhline(y=lower_bound, color="red", linestyle="--", alpha=0.5, label=f"Lower Bound (${lower_bound:,.0f})")

    ax.set_title("Anomaly Detection - Daily Revenue (Z-Score + IQR + Isolation Forest)",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.legend(fontsize=10, loc="upper left")

    save_fig(fig, "16_anomaly_detection")

    # ── Save anomaly report ──────────────────────────────────────────────────
    anomaly_report = daily[daily["Is_Anomaly"] == 1].copy()
    anomaly_report = anomaly_report.sort_values("Revenue", ascending=False)
    anomaly_csv = REPORTS_DIR / "anomaly_report.csv"
    anomaly_report.to_csv(anomaly_csv, index=False)
    log.info("Anomaly report saved: %s", anomaly_csv)

    return daily


# ═══════════════════════════════════════════════════════════════════════════════
# 9. GROWTH DRIVERS & CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_growth_drivers(df):
    """Identify key growth drivers through correlation and statistical analysis."""
    log.info("Analyzing growth drivers ...")

    # ── Correlation Matrix ────────────────────────────────────────────────────
    numeric_cols = ["Unit_Price", "Units_Sold", "Total_Sales", "Discount_Percentage",
                    "Revenue", "Stock_On_Hand", "Store_Rating", "Holiday_Flag",
                    "IsWeekend", "Month", "Quarter"]

    available_cols = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[available_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax, annot_kws={"fontsize": 10})
    ax.set_title("Correlation Matrix - Key Business Variables", fontsize=16, fontweight="bold")
    save_fig(fig, "17_correlation_matrix")

    # ── Revenue by Month (YoY Comparison) ────────────────────────────────────
    monthly_yoy = df.groupby(["Year", "Month"]).agg(
        Revenue=("Revenue", "sum"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    for year in sorted(monthly_yoy["Year"].unique()):
        year_data = monthly_yoy[monthly_yoy["Year"] == year]
        ax.plot(year_data["Month"], year_data["Revenue"], marker="o", linewidth=2.5,
                markersize=8, label=str(year))

    ax.set_title("Year-over-Year Monthly Revenue Comparison", fontsize=16, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.legend(title="Year", fontsize=12)
    save_fig(fig, "18_yoy_comparison")

    # ── Growth Rate Analysis ─────────────────────────────────────────────────
    monthly_rev = df.groupby("YearMonth")["Revenue"].sum().reset_index()
    monthly_rev["Growth_Rate"] = monthly_rev["Revenue"].pct_change() * 100

    fig, ax = plt.subplots(figsize=(16, 6))
    colors_g = [ACCENT if x < 0 else "#27ae60" for x in monthly_rev["Growth_Rate"].fillna(0)]
    ax.bar(monthly_rev["YearMonth"], monthly_rev["Growth_Rate"].fillna(0), color=colors_g,
           edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_title("Month-over-Month Revenue Growth Rate (%)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Growth Rate (%)")
    ax.tick_params(axis="x", rotation=45)

    # Add average line
    avg_growth = monthly_rev["Growth_Rate"].mean()
    ax.axhline(y=avg_growth, color=ACCENT2, linestyle="--", linewidth=1.5,
               label=f"Average Growth: {avg_growth:.1f}%")
    ax.legend(fontsize=11)

    save_fig(fig, "19_growth_rate_analysis")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. GENERATE INSIGHTS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
def generate_insights(df):
    """Generate key business insights from EDA."""
    log.info("Generating business insights ...")

    insights = []

    # Revenue insights
    total_rev = df["Revenue"].sum()
    avg_daily_rev = df.groupby("Date")["Revenue"].sum().mean()
    insights.append(f"Total revenue over 2 years: ${total_rev:,.2f}")
    insights.append(f"Average daily revenue: ${avg_daily_rev:,.2f}")

    # Top category
    top_cat = df.groupby("Product_Category")["Revenue"].sum().idxmax()
    top_cat_rev = df.groupby("Product_Category")["Revenue"].sum().max()
    insights.append(f"Top performing category: {top_cat} (${top_cat_rev:,.2f})")

    # Top store
    store_rev = df.groupby("Store_ID")["Revenue"].sum()
    top_store = store_rev.idxmax()
    insights.append(f"Top performing store: {top_store} (${store_rev.max():,.2f})")

    # Top region
    region_rev = df.groupby("Region")["Revenue"].sum()
    top_region = region_rev.idxmax()
    insights.append(f"Top performing region: {top_region} (${region_rev.max():,.2f})")

    # Seasonality
    monthly_avg = df.groupby("Month")["Revenue"].mean()
    peak_month = monthly_avg.idxmax()
    low_month = monthly_avg.idxmin()
    month_names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                   7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
    insights.append(f"Peak sales month: {month_names.get(peak_month, peak_month)}")
    insights.append(f"Lowest sales month: {month_names.get(low_month, low_month)}")

    # Weekend vs Weekday
    weekend_rev = df[df["IsWeekend"] == 1]["Revenue"].mean()
    weekday_rev = df[df["IsWeekend"] == 0]["Revenue"].mean()
    insights.append(f"Avg revenue - Weekends: ${weekend_rev:.2f} vs Weekdays: ${weekday_rev:.2f}")

    # Promotion impact
    promo_rev = df[df["Promo_And_Discount"] == 1]["Revenue"].mean()
    no_promo_rev = df[df["Promo_And_Discount"] == 0]["Revenue"].mean()
    if no_promo_rev > 0:
        lift = ((promo_rev - no_promo_rev) / no_promo_rev * 100)
        insights.append(f"Promotion revenue lift: {lift:.1f}%")

    # Holiday impact
    holiday_rev = df[df["Holiday_Flag"] == 1]["Revenue"].mean()
    non_holiday_rev = df[df["Holiday_Flag"] == 0]["Revenue"].mean()
    if non_holiday_rev > 0:
        h_lift = ((holiday_rev - non_holiday_rev) / non_holiday_rev * 100)
        insights.append(f"Holiday revenue lift: {h_lift:.1f}%")

    # Stock status
    low_stock_pct = (df["Stock_On_Hand"] <= 20).mean() * 100
    insights.append(f"Transactions with low stock: {low_stock_pct:.1f}%")

    # Save insights
    report_path = REPORTS_DIR / "eda_insights.txt"
    lines = ["=" * 60, "KEY BUSINESS INSIGHTS - EDA PHASE 3", "=" * 60, ""]
    for i, insight in enumerate(insights, 1):
        lines.append(f"  {i}. {insight}")
    lines += ["", "=" * 60]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Print to console
    print("\n" + "=" * 60)
    print("  KEY BUSINESS INSIGHTS")
    print("=" * 60)
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")
    print("=" * 60)

    return insights


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def run_eda():
    print("\n" + "=" * 60)
    print("  RETAIL AI ANALYTICS - PHASE 3")
    print("  Advanced EDA + Anomaly Detection + Seasonal Trends")
    print("=" * 60 + "\n")

    # Load processed data
    log.info("Loading processed data ...")
    df = pd.read_csv(PROCESSED_CSV, parse_dates=["Date"])
    log.info("Loaded %s rows x %s columns", f"{len(df):,}", df.shape[1])

    # Run all analyses
    compute_overview(df)
    plot_sales_trends(df)
    plot_category_analysis(df)
    plot_store_region_analysis(df)
    plot_customer_analysis(df)
    plot_promotion_analysis(df)
    plot_seasonality(df)
    detect_anomalies(df)
    analyze_growth_drivers(df)
    generate_insights(df)

    # Final summary
    print("\n" + "-" * 60)
    print("  PHASE 3 SUMMARY")
    print("-" * 60)
    print("  Charts generated: 19")
    print(f"  All charts saved to: {OUTPUTS_DIR}")
    print(f"  Reports saved to: {REPORTS_DIR}")
    print("-" * 60)
    print("  PHASE 3 COMPLETE")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    run_eda()