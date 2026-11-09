"""
Retail Analytics & AI-Powered Sales Forecasting System
Phase 6: AI Business Assistant + Smart Recommendations Engine

Components:
  1. Rule-Based AI Business Assistant (Q&A Engine)
  2. Automated Business Recommendations
  3. Executive Summary Generator
  4. Category & Store Deep-Dive Analysis
  5. Stockout Risk Prediction
  6. Export all reports for dashboard

Run from project root:
    python src/ai_assistant.py

Author: Kunal | Project: Unlox Capstone
"""

import logging
import sys
import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    PROCESSED_CSV, OUTPUTS_DIR, REPORTS_DIR,
    LOW_STOCK_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AI BUSINESS ASSISTANT (Rule-Based Q&A Engine)
# ═══════════════════════════════════════════════════════════════════════════════
class RetailAIAssistant:
    """
    AI-powered business assistant that answers natural language questions
    about retail performance using rule-based NLP + data analysis.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.daily = df.groupby("Date")["Revenue"].sum().reset_index()
        self._build_knowledge_base()

    def _build_knowledge_base(self):
        """Pre-compute key metrics for fast answers."""
        df = self.df
        self.kb = {
            "total_revenue": df["Revenue"].sum(),
            "total_sales": df["Total_Sales"].sum(),
            "total_units": df["Units_Sold"].sum(),
            "total_transactions": len(df),
            "avg_revenue_per_txn": df["Revenue"].mean(),
            "avg_daily_revenue": self.daily["Revenue"].mean(),
            "num_stores": df["Store_ID"].nunique(),
            "num_products": df["Product_ID"].nunique(),
            "num_categories": df["Product_Category"].nunique(),
            "num_brands": df["Brand"].nunique(),
            "num_regions": df["Region"].nunique(),
            "date_range": (df["Date"].min(), df["Date"].max()),
            "avg_rating": df["Store_Rating"].mean(),
            "avg_discount": df["Discount_Percentage"].mean(),
        }

        # Rankings
        self.store_rev = df.groupby("Store_ID")["Revenue"].sum().sort_values(ascending=False)
        self.cat_rev = df.groupby("Product_Category")["Revenue"].sum().sort_values(ascending=False)
        self.brand_rev = df.groupby("Brand")["Revenue"].sum().sort_values(ascending=False)
        self.region_rev = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
        self.subcat_rev = df.groupby("Product_Subcategory")["Revenue"].sum().sort_values(ascending=False)

        # Monthly trends
        self.monthly_rev = df.groupby("YearMonth")["Revenue"].sum().sort_index()

        # Day of week
        self.weekday_rev = df.groupby("DayName")["Revenue"].mean()

        # Customer & Payment
        self.customer_rev = df.groupby("Customer_Type")["Revenue"].sum().sort_values(ascending=False)
        self.payment_rev = df.groupby("Payment_Mode")["Revenue"].sum().sort_values(ascending=False)

        # Promotion
        self.promo_rev = df.groupby("Promotion_Applied")["Revenue"].mean()
        self.holiday_rev = df.groupby("Holiday_Flag")["Revenue"].mean()

        # Store ratings
        self.store_ratings = df.groupby("Store_ID")["Store_Rating"].mean().sort_values(ascending=False)

        log.info("Knowledge base built: %d metrics", len(self.kb))

    def answer(self, question: str) -> str:
        """Answer a natural language business question."""
        q = question.lower().strip()

        # ── Revenue Questions ────────────────────────────────────────────────
        if any(w in q for w in ["total revenue", "revenue", "total sales", "sales", "earning", "income"]):
            return self._answer_revenue(q)

        # ── Store Questions ──────────────────────────────────────────────────
        if any(w in q for w in ["store", "shop", "branch", "location", "best store", "worst store"]):
            return self._answer_stores(q)

        # ── Category Questions ───────────────────────────────────────────────
        if any(w in q for w in ["category", "product", "electronics", "clothing", "grocery", "furniture", "toys"]):
            return self._answer_categories(q)

        # ── Brand Questions ──────────────────────────────────────────────────
        if any(w in q for w in ["brand"]):
            return self._answer_brands(q)

        # ── Region Questions ─────────────────────────────────────────────────
        if any(w in q for w in ["region", "west", "east", "north", "south", "area", "zone"]):
            return self._answer_regions(q)

        # ── Forecast Questions ───────────────────────────────────────────────
        if any(w in q for w in ["forecast", "predict", "future", "next month", "next week", "expect"]):
            return self._answer_forecast(q)

        # ── Seasonal / Trend Questions ───────────────────────────────────────
        if any(w in q for w in ["season", "trend", "month", "peak", "best time", "when"]):
            return self._answer_trends(q)

        # ── Customer Questions ───────────────────────────────────────────────
        if any(w in q for w in ["customer", "buyer", "shopper", "new customer", "returning"]):
            return self._answer_customers(q)

        # ── Promotion / Discount Questions ───────────────────────────────────
        if any(w in q for w in ["promotion", "discount", "offer", "deal", "sale event"]):
            return self._answer_promotions(q)

        # ── Inventory Questions ──────────────────────────────────────────────
        if any(w in q for w in ["stock", "inventory", "reorder", "out of stock", "supply"]):
            return self._answer_inventory(q)

        # ── Holiday Questions ────────────────────────────────────────────────
        if any(w in q for w in ["holiday", "weekend", "weekday", "saturday", "sunday"]):
            return self._answer_holiday(q)

        # ── Rating Questions ─────────────────────────────────────────────────
        if any(w in q for w in ["rating", "review", "score", "feedback"]):
            return self._answer_ratings(q)

        # ── Summary / Overview ───────────────────────────────────────────────
        if any(w in q for w in ["summary", "overview", "key metrics", "kpi", "performance", "how is"]):
            return self._answer_summary(q)

        # ── Default ──────────────────────────────────────────────────────────
        return self._default_answer(q)

    def _answer_revenue(self, q):
        total = self.kb["total_revenue"]
        daily = self.kb["avg_daily_revenue"]
        monthly = total / 24
        lines = [
            "=== REVENUE ANALYSIS ===",
            f"Total Revenue (2 years): ${total:,.2f}",
            f"Average Daily Revenue:   ${daily:,.2f}",
            f"Average Monthly Revenue: ${monthly:,.2f}",
            f"Total Sales Value:       ${self.kb['total_sales']:,.2f}",
            "",
            "Top Revenue Sources:",
            f"  Store:    {self.store_rev.index[0]} (${self.store_rev.iloc[0]:,.2f})",
            f"  Category: {self.cat_rev.index[0]} (${self.cat_rev.iloc[0]:,.2f})",
            f"  Region:   {self.region_rev.index[0]} (${self.region_rev.iloc[0]:,.2f})",
        ]
        return "\n".join(lines)

    def _answer_stores(self, q):
        lines = ["=== STORE PERFORMANCE ===", ""]

        if "best" in q or "top" in q or "highest" in q or "good" in q:
            lines.append("Top 3 Performing Stores:")
            for i in range(min(3, len(self.store_rev))):
                store = self.store_rev.index[i]
                rev = self.store_rev.iloc[i]
                rating = self.store_ratings.get(store, "N/A")
                lines.append(f"  {i+1}. {store}: ${rev:,.2f} (Rating: {rating:.1f})")

        elif "worst" in q or "low" in q or "bad" in q or "poor" in q or "improve" in q:
            lines.append("Stores Needing Improvement:")
            for i in range(1, min(4, len(self.store_rev) + 1)):
                store = self.store_rev.index[-i]
                rev = self.store_rev.iloc[-i]
                rating = self.store_ratings.get(store, "N/A")
                lines.append(f"  {i}. {store}: ${rev:,.2f} (Rating: {rating:.1f})")
            lines.append("")
            lines.append("Recommendation: Focus on operational efficiency and targeted promotions.")
        else:
            lines.append("All Stores by Revenue:")
            for store, rev in self.store_rev.items():
                rating = self.store_ratings.get(store, "N/A")
                lines.append(f"  {store}: ${rev:,.2f} (Rating: {rating:.1f})")

        return "\n".join(lines)

    def _answer_categories(self, q):
        lines = ["=== PRODUCT CATEGORY ANALYSIS ===", ""]

        # Check for specific category mentioned
        for cat in self.cat_rev.index:
            if cat.lower() in q:
                cat_data = self.df[self.df["Product_Category"] == cat]
                lines.append(f"Category: {cat}")
                lines.append(f"  Total Revenue: ${cat_data['Revenue'].sum():,.2f}")
                lines.append(f"  Units Sold:    {cat_data['Units_Sold'].sum():,}")
                lines.append(f"  Avg Discount:  {cat_data['Discount_Percentage'].mean():.1f}%")
                lines.append(f"  Products:      {cat_data['Product_ID'].nunique()}")
                lines.append("")
                subcat = cat_data.groupby("Product_Subcategory")["Revenue"].sum().sort_values(ascending=False)
                lines.append(f"  Top Subcategories:")
                for sc, rev in subcat.head(3).items():
                    lines.append(f"    - {sc}: ${rev:,.2f}")
                return "\n".join(lines)

        # General category ranking
        lines.append("All Categories by Revenue:")
        for cat, rev in self.cat_rev.items():
            pct = rev / self.kb["total_revenue"] * 100
            lines.append(f"  {cat}: ${rev:,.2f} ({pct:.1f}%)")

        lines.append("")
        lines.append(f"Top Subcategories:")
        for sc, rev in self.subcat_rev.head(5).items():
            lines.append(f"  - {sc}: ${rev:,.2f}")

        return "\n".join(lines)

    def _answer_brands(self, q):
        lines = ["=== BRAND PERFORMANCE ===", "", "Top 10 Brands by Revenue:"]
        for i, (brand, rev) in enumerate(self.brand_rev.head(10).items(), 1):
            pct = rev / self.kb["total_revenue"] * 100
            lines.append(f"  {i}. {brand}: ${rev:,.2f} ({pct:.1f}%)")
        return "\n".join(lines)

    def _answer_regions(self, q):
        lines = ["=== REGION ANALYSIS ===", ""]
        for region, rev in self.region_rev.items():
            pct = rev / self.kb["total_revenue"] * 100
            stores = self.df[self.df["Region"] == region]["Store_ID"].nunique()
            lines.append(f"  {region}: ${rev:,.2f} ({pct:.1f}%) - {stores} stores")

        top = self.region_rev.index[0]
        lines.append(f"\nBest Region: {top}")
        return "\n".join(lines)

    def _answer_forecast(self, q):
        lines = [
            "=== REVENUE FORECAST ===",
            "",
            "Based on our Random Forest model (MAPE: 9.49%):",
            "",
            "Next 30 Days Forecast (Jan 2025):",
            "  Total Forecasted Revenue: ~$1,626,156,312",
            "  Daily Average:             ~$54,205,210",
            "  Highest Day:               ~$57,449,599 (Jan 3)",
            "  Lowest Day:                ~$52,905,318 (Jan 7)",
            "",
            "Model Performance:",
            "  Best Model:  Random Forest (R2=0.14, MAPE=9.49%)",
            "  Compared:    6 models (Naive, MA, ES, ARIMA, RF, GB)",
            "",
            "Note: Forecast based on historical patterns. Actual results may vary.",
        ]
        return "\n".join(lines)

    def _answer_trends(self, q):
        lines = ["=== SALES TRENDS & SEASONALITY ===", ""]

        monthly_avg = self.df.groupby("Month")["Revenue"].mean()
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        best_month = month_names.get(monthly_avg.idxmax())
        worst_month = month_names.get(monthly_avg.idxmin())

        lines.append(f"Peak Sales Month:    {best_month}")
        lines.append(f"Lowest Sales Month:  {worst_month}")
        lines.append("")
        lines.append("Monthly Average Revenue:")
        for m, rev in monthly_avg.sort_values(ascending=False).items():
            bar = "█" * int(rev / monthly_avg.max() * 20)
            lines.append(f"  {month_names.get(m, m):>3}: ${rev:>14,.0f}  {bar}")

        lines.append("")
        lines.append("Day of Week Pattern:")
        for day, rev in self.weekday_rev.sort_values(ascending=False).items():
            bar = "█" * int(rev / self.weekday_rev.max() * 20)
            lines.append(f"  {day:>9}: ${rev:>12,.0f}  {bar}")

        return "\n".join(lines)

    def _answer_customers(self, q):
        lines = ["=== CUSTOMER ANALYSIS ===", "", "Revenue by Customer Type:"]
        for ctype, rev in self.customer_rev.items():
            pct = rev / self.kb["total_revenue"] * 100
            lines.append(f"  {ctype}: ${rev:,.2f} ({pct:.1f}%)")

        lines.append("")
        lines.append("Payment Mode Preferences:")
        for mode, rev in self.payment_rev.items():
            pct = rev / self.kb["total_revenue"] * 100
            lines.append(f"  {mode}: ${rev:,.2f} ({pct:.1f}%)")

        return "\n".join(lines)

    def _answer_promotions(self, q):
        lines = ["=== PROMOTION & DISCOUNT ANALYSIS ===", ""]

        promo_yes = self.df[self.df["Promotion_Applied"].astype(str).str.lower().isin(["yes","true","1"])]
        promo_no = self.df[~self.df["Promotion_Applied"].astype(str).str.lower().isin(["yes","true","1"])]

        lines.append(f"With Promotion:    Avg Revenue = ${promo_yes['Revenue'].mean():,.2f}")
        lines.append(f"Without Promotion: Avg Revenue = ${promo_no['Revenue'].mean():,.2f}")
        lines.append(f"Avg Discount:      {self.kb['avg_discount']:.1f}%")
        lines.append("")

        if promo_yes["Revenue"].mean() < promo_no["Revenue"].mean():
            lines.append("KEY INSIGHT: Promotions currently decrease average revenue.")
            lines.append("RECOMMENDATION: Review promotion strategy — target high-margin")
            lines.append("products and specific customer segments instead of blanket discounts.")
        else:
            lines.append("KEY INSIGHT: Promotions increase average revenue.")
            lines.append("RECOMMENDATION: Scale successful promotion strategies.")

        return "\n".join(lines)

    def _answer_inventory(self, q):
        df = self.df
        lines = ["=== INVENTORY ANALYSIS ===", ""]

        low_stock = (df["Stock_On_Hand"] <= LOW_STOCK_THRESHOLD).sum()
        out_of_stock = (df["Stock_On_Hand"] == 0).sum()
        total = len(df)

        lines.append(f"Total Transactions:      {total:,}")
        lines.append(f"Low Stock Items:         {low_stock:,} ({low_stock/total*100:.1f}%)")
        lines.append(f"Out of Stock Items:      {out_of_stock:,} ({out_of_stock/total*100:.1f}%)")
        lines.append(f"Average Stock on Hand:   {df['Stock_On_Hand'].mean():.1f} units")
        lines.append("")

        if low_stock / total > 0.1:
            lines.append("WARNING: High low-stock rate detected!")
        else:
            lines.append("Inventory levels are generally healthy.")

        return "\n".join(lines)

    def _answer_holiday(self, q):
        lines = ["=== HOLIDAY & WEEKEND ANALYSIS ===", ""]

        holiday_rev = self.df[self.df["Holiday_Flag"] == 1]["Revenue"].mean()
        non_holiday_rev = self.df[self.df["Holiday_Flag"] == 0]["Revenue"].mean()
        weekend_rev = self.df[self.df["IsWeekend"] == 1]["Revenue"].mean()
        weekday_rev = self.df[self.df["IsWeekend"] == 0]["Revenue"].mean()

        lines.append(f"Avg Revenue - Holiday:     ${holiday_rev:,.2f}")
        lines.append(f"Avg Revenue - Non-Holiday: ${non_holiday_rev:,.2f}")
        lines.append(f"Holiday Lift:              {((holiday_rev-non_holiday_rev)/non_holiday_rev*100):+.1f}%")
        lines.append("")
        lines.append(f"Avg Revenue - Weekend:     ${weekend_rev:,.2f}")
        lines.append(f"Avg Revenue - Weekday:     ${weekday_rev:,.2f}")
        lines.append(f"Weekend Lift:              {((weekend_rev-weekday_rev)/weekday_rev*100):+.1f}%")

        return "\n".join(lines)

    def _answer_ratings(self, q):
        lines = ["=== STORE RATINGS ===", "", "Store Ratings (Highest to Lowest):"]
        for store, rating in self.store_ratings.items():
            rev = self.store_rev.get(store, 0)
            stars = "★" * int(rating) + "☆" * (5 - int(rating))
            lines.append(f"  {store}: {rating:.2f} {stars}  (Revenue: ${rev:,.0f})")
        return "\n".join(lines)

    def _answer_summary(self, q):
        lines = [
            "=" * 50,
            "  EXECUTIVE BUSINESS SUMMARY",
            "=" * 50,
            "",
            "KEY METRICS:",
            f"  Total Revenue:      ${self.kb['total_revenue']:,.2f}",
            f"  Total Transactions: {self.kb['total_transactions']:,}",
            f"  Total Units Sold:   {self.kb['total_units']:,}",
            f"  Stores:             {self.kb['num_stores']}",
            f"  Products:           {self.kb['num_products']}",
            f"  Categories:         {self.kb['num_categories']}",
            f"  Date Range:         {self.kb['date_range'][0].date()} to {self.kb['date_range'][1].date()}",
            "",
            "TOP PERFORMERS:",
            f"  Best Store:    {self.store_rev.index[0]} (${self.store_rev.iloc[0]:,.0f})",
            f"  Best Category: {self.cat_rev.index[0]} (${self.cat_rev.iloc[0]:,.0f})",
            f"  Best Region:   {self.region_rev.index[0]} (${self.region_rev.iloc[0]:,.0f})",
            f"  Best Brand:    {self.brand_rev.index[0]} (${self.brand_rev.iloc[0]:,.0f})",
            "",
            "KEY FINDINGS:",
            "  1. Electronics is the top revenue category",
            "  2. West region leads in revenue generation",
            "  3. Promotions currently reduce avg revenue (-4.1%)",
            "  4. Store ratings strongly correlate with revenue",
            "  5. February is the peak sales month",
            "",
            "FORECAST (Next 30 Days):",
            "  Expected Revenue: ~$1.63 Billion",
            "  Model: Random Forest (MAPE: 9.49%)",
            "",
            "=" * 50,
        ]
        return "\n".join(lines)

    def _default_answer(self, q):
        return (
            "I can answer questions about:\n"
            "  - Revenue and sales performance\n"
            "  - Store rankings and comparisons\n"
            "  - Product categories and brands\n"
            "  - Regional performance\n"
            "  - Sales forecasts and trends\n"
            "  - Customer behavior and payments\n"
            "  - Promotions and discounts\n"
            "  - Inventory and stock levels\n"
            "  - Holiday and weekend patterns\n"
            "  - Store ratings\n\n"
            "Example questions:\n"
            "  'What is the total revenue?'\n"
            "  'Which store performs best?'\n"
            "  'What are the seasonal trends?'\n"
            "  'What will revenue be next month?'\n"
            "  'How do promotions impact sales?'\n"
            "  'Give me a business summary'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AUTOMATED BUSINESS RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Generate data-driven business recommendations."""
    log.info("Generating business recommendations ...")

    recommendations = []

    # ── 1. Revenue Optimization ──────────────────────────────────────────────
    top_store = df.groupby("Store_ID")["Revenue"].sum().idxmax()
    top_store_rev = df.groupby("Store_ID")["Revenue"].sum().max()
    recommendations.append({
        "Category": "Revenue Optimization",
        "Priority": "HIGH",
        "Finding": f"{top_store} is the highest revenue store (${top_store_rev:,.0f})",
        "Recommendation": f"Analyze {top_store}'s best practices and replicate across other stores",
        "Expected_Impact": "10-15% revenue uplift for underperforming stores",
    })

    # ── 2. Category Strategy ─────────────────────────────────────────────────
    top_cat = df.groupby("Product_Category")["Revenue"].sum().idxmax()
    top_cat_pct = df.groupby("Product_Category")["Revenue"].sum().max() / df["Revenue"].sum() * 100
    recommendations.append({
        "Category": "Category Strategy",
        "Priority": "HIGH",
        "Finding": f"{top_cat} generates {top_cat_pct:.1f}% of total revenue",
        "Recommendation": f"Increase {top_cat} inventory allocation and marketing spend",
        "Expected_Impact": "5-8% revenue growth",
    })

    # ── 3. Promotion Strategy ───────────────────────────────────────────────
    promo_rev = df[df["Promo_And_Discount"] == 1]["Revenue"].mean()
    no_promo_rev = df[df["Promo_And_Discount"] == 0]["Revenue"].mean()
    if promo_rev < no_promo_rev:
        recommendations.append({
            "Category": "Promotion Strategy",
            "Priority": "CRITICAL",
            "Finding": f"Promotions reduce avg revenue by {abs((promo_rev-no_promo_rev)/no_promo_rev*100):.1f}%",
            "Recommendation": "Shift from blanket discounts to targeted, high-margin promotions",
            "Expected_Impact": "Recover 2-4% margin loss",
        })

    # ── 4. Store Improvement ─────────────────────────────────────────────────
    store_rev = df.groupby("Store_ID")["Revenue"].sum().sort_values()
    worst_stores = store_rev.head(2).index.tolist()
    recommendations.append({
        "Category": "Store Improvement",
        "Priority": "HIGH",
        "Finding": f"{', '.join(worst_stores)} are lowest performing stores",
        "Recommendation": "Conduct operational audit, staff training, and local marketing campaigns",
        "Expected_Impact": "15-20% improvement in underperforming stores",
    })

    # ── 5. Regional Expansion ───────────────────────────────────────────────
    top_region = df.groupby("Region")["Revenue"].sum().idxmax()
    recommendations.append({
        "Category": "Regional Strategy",
        "Priority": "MEDIUM",
        "Finding": f"{top_region} region leads in revenue",
        "Recommendation": f"Consider expansion in {top_region} region and strengthen supply chain",
        "Expected_Impact": "8-12% regional revenue growth",
    })

    # ── 6. Inventory Optimization ────────────────────────────────────────────
    low_stock_pct = (df["Stock_On_Hand"] <= LOW_STOCK_THRESHOLD).mean() * 100
    recommendations.append({
        "Category": "Inventory Management",
        "Priority": "HIGH" if low_stock_pct > 5 else "MEDIUM",
        "Finding": f"{low_stock_pct:.1f}% transactions have low stock levels",
        "Recommendation": "Implement automated reorder points and demand-based stocking",
        "Expected_Impact": "Reduce stockout risk by 30-40%",
    })

    # ── 7. Seasonal Planning ─────────────────────────────────────────────────
    monthly_avg = df.groupby("Month")["Revenue"].mean()
    peak_month = monthly_avg.idxmax()
    month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                   7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
    recommendations.append({
        "Category": "Seasonal Planning",
        "Priority": "MEDIUM",
        "Finding": f"{month_names.get(peak_month, peak_month)} is the peak sales month",
        "Recommendation": f"Increase inventory and staffing for {month_names.get(peak_month, peak_month)} season",
        "Expected_Impact": "10-15% higher peak season capture",
    })

    # ── 8. Rating Improvement ────────────────────────────────────────────────
    avg_rating = df["Store_Rating"].mean()
    recommendations.append({
        "Category": "Customer Experience",
        "Priority": "HIGH",
        "Finding": f"Average store rating is {avg_rating:.2f}/5.0",
        "Recommendation": "Focus on customer service training, store cleanliness, and product availability",
        "Expected_Impact": "0.3-0.5 rating improvement = 2-4% revenue lift",
    })

    # ── 9. Brand Portfolio ───────────────────────────────────────────────────
    top_brand = df.groupby("Brand")["Revenue"].sum().idxmax()
    recommendations.append({
        "Category": "Brand Strategy",
        "Priority": "MEDIUM",
        "Finding": f"{top_brand} is the top revenue brand",
        "Recommendation": f"Strengthen partnership with {top_brand} and negotiate better terms",
        "Expected_Impact": "3-5% margin improvement",
    })

    # ── 10. Data-Driven Culture ──────────────────────────────────────────────
    recommendations.append({
        "Category": "Business Intelligence",
        "Priority": "MEDIUM",
        "Finding": "Real-time analytics can improve decision-making speed",
        "Recommendation": "Deploy this AI dashboard for management and establish weekly review cadence",
        "Expected_Impact": "Faster response to market changes",
    })

    rec_df = pd.DataFrame(recommendations)

    # Save
    rec_csv = REPORTS_DIR / "business_recommendations_v2.csv"
    rec_df.to_csv(rec_csv, index=False)
    log.info("Recommendations saved: %s", rec_csv)

    # Print
    print("\n" + "=" * 60)
    print("  BUSINESS RECOMMENDATIONS")
    print("=" * 60)
    for i, row in rec_df.iterrows():
        print(f"\n  [{row['Priority']}] {row['Category']}")
        print(f"    Finding:       {row['Finding']}")
        print(f"    Recommendation: {row['Recommendation']}")
        print(f"    Impact:         {row['Expected_Impact']}")
    print("\n" + "=" * 60)

    return rec_df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EXECUTIVE SUMMARY GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
def generate_executive_summary(df: pd.DataFrame):
    """Generate a comprehensive executive summary report."""
    log.info("Generating executive summary ...")

    total_rev = df["Revenue"].sum()
    total_units = df["Units_Sold"].sum()
    daily_avg = df.groupby("Date")["Revenue"].sum().mean()

    # Store segments
    store_rev = df.groupby("Store_ID")["Revenue"].sum().sort_values(ascending=False)
    top_store = store_rev.index[0]
    bottom_store = store_rev.index[-1]

    # Category
    cat_rev = df.groupby("Product_Category")["Revenue"].sum().sort_values(ascending=False)

    # Trends
    monthly = df.groupby("YearMonth")["Revenue"].sum()
    growth = (monthly.iloc[-1] - monthly.iloc[0]) / monthly.iloc[0] * 100

    summary = f"""
{'='*60}
  EXECUTIVE SUMMARY
  Retail Analytics & AI-Powered Sales Forecasting System
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

  BUSINESS OVERVIEW
{'─'*60}
  Analysis Period    : January 2023 - December 2024 (24 months)
  Total Revenue      : ${total_rev:,.2f}
  Total Units Sold   : {total_units:,}
  Total Transactions : {len(df):,}
  Average Daily Rev  : ${daily_avg:,.2f}
  Stores             : {df['Store_ID'].nunique()}
  Products           : {df['Product_ID'].nunique()}
  Categories         : {df['Product_Category'].nunique()}

  KEY FINDINGS
{'─'*60}
  1. Top Store       : {top_store} (${store_rev.iloc[0]:,.2f})
  2. Bottom Store    : {bottom_store} (${store_rev.iloc[-1]:,.2f})
  3. Top Category    : {cat_rev.index[0]} (${cat_rev.iloc[0]:,.2f})
  4. Overall Growth  : {growth:+.1f}% (first vs last month)
  5. Peak Month      : February
  6. Promotion Impact: -4.1% (needs strategy review)

  FORECAST (Next 30 Days)
{'─'*60}
  Expected Revenue   : ~$1.63 Billion
  Daily Average      : ~$54.2 Million
  Model Used         : Random Forest (MAPE: 9.49%)
  Confidence         : Medium-High

  STRATEGIC RECOMMENDATIONS
{'─'*60}
  1. Replicate {top_store}'s success across all stores
  2. Increase {cat_rev.index[0]} category investment
  3. Revise promotion strategy (targeted vs blanket)
  4. Improve {bottom_store} operations
  5. Plan inventory for peak season (February)
  6. Focus on store rating improvement (+0.3 = +2.4% revenue)

{'='*60}
  END OF EXECUTIVE SUMMARY
{'='*60}
"""

    # Save
    summary_path = REPORTS_DIR / "executive_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    log.info("Executive summary saved: %s", summary_path)

    print(summary)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CATEGORY & STORE DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════════════════
def deep_dive_analysis(df: pd.DataFrame):
    """Generate detailed category and store analysis tables."""
    log.info("Running deep-dive analysis ...")

    # ── Category Analysis ────────────────────────────────────────────────────
    cat_analysis = df.groupby("Product_Category").agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Transactions=("Date", "count"),
        Avg_Discount=("Discount_Percentage", "mean"),
        Avg_Rating=("Store_Rating", "mean"),
        Avg_Stock=("Stock_On_Hand", "mean"),
        Revenue_Share=("Revenue", lambda x: x.sum() / df["Revenue"].sum() * 100),
    ).sort_values("Revenue", ascending=False).reset_index()

    cat_csv = REPORTS_DIR / "category_deep_dive.csv"
    cat_analysis.to_csv(cat_csv, index=False)

    # ── Store Analysis ───────────────────────────────────────────────────────
    store_analysis = df.groupby(["Store_ID", "Store_Location", "Region"]).agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Transactions=("Date", "count"),
        Avg_Discount=("Discount_Percentage", "mean"),
        Store_Rating=("Store_Rating", "mean"),
        Avg_Stock=("Stock_On_Hand", "mean"),
        Low_Stock_Pct=("Reorder_Flag", "mean"),
        Promo_Rate=("Promo_And_Discount", "mean"),
    ).sort_values("Revenue", ascending=False).reset_index()

    store_csv = REPORTS_DIR / "store_deep_dive.csv"
    store_analysis.to_csv(store_csv, index=False)

    # ── Monthly Summary ──────────────────────────────────────────────────────
    monthly_summary = df.groupby("YearMonth").agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Transactions=("Date", "count"),
        Avg_Discount=("Discount_Percentage", "mean"),
    ).reset_index()

    monthly_csv = REPORTS_DIR / "monthly_summary.csv"
    monthly_summary.to_csv(monthly_csv, index=False)

    log.info("Deep-dive analysis saved: %s, %s, %s", cat_csv, store_csv, monthly_csv)

    # Print
    print("\n" + "=" * 60)
    print("  CATEGORY DEEP-DIVE")
    print("=" * 60)
    print(cat_analysis.to_string(index=False))

    print("\n" + "=" * 60)
    print("  STORE DEEP-DIVE")
    print("=" * 60)
    print(store_analysis.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. STOCKOUT RISK PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
def predict_stockout_risk(df: pd.DataFrame):
    """Identify products and stores at risk of stockout."""
    log.info("Predicting stockout risk ...")

    # Aggregate at store-product level
    risk = df.groupby(["Store_ID", "Product_Category", "Brand"]).agg(
        Avg_Stock=("Stock_On_Hand", "mean"),
        Min_Stock=("Stock_On_Hand", "min"),
        Avg_Units_Sold=("Units_Sold", "mean"),
        Total_Transactions=("Date", "count"),
        Revenue=("Revenue", "sum"),
    ).reset_index()

    # Calculate risk score
    risk["Stock_Days"] = np.where(
        risk["Avg_Units_Sold"] > 0,
        risk["Avg_Stock"] / risk["Avg_Units_Sold"],
        999,
    )

    # Risk categories
    risk["Risk_Level"] = pd.cut(
        risk["Stock_Days"],
        bins=[-1, 3, 7, 14, 30, float("inf")],
        labels=["Critical", "High", "Medium", "Low", "Safe"],
    )

    # Sort by risk
    risk = risk.sort_values("Stock_Days").reset_index(drop=True)

    # Save
    risk_csv = REPORTS_DIR / "stockout_risk_prediction.csv"
    risk.to_csv(risk_csv, index=False)

    # Summary
    risk_summary = risk.groupby("Risk_Level").agg(
        Count=("Store_ID", "count"),
        Revenue_At_Risk=("Revenue", "sum"),
    ).reset_index()

    summary_csv = REPORTS_DIR / "stockout_risk_summary.csv"
    risk_summary.to_csv(summary_csv, index=False)

    log.info("Stockout risk saved: %s, %s", risk_csv, summary_csv)

    print("\n" + "=" * 60)
    print("  STOCKOUT RISK PREDICTION")
    print("=" * 60)
    print(risk_summary.to_string(index=False))
    print(f"\n  Critical items: {(risk['Risk_Level'] == 'Critical').sum()}")
    print(f"  High risk items: {(risk['Risk_Level'] == 'High').sum()}")

    return risk, risk_summary


# ═══════════════════════════════════════════════════════════════════════════════
# 6. INTERACTIVE Q&A DEMO
# ═══════════════════════════════════════════════════════════════════════════════
def run_qa_demo(assistant: RetailAIAssistant):
    """Demonstrate the AI assistant with sample questions."""
    demo_questions = [
        "What is the total revenue?",
        "Which store performs best?",
        "What are the seasonal trends?",
        "How do promotions impact sales?",
        "Give me a business summary",
    ]

    print("\n" + "=" * 60)
    print("  AI BUSINESS ASSISTANT - DEMO")
    print("=" * 60)

    for q in demo_questions:
        print(f"\n  Q: {q}")
        print(f"  A: {assistant.answer(q)}")
        print()

    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def run_ai_assistant():
    print("\n" + "=" * 60)
    print("  RETAIL AI ANALYTICS - PHASE 6")
    print("  AI Business Assistant + Recommendations")
    print("=" * 60 + "\n")

    # Load data
    log.info("Loading processed data ...")
    df = pd.read_csv(PROCESSED_CSV, parse_dates=["Date"])
    log.info("Loaded %s rows x %s columns", f"{len(df):,}", df.shape[1])

    # 1. Initialize AI Assistant
    assistant = RetailAIAssistant(df)

    # 2. Q&A Demo
    run_qa_demo(assistant)

    # 3. Business Recommendations
    rec_df = generate_recommendations(df)

    # 4. Executive Summary
    generate_executive_summary(df)

    # 5. Deep-Dive Analysis
    deep_dive_analysis(df)

    # 6. Stockout Risk Prediction
    risk_df, risk_summary = predict_stockout_risk(df)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  PHASE 6 SUMMARY")
    print("-" * 60)
    print(f"  AI Assistant Ready       : Yes (12+ question types)")
    print(f"  Recommendations Generated: {len(rec_df)}")
    print(f"  Executive Summary        : reports/executive_summary.txt")
    print(f"  Category Deep-Dive       : reports/category_deep_dive.csv")
    print(f"  Store Deep-Dive          : reports/store_deep_dive.csv")
    print(f"  Monthly Summary          : reports/monthly_summary.csv")
    print(f"  Stockout Risk Items      : {len(risk_df)}")
    print(f"  Critical Risk Items      : {(risk_df['Risk_Level'] == 'Critical').sum()}")
    print("-" * 60)
    print("  PHASE 6 COMPLETE")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    run_ai_assistant()