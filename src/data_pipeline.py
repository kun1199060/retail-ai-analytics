"""
Retail Analytics & AI-Powered Sales Forecasting System
Phase 2: Data Pipeline - Load -> Validate -> Clean -> Feature Engineering -> Save

Run from project root:
    python src/data_pipeline.py

Author: Kunal | Project: Unlox Capstone
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    PROCESSED_CSV, RAW_CSV, VALIDATION_REPORT,
    EXPECTED_COLUMNS, NUMERIC_COLS, CATEGORICAL_COLS,
    LOW_STOCK_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ========================== 1. LOAD DATA ==========================
def load_data(path=RAW_CSV):
    if not path.exists():
        log.error("File not found: %s", path)
        raise FileNotFoundError(f"Raw data file not found: {path}")
    log.info("Loading data from: %s", path.name)
    df = pd.read_csv(path)
    log.info("Loaded %s rows x %s columns", f"{len(df):,}", df.shape[1])
    return df


# ========================== 2. VALIDATE DATA ==========================
def validate_data(df):
    log.info("Running data quality validation ...")
    report = {"checks_passed": [], "warnings": [], "errors": []}

    # Schema check
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing_cols:
        report["errors"].append(f"Missing columns: {missing_cols}")
    else:
        report["checks_passed"].append("All expected columns present")
    if extra_cols:
        report["warnings"].append(f"Extra columns: {extra_cols}")

    # Missing values
    total_nulls = df.isnull().sum().sum()
    if total_nulls == 0:
        report["checks_passed"].append("No missing values")
    else:
        report["warnings"].append(f"Missing values: {total_nulls}")

    # Duplicates
    dupes = df.duplicated().sum()
    if dupes == 0:
        report["checks_passed"].append("No duplicate rows")
    else:
        report["warnings"].append(f"Duplicate rows: {dupes}")

    # Date range
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        bad_dates = dates.isna().sum()
        if bad_dates > 0:
            report["errors"].append(f"Unparseable dates: {bad_dates}")
        else:
            report["checks_passed"].append(
                f"Date range: {dates.min().date()} to {dates.max().date()}"
            )

    # Numeric sanity
    for col in NUMERIC_COLS:
        if col in df.columns:
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                report["warnings"].append(f"'{col}' has {negatives} negative values")

    log.info("Checks passed: %d | Warnings: %d | Errors: %d",
             len(report["checks_passed"]), len(report["warnings"]), len(report["errors"]))
    return report


def save_validation_report(report, df):
    lines = [
        "=" * 60,
        "DATA VALIDATION REPORT",
        "Retail Analytics & AI-Powered Sales Forecasting System",
        "=" * 60,
        "",
        f"Dataset shape : {df.shape[0]:,} rows x {df.shape[1]} columns",
        f"Memory usage  : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        "",
        "-" * 60, "PASSED CHECKS", "-" * 60,
    ]
    for item in report["checks_passed"]:
        lines.append(f"  [PASS] {item}")
    lines += ["", "-" * 60, "WARNINGS", "-" * 60]
    if report["warnings"]:
        for item in report["warnings"]:
            lines.append(f"  [WARN] {item}")
    else:
        lines.append("  (none)")
    lines += ["", "-" * 60, "ERRORS", "-" * 60]
    if report["errors"]:
        for item in report["errors"]:
            lines.append(f"  [FAIL] {item}")
    else:
        lines.append("  (none)")
    lines += ["", "=" * 60, "END OF REPORT", "=" * 60]
    VALIDATION_REPORT.write_text("\n".join(lines), encoding="utf-8")
    log.info("Validation report saved: %s", VALIDATION_REPORT)


# ========================== 3. CLEAN DATA ==========================
def clean_data(df):
    log.info("Cleaning data ...")
    df = df.copy()
    initial_rows = len(df)

    # Date parsing
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    bad_dates = df["Date"].isna().sum()
    if bad_dates > 0:
        df = df.dropna(subset=["Date"])
        log.warning("Dropped %d rows with unparseable dates", bad_dates)

    # String standardization
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Numeric dtype enforcement
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        log.info("Removed %d duplicate rows", dupes)

    # Sort by Date
    df = df.sort_values("Date").reset_index(drop=True)
    log.info("Cleaning complete: %s -> %s rows (%d removed)",
             f"{initial_rows:,}", f"{len(df):,}", initial_rows - len(df))
    return df


# ========================== 4. FEATURE ENGINEERING ==========================
def engineer_features(df):
    log.info("Engineering features ...")
    df = df.copy()

    # A. Calendar / Time Features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayName"] = df["Date"].dt.day_name()
    df["Quarter"] = df["Date"].dt.quarter
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

    # B. Price & Discount Features
    df["Effective_Price"] = df["Unit_Price"] * (1 - df["Discount_Percentage"] / 100)
    df["Discount_Bucket"] = pd.cut(
        df["Discount_Percentage"],
        bins=[-0.01, 0, 10, 20, 30, 50, 100],
        labels=["No Discount", "Low (1-10%)", "Moderate (11-20%)",
                "High (21-30%)", "Very High (31-50%)", "Extreme (50%+)"],
    )

    # C. Revenue & Sales Features
    df["Expected_Sales"] = df["Effective_Price"] * df["Units_Sold"]
    df["Sales_Calc_Diff"] = df["Total_Sales"] - df["Expected_Sales"]
    df["Discount_Amount"] = df["Unit_Price"] * df["Units_Sold"] - df["Total_Sales"]

    # D. Inventory / Stock Features
    df["Stock_Status"] = pd.cut(
        df["Stock_On_Hand"],
        bins=[-1, 0, LOW_STOCK_THRESHOLD, 100, float("inf")],
        labels=["Out of Stock", "Low Stock", "Healthy", "Overstock"],
    )
    df["Days_Of_Stock"] = np.where(
        df["Units_Sold"] > 0,
        df["Stock_On_Hand"] / df["Units_Sold"],
        np.inf,
    )
    df["Reorder_Flag"] = (df["Stock_On_Hand"] <= LOW_STOCK_THRESHOLD).astype(int)

    # E. Customer & Promotion Features
    df["Promo_And_Discount"] = (
        (df["Promotion_Applied"].astype(str).str.lower().isin(["yes", "true", "1"]))
        & (df["Discount_Percentage"] > 0)
    ).astype(int)

    log.info("Feature engineering complete: %d total columns", df.shape[1])
    return df


# ========================== 5. FEATURE SUMMARY ==========================
def generate_feature_summary(df):
    log.info("Generating feature summary ...")
    summary = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.values,
        "Non_Null": df.notna().sum().values,
        "Null_Count": df.isna().sum().values,
        "Null_%": (df.isna().sum() / len(df) * 100).round(2).values,
        "Unique_Values": [df[col].nunique() for col in df.columns],
        "Sample_Value": [
            str(df[col].dropna().iloc[0]) if df[col].notna().any() else "N/A"
            for col in df.columns
        ],
    })
    return summary


# ========================== 6. MAIN PIPELINE ==========================
def run_pipeline():
    print("\n" + "=" * 60)
    print("  RETAIL AI ANALYTICS - DATA PIPELINE")
    print("  Phase 2: Load -> Validate -> Clean -> Engineer -> Save")
    print("=" * 60 + "\n")

    # Step 1: Load
    df_raw = load_data()

    # Step 2: Validate
    validation = validate_data(df_raw)
    save_validation_report(validation, df_raw)

    # Step 3: Clean
    df_clean = clean_data(df_raw)

    # Step 4: Feature Engineering
    df_featured = engineer_features(df_clean)

    # Step 5: Feature Summary
    summary = generate_feature_summary(df_featured)
    summary_path = PROCESSED_CSV.parent / "feature_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Feature summary saved: %s", summary_path)

    # Step 6: Save processed data
    df_featured.to_csv(PROCESSED_CSV, index=False)
    log.info("Processed data saved: %s", PROCESSED_CSV)

    # Final Summary
    print("\n" + "-" * 60)
    print("  PIPELINE SUMMARY")
    print("-" * 60)
    print(f"  Raw rows        : {len(df_raw):,}")
    print(f"  Clean rows      : {len(df_featured):,}")
    print(f"  Total columns   : {df_featured.shape[1]}")
    print(f"  Date range      : {df_featured['Date'].min().date()} -> {df_featured['Date'].max().date()}")
    print(f"  Unique stores   : {df_featured['Store_ID'].nunique()}")
    print(f"  Unique products : {df_featured['Product_ID'].nunique()}")
    print(f"  Categories      : {df_featured['Product_Category'].nunique()}")
    print(f"  Brands          : {df_featured['Brand'].nunique()}")
    print(f"  Regions         : {df_featured['Region'].nunique()}")
    print("-" * 60)
    print(f"  Output file  : {PROCESSED_CSV}")
    print(f"  Report file  : {VALIDATION_REPORT}")
    print(f"  Summary file : {summary_path}")
    print("-" * 60)
    print("  PHASE 2 COMPLETE")
    print("-" * 60 + "\n")

    return df_featured


if __name__ == "__main__":
    df = run_pipeline()