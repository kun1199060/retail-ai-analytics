"""
Retail Analytics & AI-Powered Sales Forecasting System
Central Configuration - Paths, Schema, Constants
Author: Kunal | Project: Unlox Capstone
"""

from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory Paths
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Auto-create directories
for _dir in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR, NOTEBOOKS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_CSV = RAW_DIR / "Retail_Sales_Data_Unlox.csv"
PROCESSED_CSV = PROCESSED_DIR / "retail_sales_clean.csv"
VALIDATION_REPORT = REPORTS_DIR / "data_validation_report.txt"

# Expected Schema (19 columns from Unlox dataset)
EXPECTED_COLUMNS = [
    "Date", "Store_ID", "Store_Location", "Product_ID",
    "Product_Category", "Product_Subcategory", "Brand",
    "Unit_Price", "Units_Sold", "Total_Sales",
    "Discount_Percentage", "Revenue",
    "Customer_Type", "Payment_Mode",
    "Promotion_Applied", "Stock_On_Hand",
    "Store_Rating", "Region", "Holiday_Flag",
]

NUMERIC_COLS = [
    "Unit_Price", "Units_Sold", "Total_Sales",
    "Discount_Percentage", "Revenue",
    "Stock_On_Hand", "Store_Rating",
]

CATEGORICAL_COLS = [
    "Store_ID", "Store_Location", "Product_ID",
    "Product_Category", "Product_Subcategory", "Brand",
    "Customer_Type", "Payment_Mode",
    "Promotion_Applied", "Region", "Holiday_Flag",
]

# Business Rules
LOW_STOCK_THRESHOLD = 20
HIGH_PERF_PERCENTILE = 75.0
LOW_PERF_PERCENTILE = 25.0
FORECAST_HORIZON_DAYS = 30