import pandas as pd
import os

# Dataset folder
DATA_PATH = "data/raw"

# CSV files
files = [
    "train.csv",
    "test.csv",
    "stores.csv",
    "oil.csv",
    "holidays_events.csv",
    "transactions.csv",
    "sample_submission.csv"
]

print("=" * 60)
print("RETAIL AI PROJECT - DATASET INSPECTION")
print("=" * 60)

for file in files:
    path = os.path.join(DATA_PATH, file)

    df = pd.read_csv(path)

    print(f"\nFILE: {file}")
    print("-" * 40)
    print("Rows and Columns:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nMissing Values:")
    print(df.isnull().sum())

print("\nDATASET INSPECTION COMPLETED!")