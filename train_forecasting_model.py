import pandas as pd
import numpy as np
import os
import time

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import joblib

print("=" * 55)
print("RETAIL AI - DEMAND FORECASTING")
print("=" * 55)

start_time = time.time()

# -------------------------------------------------
# 1. Configuration
# -------------------------------------------------

FILE_PATH = "data/processed/retail_features.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

FEATURES = [
    "store_nbr",
    "onpromotion",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30"
]

# -------------------------------------------------
# 2. Load data in chunks
# -------------------------------------------------

print("\nReading dataset in chunks...")

train_parts = []
valid_parts = []

all_families = sorted(
    pd.read_csv(
        FILE_PATH,
        usecols=["family"]
    )["family"].dropna().unique()
)

cutoff_date = pd.Timestamp("2017-07-17")

dtypes = {
    "store_nbr": "int16",
    "onpromotion": "int16",
    "year": "int16",
    "month": "int8",
    "day": "int8",
    "day_of_week": "int8",
    "week_of_year": "int16",
    "is_weekend": "int8",
    "sales": "float32",
    "lag_7": "float32",
    "lag_14": "float32",
    "lag_30": "float32",
    "rolling_mean_7": "float32",
    "rolling_mean_30": "float32"
}

columns = FEATURES + ["family", "date", "sales"]

for chunk in pd.read_csv(
    FILE_PATH,
    usecols=columns,
    dtype=dtypes,
    chunksize=200000
):

    chunk["date"] = pd.to_datetime(chunk["date"])

    chunk["family_code"] = pd.Categorical(
        chunk["family"],
        categories=all_families
    ).codes.astype("int16")

    chunk.drop(columns=["family"], inplace=True)

    # Latest 30 days for validation
    validation = chunk[chunk["date"] >= cutoff_date]

    if not validation.empty:
        valid_parts.append(validation)

    # Historical data for training
    training = chunk[chunk["date"] < cutoff_date]

    if not training.empty:
        sampled = training.sample(
            n=min(25000, len(training)),
            random_state=42
        )

        train_parts.append(sampled)

    print(
        "Processed rows:",
        sum(len(x) for x in train_parts)
    )

train_df = pd.concat(train_parts, ignore_index=True)
valid_df = pd.concat(valid_parts, ignore_index=True)

print("\nTraining records:", len(train_df))
print("Validation records:", len(valid_df))

# -------------------------------------------------
# 3. Prepare X and y
# -------------------------------------------------

FEATURES.append("family_code")

X_train = train_df[FEATURES]
y_train = train_df["sales"]

X_valid = valid_df[FEATURES]
y_valid = valid_df["sales"]

# -------------------------------------------------
# 4. Define models
# -------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.1,
        random_state=42
    )
}

# -------------------------------------------------
# 5. Train and evaluate
# -------------------------------------------------

results = []

print("\nStarting model training...")

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_valid, predictions)
    )

    r2 = r2_score(
        y_valid,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    # Save model
    model_name = name.lower().replace(" ", "_")

    joblib.dump(
        model,
        f"models/{model_name}.pkl"
    )

# -------------------------------------------------
# 6. Compare models
# -------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE"
)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

print(results_df.to_string(index=False))

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

# Save validation data for later analysis
valid_df.to_csv(
    "data/processed/validation_data.csv",
    index=False
)

print("\nBest model:", results_df.iloc[0]["Model"])

print("\nTotal time:", round(time.time() - start_time, 2), "seconds")

print("\nDEMAND FORECASTING TRAINING COMPLETED!")