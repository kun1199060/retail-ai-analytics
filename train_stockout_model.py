import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

print("=" * 60)
print("RETAIL AI - STOCKOUT RISK PREDICTION")
print("=" * 60)

start_time = time.time()

# --------------------------------------------------
# STEP 1: LOAD DATASET
# --------------------------------------------------

print("\nLoading stockout dataset...")

df = pd.read_csv(
    "data/processed/stockout_dataset.csv"
)

print("Dataset Shape:", df.shape)

# --------------------------------------------------
# STEP 2: PREPARE FEATURES
# --------------------------------------------------

# These inputs are available before the simulated
# stockout-risk outcome is calculated.
features = [
    "available_inventory",
    "daily_demand",
    "lead_time",
    "onpromotion",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30"
]

target = "stockout_risk"

X = df[features]
y = df[target]

print("\nFeatures:", features)

print("\nTarget Distribution:")
print(y.value_counts())

# --------------------------------------------------
# STEP 3: TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# --------------------------------------------------
# STEP 4: DEFINE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        random_state=42
    )
}

# --------------------------------------------------
# STEP 5: TRAIN AND EVALUATE
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test, probabilities
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    filename = (
        name.lower().replace(" ", "_")
        + "_stockout.pkl"
    )

    joblib.dump(
        model,
        os.path.join("models", filename)
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# --------------------------------------------------
# STEP 6: MODEL COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n" + "=" * 60)
print("STOCKOUT MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

results_df.to_csv(
    "reports/stockout_model_comparison.csv",
    index=False
)

best_model = results_df.iloc[0]["Model"]

print("\nBest Model:", best_model)

print(
    "\nTotal Time:",
    round(time.time() - start_time, 2),
    "seconds"
)

print("\nSTOCKOUT MODEL TRAINING COMPLETED!")