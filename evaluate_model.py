import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("RETAIL AI - MODEL EVALUATION")
print("=" * 60)

# Load validation dataset
data = pd.read_csv(
    "data/processed/validation_data.csv",
    parse_dates=["date"]
)

print("\nValidation Data Shape:", data.shape)

# Load trained model
model = joblib.load("models/random_forest.pkl")

# Use the exact feature order from model training
features = list(model.feature_names_in_)

print("\nModel Features:", features)

# Prepare input and actual values
X = data[features]
y_actual = data["sales"]

# Generate predictions
y_pred = model.predict(X)

# Calculate metrics
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 40)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# Save predictions
data["predicted_sales"] = y_pred
data["prediction_error"] = (
    data["sales"] - data["predicted_sales"]
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

data.to_csv(
    "outputs/validation_predictions.csv",
    index=False
)

# Save evaluation metrics
metrics = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2 Score"],
    "Value": [mae, rmse, r2]
})

metrics.to_csv(
    "reports/evaluation_metrics.csv",
    index=False
)

# Actual vs Predicted graph
sample = data.sample(
    n=min(1000, len(data)),
    random_state=42
).sort_values("date")

plt.figure(figsize=(12, 6))

plt.plot(
    sample["date"],
    sample["sales"],
    label="Actual Sales"
)

plt.plot(
    sample["date"],
    sample["predicted_sales"],
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

# Prediction error distribution
plt.figure(figsize=(10, 6))

plt.hist(data["prediction_error"], bins=50)

plt.title("Prediction Error Distribution")
plt.xlabel("Actual Sales - Predicted Sales")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("outputs/prediction_error_distribution.png")
plt.close()

print("\nFILES SAVED SUCCESSFULLY:")
print("1. outputs/validation_predictions.csv")
print("2. outputs/actual_vs_predicted.png")
print("3. outputs/prediction_error_distribution.png")
print("4. reports/evaluation_metrics.csv")

print("\nMODEL EVALUATION COMPLETED!")