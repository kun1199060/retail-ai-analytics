"""
Retail Analytics & AI-Powered Sales Forecasting System
Phase 4: Time-Series Revenue Forecasting

Models:
  1. Naive Forecast (baseline)
  2. Moving Average
  3. Exponential Smoothing (Holt-Winters)
  4. ARIMA
  5. Random Forest Regressor
  6. Gradient Boosting Regressor

Evaluation: MAE, RMSE, MAPE, R2 Score
Train/Test: Chronological split (no random split for time series!)

Run from project root:
    python src/forecasting.py

Author: Kunal | Project: Unlox Capstone
"""

import logging
import sys
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_CSV, OUTPUTS_DIR, REPORTS_DIR, MODELS_DIR

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Plot Style ────────────────────────────────────────────────────────────────
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

PRIMARY = "#1a1a2e"
ACCENT = "#e94560"
ACCENT2 = "#0f3460"
GREEN = "#27ae60"
ORANGE = "#f39c12"


def save_fig(fig, name):
    fig.tight_layout()
    path = OUTPUTS_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    log.info("Chart saved: %s", path.name)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_model(y_true, y_pred, model_name):
    """Calculate MAE, RMSE, MAPE, R2 for a model."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove any NaN or inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoid division by zero)
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan

    metrics = {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "R2 Score": round(r2, 4),
    }

    log.info("%s -> MAE: %.2f | RMSE: %.2f | MAPE: %.2f%% | R2: %.4f",
             model_name, mae, rmse, mape, r2)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_daily_data(df):
    """Aggregate to daily revenue time series."""
    daily = df.groupby("Date").agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Transactions=("Date", "count"),
    ).reset_index()
    daily = daily.sort_values("Date").reset_index(drop=True)

    # Fill any missing dates
    date_range = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    daily = daily.set_index("Date").reindex(date_range).fillna(0).reset_index()
    daily.columns = ["Date", "Revenue", "Units_Sold", "Transactions"]

    log.info("Daily time series: %d days, from %s to %s",
             len(daily), daily["Date"].min().date(), daily["Date"].max().date())
    return daily


def create_time_features(df):
    """Create time-based features for ML models."""
    df = df.copy()
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Quarter"] = df["Date"].dt.quarter
    df["Year"] = df["Date"].dt.year
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    # Lag features
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"Revenue_Lag{lag}"] = df["Revenue"].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        df[f"Revenue_RollMean{window}"] = df["Revenue"].rolling(window=window, min_periods=1).mean()
        df[f"Revenue_RollStd{window}"] = df["Revenue"].rolling(window=window, min_periods=1).std()

    # Expanding mean
    df["Revenue_ExpandingMean"] = df["Revenue"].expanding().mean()

    # Drop NaN rows created by lags
    df = df.dropna().reset_index(drop=True)
    return df


def chronological_split(df, test_ratio=0.2):
    """Split data chronologically — NO random split for time series!"""
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    log.info("Train: %d days (%s to %s)", len(train),
             train["Date"].min().date(), train["Date"].max().date())
    log.info("Test:  %d days (%s to %s)", len(test),
             test["Date"].min().date(), test["Date"].max().date())
    return train, test


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1: NAIVE FORECAST (Baseline)
# ═══════════════════════════════════════════════════════════════════════════════
def naive_forecast(train, test):
    """Naive: tomorrow = today (last known value)."""
    last_value = train["Revenue"].iloc[-1]
    predictions = np.full(len(test), last_value)
    return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2: MOVING AVERAGE
# ═══════════════════════════════════════════════════════════════════════════════
def moving_average_forecast(train, test, window=7):
    """Moving average of last N days."""
    combined = pd.concat([train["Revenue"], test["Revenue"]]).reset_index(drop=True)
    predictions = []
    for i in range(len(test)):
        start = len(train) + i - window
        end = len(train) + i
        if start < 0:
            start = 0
        predictions.append(combined.iloc[start:end].mean())
    return np.array(predictions)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3: EXPONENTIAL SMOOTHING (Holt-Winters)
# ═══════════════════════════════════════════════════════════════════════════════
def exponential_smoothing_forecast(train, test):
    """Holt-Winters triple exponential smoothing."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        model = ExponentialSmoothing(
            train["Revenue"],
            seasonal_periods=7,
            trend="add",
            seasonal="add",
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)
        predictions = fitted.forecast(steps=len(test))
        return predictions.values
    except Exception as e:
        log.warning("Exponential Smoothing failed (%s), using simple ES", e)
        # Fallback: simple exponential smoothing
        alpha = 0.3
        pred = []
        last = train["Revenue"].iloc[-1]
        for _ in range(len(test)):
            pred.append(last)
            last = alpha * pred[-1] + (1 - alpha) * last
        return np.array(pred)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 4: ARIMA
# ═══════════════════════════════════════════════════════════════════════════════
def arima_forecast(train, test):
    """ARIMA model for time series forecasting."""
    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(train["Revenue"], order=(7, 1, 2))
        fitted = model.fit()
        predictions = fitted.forecast(steps=len(test))
        return predictions.values
    except Exception as e:
        log.warning("ARIMA failed (%s), using simpler order", e)
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train["Revenue"], order=(1, 1, 1))
            fitted = model.fit()
            predictions = fitted.forecast(steps=len(test))
            return predictions.values
        except Exception as e2:
            log.warning("ARIMA fallback also failed (%s), using moving average", e2)
            return moving_average_forecast(train, test, window=7)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 5 & 6: ML MODELS (Random Forest + Gradient Boosting)
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_ml_features(df):
    """Create feature matrix for ML models."""
    features = [
        "DayOfWeek", "Month", "Day", "Quarter", "Year",
        "IsWeekend", "IsMonthStart", "IsMonthEnd", "WeekOfYear",
        "Revenue_Lag1", "Revenue_Lag2", "Revenue_Lag3",
        "Revenue_Lag7", "Revenue_Lag14", "Revenue_Lag30",
        "Revenue_RollMean7", "Revenue_RollMean14", "Revenue_RollMean30",
        "Revenue_RollStd7", "Revenue_RollStd14", "Revenue_RollStd30",
        "Revenue_ExpandingMean",
    ]
    available = [f for f in features if f in df.columns]
    return df[available].values, available


def ml_forecast(train, test, model_type="rf"):
    """Random Forest or Gradient Boosting forecast."""
    X_train, feature_names = prepare_ml_features(train)
    y_train = train["Revenue"].values
    X_test, _ = prepare_ml_features(test)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_split=5, min_samples_leaf=2, random_state=42,
        )

    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    # Feature importance
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return predictions, importance, model


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def plot_model_comparison(test, all_predictions, all_metrics):
    """Plot all model predictions vs actual + metrics comparison."""

    # ── Chart 1: Actual vs All Predictions ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(test["Date"], test["Revenue"], color="black", linewidth=2.5,
            label="Actual Revenue", zorder=5)

    colors = {
        "Naive Forecast": "#95a5a6",
        "Moving Average (7d)": "#e67e22",
        "Exponential Smoothing": "#9b59b6",
        "ARIMA": "#e74c3c",
        "Random Forest": "#2980b9",
        "Gradient Boosting": "#27ae60",
    }

    for name, preds in all_predictions.items():
        ax.plot(test["Date"], preds, color=colors.get(name, "gray"),
                linewidth=1.5, alpha=0.8, label=name)

    ax.set_title("Revenue Forecasting — Model Comparison (Test Period)",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.legend(fontsize=10, loc="upper left", ncol=2)
    save_fig(fig, "20_model_comparison_forecast")

    # ── Chart 2: Metrics Comparison Bar Chart ────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # MAE
    colors_bar = [ACCENT if m == metrics_df["MAE"].min() else ACCENT2 for m in metrics_df["MAE"]]
    axes[0].barh(metrics_df["Model"], metrics_df["MAE"], color=colors_bar, edgecolor="white")
    axes[0].set_title("MAE (Lower = Better)", fontsize=14, fontweight="bold")
    axes[0].invert_yaxis()

    # RMSE
    colors_bar = [GREEN if m == metrics_df["RMSE"].min() else ACCENT2 for m in metrics_df["RMSE"]]
    axes[1].barh(metrics_df["Model"], metrics_df["RMSE"], color=colors_bar, edgecolor="white")
    axes[1].set_title("RMSE (Lower = Better)", fontsize=14, fontweight="bold")
    axes[1].invert_yaxis()

    # R2 Score
    colors_bar = [ORANGE if m == metrics_df["R2 Score"].max() else ACCENT2 for m in metrics_df["R2 Score"]]
    axes[2].barh(metrics_df["Model"], metrics_df["R2 Score"], color=colors_bar, edgecolor="white")
    axes[2].set_title("R2 Score (Higher = Better)", fontsize=14, fontweight="bold")
    axes[2].invert_yaxis()

    save_fig(fig, "21_model_metrics_comparison")

    # ── Chart 3: Best Model — Actual vs Predicted with Confidence ────────────
    best_model = metrics_df.loc[metrics_df["R2 Score"].idxmax(), "Model"]
    best_preds = all_predictions[best_model]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Time series
    axes[0].plot(test["Date"], test["Revenue"], color="black", linewidth=2, label="Actual")
    axes[0].plot(test["Date"], best_preds, color=ACCENT, linewidth=2, label=f"Predicted ({best_model})")
    axes[0].fill_between(test["Date"], test["Revenue"], best_preds, alpha=0.15, color=ACCENT)
    axes[0].set_title(f"Best Model: {best_model}", fontsize=16, fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Revenue ($)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    axes[0].legend(fontsize=12)

    # Scatter: Actual vs Predicted
    axes[1].scatter(test["Revenue"], best_preds, alpha=0.5, s=30, color=ACCENT2, edgecolors="white")
    min_val = min(test["Revenue"].min(), best_preds.min())
    max_val = max(test["Revenue"].max(), best_preds.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    axes[1].set_title(f"Actual vs Predicted (R2 = {metrics_df.loc[metrics_df['R2 Score'].idxmax(), 'R2 Score']:.4f})",
                      fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Actual Revenue ($)")
    axes[1].set_ylabel("Predicted Revenue ($)")
    axes[1].legend(fontsize=12)

    save_fig(fig, "22_best_model_analysis")

    return best_model


def plot_feature_importance(importance_df):
    """Plot feature importance from the best ML model."""
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = importance_df.head(15)
    colors = sns.color_palette("viridis", len(top_features))
    ax.barh(top_features["Feature"], top_features["Importance"], color=colors, edgecolor="white")
    ax.set_title("Top 15 Feature Importance (ML Forecasting Model)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    save_fig(fig, "23_feature_importance")


# ═══════════════════════════════════════════════════════════════════════════════
# FUTURE FORECAST (Next 30 Days)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_future_forecast(df, best_model_name, all_predictions, horizon=30):
    """Generate 30-day future forecast using the best model."""
    log.info("Generating %d-day future forecast using %s ...", horizon, best_model_name)

    # Prepare full dataset with features
    featured = create_time_features(df)

    # Train on ALL data
    X_all, feature_names = prepare_ml_features(featured)
    y_all = featured["Revenue"].values

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    model = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    model.fit(X_all_scaled, y_all)

    # Generate future dates
    last_date = df["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Create future features (iterative prediction)
    future_predictions = []
    recent_revenue = list(df["Revenue"].values[-30:])  # Last 30 days for lags

    for future_date in future_dates:
        # Time features
        dow = future_date.dayofweek
        month = future_date.month
        day = future_date.day
        quarter = (month - 1) // 3 + 1
        year = future_date.year
        is_weekend = 1 if dow >= 5 else 0
        is_month_start = 1 if day == 1 else 0
        is_month_end = 1 if future_date.is_month_end else 0
        week_of_year = future_date.isocalendar()[1]

        # Lag features from recent revenue
        rev_series = pd.Series(recent_revenue)
        lag1 = rev_series.iloc[-1] if len(rev_series) >= 1 else 0
        lag2 = rev_series.iloc[-2] if len(rev_series) >= 2 else 0
        lag3 = rev_series.iloc[-3] if len(rev_series) >= 3 else 0
        lag7 = rev_series.iloc[-7] if len(rev_series) >= 7 else 0
        lag14 = rev_series.iloc[-14] if len(rev_series) >= 14 else 0
        lag30 = rev_series.iloc[-30] if len(rev_series) >= 30 else 0

        # Rolling stats
        roll_mean7 = rev_series.tail(7).mean() if len(rev_series) >= 7 else rev_series.mean()
        roll_mean14 = rev_series.tail(14).mean() if len(rev_series) >= 14 else rev_series.mean()
        roll_mean30 = rev_series.tail(30).mean() if len(rev_series) >= 30 else rev_series.mean()
        roll_std7 = rev_series.tail(7).std() if len(rev_series) >= 7 else rev_series.std()
        roll_std14 = rev_series.tail(14).std() if len(rev_series) >= 14 else rev_series.std()
        roll_std30 = rev_series.tail(30).std() if len(rev_series) >= 30 else rev_series.std()
        expanding_mean = np.mean(recent_revenue)

        features = np.array([[
            dow, month, day, quarter, year, is_weekend, is_month_start,
            is_month_end, week_of_year, lag1, lag2, lag3, lag7, lag14, lag30,
            roll_mean7, roll_mean14, roll_mean30, roll_std7, roll_std14,
            roll_std30, expanding_mean,
        ]])

        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        future_predictions.append(max(pred, 0))  # Revenue can't be negative

        # Update recent revenue for next iteration
        recent_revenue.append(pred)
        if len(recent_revenue) > 60:
            recent_revenue = recent_revenue[-60:]

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Revenue": future_predictions,
    })

    # ── Plot Future Forecast ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 7))

    # Historical (last 60 days)
    hist = df.tail(60)
    ax.plot(hist["Date"], hist["Revenue"], color=ACCENT2, linewidth=2, label="Historical Revenue")

    # Future forecast
    ax.plot(future_df["Date"], future_df["Predicted_Revenue"], color=ACCENT,
            linewidth=2.5, linestyle="--", marker="o", markersize=4,
            label=f"{horizon}-Day Forecast")

    # Confidence band (using rolling std as proxy)
    rolling_std = df["Revenue"].tail(30).std()
    ax.fill_between(
        future_df["Date"],
        future_df["Predicted_Revenue"] - 1.96 * rolling_std,
        future_df["Predicted_Revenue"] + 1.96 * rolling_std,
        alpha=0.15, color=ACCENT, label="95% Confidence Band",
    )

    # Divider line
    ax.axvline(x=df["Date"].max(), color="gray", linestyle=":", linewidth=2, alpha=0.7)
    ax.annotate("Forecast Start", xy=(df["Date"].max(), ax.get_ylim()[1] * 0.95),
                fontsize=11, fontweight="bold", color="gray", ha="left")

    ax.set_title(f"{horizon}-Day Revenue Forecast", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.legend(fontsize=11)
    save_fig(fig, "24_future_forecast_30day")

    # Save forecast CSV
    forecast_csv = REPORTS_DIR / "future_forecast_30day.csv"
    future_df.to_csv(forecast_csv, index=False)
    log.info("Future forecast saved: %s", forecast_csv)

    # Print forecast summary
    print("\n" + "=" * 60)
    print(f"  {horizon}-DAY FUTURE REVENUE FORECAST")
    print("=" * 60)
    print(f"  Forecast Period  : {future_dates[0].date()} to {future_dates[-1].date()}")
    print(f"  Total Forecast   : ${future_df['Predicted_Revenue'].sum():,.2f}")
    print(f"  Daily Average    : ${future_df['Predicted_Revenue'].mean():,.2f}")
    print(f"  Highest Day      : ${future_df['Predicted_Revenue'].max():,.2f} ({future_df.loc[future_df['Predicted_Revenue'].idxmax(), 'Date'].date()})")
    print(f"  Lowest Day       : ${future_df['Predicted_Revenue'].min():,.2f} ({future_df.loc[future_df['Predicted_Revenue'].idxmin(), 'Date'].date()})")
    print("=" * 60)

    return future_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def run_forecasting():
    print("\n" + "=" * 60)
    print("  RETAIL AI ANALYTICS - PHASE 4")
    print("  Time-Series Revenue Forecasting")
    print("=" * 60 + "\n")

    # Load data
    log.info("Loading processed data ...")
    df = pd.read_csv(PROCESSED_CSV, parse_dates=["Date"])
    log.info("Loaded %s rows x %s columns", f"{len(df):,}", df.shape[1])

    # Prepare daily time series
    daily = prepare_daily_data(df)

    # Create features for ML models
    featured = create_time_features(daily)
    log.info("Feature matrix: %d rows x %d columns", featured.shape[0], featured.shape[1])

    # Chronological split (80/20)
    train, test = chronological_split(featured, test_ratio=0.2)

    # ── Run All Models ───────────────────────────────────────────────────────
    all_predictions = {}
    all_metrics = []

    # Model 1: Naive
    print("\n  Training models ...")
    print("  " + "-" * 50)
    preds = naive_forecast(train, test)
    all_predictions["Naive Forecast"] = preds
    all_metrics.append(evaluate_model(test["Revenue"], preds, "Naive Forecast"))

    # Model 2: Moving Average
    preds = moving_average_forecast(train, test, window=7)
    all_predictions["Moving Average (7d)"] = preds
    all_metrics.append(evaluate_model(test["Revenue"], preds, "Moving Average (7d)"))

    # Model 3: Exponential Smoothing
    preds = exponential_smoothing_forecast(train, test)
    all_predictions["Exponential Smoothing"] = preds
    all_metrics.append(evaluate_model(test["Revenue"], preds, "Exponential Smoothing"))

    # Model 4: ARIMA
    preds = arima_forecast(train, test)
    all_predictions["ARIMA"] = preds
    all_metrics.append(evaluate_model(test["Revenue"], preds, "ARIMA"))

    # Model 5: Random Forest
    preds, importance_rf, model_rf = ml_forecast(train, test, model_type="rf")
    all_predictions["Random Forest"] = preds
    all_metrics.append(evaluate_model(test["Revenue"], preds, "Random Forest"))

    # Model 6: Gradient Boosting
    preds_gb, importance_gb, model_gb = ml_forecast(train, test, model_type="gb")
    all_predictions["Gradient Boosting"] = preds_gb
    all_metrics.append(evaluate_model(test["Revenue"], preds_gb, "Gradient Boosting"))

    # ── Model Comparison ─────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    best_model = plot_model_comparison(test, all_predictions, all_metrics)
    plot_feature_importance(importance_rf)

    # Save metrics
    metrics_csv = REPORTS_DIR / "forecasting_model_comparison.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Save best model info
    model_info = {
        "best_model": best_model,
        "metrics": all_metrics,
        "train_period": f"{train['Date'].min().date()} to {train['Date'].max().date()}",
        "test_period": f"{test['Date'].min().date()} to {test['Date'].max().date()}",
    }
    model_info_path = MODELS_DIR / "model_info.json"
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    # ── Print Results ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("=" * 60)
    print(f"\n  BEST MODEL: {best_model}")
    print(f"  R2 Score: {metrics_df.loc[metrics_df['Model'] == best_model, 'R2 Score'].values[0]:.4f}")
    print(f"  MAPE: {metrics_df.loc[metrics_df['Model'] == best_model, 'MAPE (%)'].values[0]:.2f}%")

    # ── Future Forecast ──────────────────────────────────────────────────────
    future_df = generate_future_forecast(daily, best_model, all_predictions, horizon=30)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  PHASE 4 SUMMARY")
    print("-" * 60)
    print(f"  Models compared     : 6")
    print(f"  Best model          : {best_model}")
    print(f"  Charts generated    : 5")
    print(f"  Metrics saved       : {metrics_csv}")
    print(f"  Forecast saved      : reports/future_forecast_30day.csv")
    print(f"  Model info saved    : {model_info_path}")
    print("-" * 60)
    print("  PHASE 4 COMPLETE")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    run_forecasting()