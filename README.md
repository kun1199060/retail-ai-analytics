# Retail AI Analytics

A retail analytics and inventory-planning portfolio project that combines historical sales analysis, demand forecasting, simulated stockout-risk analysis, inventory optimization, and business recommendations in an interactive Streamlit dashboard.

> **Important data limitation:** The source sales dataset does not contain actual on-hand inventory, supplier lead times, or observed stockout labels. Inventory levels, lead times, demand variability, and stockout labels in this project are simulated or assumption-based. The inventory and stockout outputs are demonstrations of analytical workflows—not actual purchase orders, observed stock positions, or validated real-world stockout predictions.

## Dashboard

The Streamlit application contains six sections:

1. **Executive Overview** — high-level record counts, simulated reorder/risk indicators, and inventory-status visualizations.
2. **Sales Analytics** — historical daily and monthly sales trends, weekday patterns, leading stores, and product categories.
3. **Demand Forecasting** — model comparison, validation metrics, actual-versus-predicted visualization, and a validation prediction sample.
4. **Stockout Risk** — analysis of simulated stockout-risk records and the synthetic risk distribution.
5. **Inventory Optimization** — configurable filters, inventory-status summaries, safety-stock/reorder-point calculations, and recommended order quantities.
6. **Business Recommendations** — prioritized category/store recommendations and downloadable recommendation data.

## Technology Stack

- **Language:** Python
- **Data handling:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Interactive dashboard:** Streamlit
- **Model persistence:** joblib / pickle
- **Development:** Jupyter Notebook and Visual Studio Code

## Dataset

This project uses the **Store Sales – Time Series Forecasting** dataset from Kaggle.

Expected raw files in `data/raw/`:

- `train.csv`
- `test.csv`
- `stores.csv`
- `oil.csv`
- `holidays_events.csv`
- `transactions.csv`
- `sample_submission.csv`

The training data contains historical store/product-family sales, dates, promotion counts, and identifiers. Download the dataset from Kaggle and place the CSV files in `data/raw/`. Do not upload the raw dataset to GitHub unless you have checked its license and repository-size implications.

## Project Structure

The repository is organized approximately as follows:

```text
Retail_AI_Project/
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/                 # Downloaded Kaggle CSV files (not committed)
│   └── processed/           # Generated feature/inventory datasets
├── models/                  # Trained model artifacts
├── outputs/                 # Generated charts and validation output
├── reports/                 # Generated model and business reports
├── notebooks/               # Exploratory notebooks
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── optimization/
│   └── recommendations/
├── advanced_eda.py
├── business_recommendations.py
├── eda.py
├── evaluate_model.py
├── feature_engineering.py
├── inspect_data.py
├── inventory_optimization.py
├── stockout_risk.py
├── train_forecasting_model.py
├── train_stockout_model.py
├── validate_features.py
├── requirements.txt
├── README.md
└── .gitignore
```

Some folders may be empty until the corresponding scripts or notebooks are added.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Retail_AI_Project
```

### 2. Create and activate a virtual environment

On Windows Command Prompt:

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` does not yet list all dashboard packages, install them with:

```cmd
pip install streamlit plotly
```

### 4. Add the dataset

Download the Kaggle dataset and place the required CSV files in:

```text
data/raw/
```

## Run the Project

Run the scripts from the repository root, in this order, using the activated virtual environment:

```cmd
python inspect_data.py
python eda.py
python advanced_eda.py
python feature_engineering.py
python validate_features.py
python train_forecasting_model.py
python evaluate_model.py
python stockout_risk.py
python train_stockout_model.py
python inventory_optimization.py
python business_recommendations.py
```

The scripts create processed datasets, trained models, charts, and reports used by the dashboard. Exact outputs are written under `data/processed/`, `models/`, `outputs/`, and `reports/`.

### Launch the dashboard

After the pipeline has generated its required files:

```cmd
streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`) in a browser.

## Demand Forecasting Results

The current validation run reported the following results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 65.67 | 229.20 | 0.9670 |
| HistGradientBoosting | 69.42 | 231.42 | 0.9664 |
| Linear Regression | 77.53 | 241.31 | 0.9635 |

Random Forest was the best model in this recorded comparison based on the reported validation metrics.

**Validation caveat:** These are the results of the project's current validation procedure. They should not be presented as guaranteed future accuracy or as the result of a fully recursive multi-step production forecast.

## Synthetic Inventory and Stockout Methodology

Because actual inventory and stockout observations are absent from the source dataset, the project creates a demonstration dataset using assumptions:

- Daily demand is derived from historical sales features.
- Available inventory is generated synthetically relative to estimated daily demand.
- Supplier lead time is simulated within a 1–7 day range.
- A synthetic stockout-risk label is assigned by comparing generated available inventory with estimated demand over the lead-time period.
- Inventory safety stock uses an assumed demand-variability factor and service factor.
- The inventory optimization module calculates an illustrative reorder point and recommended order quantity from those assumptions.

The stockout model is trained to reproduce a label derived from the synthetic rule and related generated variables. Its very high test metrics therefore **do not establish real-world stockout prediction capability**. Real deployment would require actual inventory snapshots, replenishment/lead-time records, and observed stockout outcomes, followed by a leakage-safe, time-based evaluation.

## Business and Operational Use

The dashboard is intended as a portfolio demonstration of how retail data can be transformed into:

- Historical sales insights
- Short-term demand estimates
- Illustrative stockout-risk flags
- Reorder-point and safety-stock calculations
- Category/store prioritization
- Downloadable business recommendations

Before operational use, replace simulated inventory inputs with verified operational data, confirm business constraints, and validate the calculations with domain stakeholders.

## Future Improvements

- Use real inventory-on-hand and supplier lead-time data.
- Perform time-based and store/category-aware validation.
- Build leakage-safe stockout labels from observed stockout events.
- Compare forecasting models using rolling-origin backtesting.
- Add cost-aware inventory optimization (holding, ordering, and shortage costs).
- Add automated data-quality checks and a repeatable pipeline.
- Deploy the dashboard to a suitable hosting platform after validating data access and security.

## Author / Project Information

**Project:** Retail AI Analytics  
**Focus:** Retail analytics, demand forecasting, simulated stockout-risk analysis, inventory optimization, and business recommendations.

Add your name, team members, university details, project guide, and GitHub repository URL here before final submission.
