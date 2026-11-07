from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Retail AI Analytics", page_icon="📊", layout="wide")
st.markdown("""
<style>
[data-testid="stMetric"]{background:#ffffff;border:1px solid #d8e0ea;padding:14px;border-radius:12px}
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricDelta"]{color:#172033 !important}
[data-testid="stMetric"] label, [data-testid="stMetric"] p{color:#344054 !important}
.block-container{padding-top:1.5rem}
.notice{background:#fff7e6;color:#392b00;border-left:5px solid #e5a000;padding:12px;border-radius:6px}
.notice b{color:#241b00}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def csv(path, **kwargs):
    return pd.read_csv(path, **kwargs) if path.exists() else None

def chart_image(name, caption):
    p = OUT / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"Missing chart: outputs/{name}")

def fmt(x):
    return f"{x:,.0f}"

def metric_value(frame, aliases):
    """Read metrics from either a one-row or Metric/Value-style CSV."""
    def norm(value):
        return str(value).strip().lower().replace(" ", "_").replace("²", "2")

    alias_set = {norm(x) for x in aliases}
    if frame is None or frame.empty:
        return None

    # Common wide format: one row with MAE, RMSE, and R2 columns.
    row = frame.iloc[0]
    values = {norm(k): v for k, v in row.items()}
    for key in alias_set:
        if key in values:
            try:
                return float(values[key])
            except (TypeError, ValueError):
                pass

    # Also support long format: columns such as Metric and Value.
    metric_col = next((c for c in frame.columns if norm(c) in {"metric", "measure", "name"}), None)
    value_col = next((c for c in frame.columns if norm(c) in {"value", "score", "result"}), None)
    if metric_col is not None and value_col is not None:
        for _, item in frame.iterrows():
            if norm(item[metric_col]) in alias_set:
                try:
                    return float(item[value_col])
                except (TypeError, ValueError):
                    return None
    return None

st.sidebar.title("📊 Retail AI Analytics")
page = st.sidebar.radio("Dashboard", [
    "Executive Overview", "Sales Analytics", "Demand Forecasting",
    "Stockout Risk", "Inventory Optimization", "Business Recommendations"
])
st.title("Retail AI Analytics Dashboard")
st.caption("Sales performance | Demand forecasting | Stockout risk | Inventory planning")

st.markdown(
    '<div class="notice"><b>Data limitation:</b> Inventory quantities, lead times, demand variability, '
    'and stockout labels are simulated or assumption-based. Results are analytical demonstrations, '
    'not actual stock records or purchase orders.</div>', unsafe_allow_html=True
)

inv = csv(DATA / "optimized_inventory.csv")
if inv is None:
    st.error("Missing data/processed/optimized_inventory.csv. Run inventory_optimization.py first.")
    st.stop()

for col in ["recommended_order_quantity", "stockout_risk"]:
    if col in inv:
        inv[col] = pd.to_numeric(inv[col], errors="coerce").fillna(0)
if "date" in inv:
    inv["date"] = pd.to_datetime(inv["date"], errors="coerce")

cats = csv(REPORTS / "category_business_analysis.csv")
stores = csv(REPORTS / "store_business_analysis.csv")
recs = csv(REPORTS / "business_recommendations.csv")
metrics = csv(REPORTS / "evaluation_metrics.csv")
model_comparison = csv(REPORTS / "model_comparison.csv")
stockout_comparison = csv(REPORTS / "stockout_model_comparison.csv")
preds = csv(OUT / "validation_predictions.csv")
status = inv["inventory_status"] if "inventory_status" in inv else pd.Series(dtype=str)

n = len(inv)
reorder = int((status == "Reorder Required").sum())
no_demand = int((status == "No Demand").sum())
sufficient = int((status == "Stock Sufficient").sum())
risk = int(inv["stockout_risk"].sum()) if "stockout_risk" in inv else 0
order_qty = inv["recommended_order_quantity"].sum() if "recommended_order_quantity" in inv else 0

if page == "Executive Overview":
    st.header("Executive Overview")
    cols = st.columns(4)
    cols[0].metric("Records analysed", f"{n:,}")
    cols[1].metric("Reorder required", f"{reorder:,}", f"{reorder/n*100:.2f}%" if n else None)
    cols[2].metric("Simulated stockout risk", f"{risk:,}", f"{risk/n*100:.2f}%" if n else None)
    cols[3].metric("Recommended order quantity", fmt(order_qty))
    st.subheader("Inventory status")
    d = pd.DataFrame({"Status":["Reorder Required","No Demand","Stock Sufficient"],
                      "Records":[reorder,no_demand,sufficient]})
    a,b = st.columns(2)
    a.plotly_chart(px.pie(d,names="Status",values="Records",hole=.55,title="Inventory status split"),width="stretch")
    b.plotly_chart(px.bar(d,x="Status",y="Records",color="Status",title="Records by status"),width="stretch")
    st.subheader("Sales snapshots")
    a,b=st.columns(2)
    with a:
        chart_image("daily_sales_trend.png","Daily sales trend")
        chart_image("top_product_categories.png","Top product categories")
    with b:
        chart_image("monthly_sales_trend.png","Monthly sales trend")
        chart_image("top_10_stores.png","Top 10 stores")
    if cats is not None and not cats.empty:
        top=cats.sort_values("total_order_quantity",ascending=False).iloc[0]
        st.info(f"Highest replenishment category in this synthetic sample: {top['family']} "
                f"({top['total_order_quantity']:,.0f} units).")

elif page == "Sales Analytics":
    st.header("Sales Analytics")
    a,b=st.columns(2)
    with a:
        chart_image("daily_sales_trend.png","Daily sales trend")
        chart_image("weekday_sales.png","Sales by weekday")
        chart_image("top_product_categories.png","Top product categories")
    with b:
        chart_image("monthly_sales_trend.png","Monthly sales trend")
        chart_image("top_10_stores.png","Top 10 stores")
    st.info("These charts are pre-generated from the historical sales dataset. The dashboard avoids loading the full 3-million-row training file.")
    if cats is not None:
        st.subheader("Category replenishment context (simulated inventory)")
        st.dataframe(cats.sort_values("total_order_quantity",ascending=False),width="stretch",hide_index=True)

elif page == "Demand Forecasting":
    st.header("Demand Forecasting")
    if metrics is not None and not metrics.empty:
        r=metrics.iloc[0]
        mae=metric_value(metrics,["mae","mean_absolute_error"])
        rmse=metric_value(metrics,["rmse","root_mean_squared_error"])
        r2=metric_value(metrics,["r2","r2_score","r_squared","r_square"])
        a,b,c=st.columns(3)
        a.metric("MAE",f"{mae:,.2f}" if mae is not None else "Not available")
        b.metric("RMSE",f"{rmse:,.2f}" if rmse is not None else "Not available")
        c.metric("R²",f"{r2:.4f}" if r2 is not None else "Not available")
        if any(v is None for v in [mae,rmse,r2]):
            st.warning("The evaluation CSV was found, but its column names do not match the expected MAE, RMSE, and R² fields. Check reports/evaluation_metrics.csv.")
    else:
        st.warning("reports/evaluation_metrics.csv not found. Run evaluate_model.py first.")
    a,b=st.columns(2)
    with a: chart_image("actual_vs_predicted.png","Actual vs predicted sales")
    with b: chart_image("prediction_error_distribution.png","Prediction error distribution")
    if model_comparison is not None:
        st.subheader("Forecast model comparison")
        st.dataframe(model_comparison,width="stretch",hide_index=True)
    if preds is not None:
        st.subheader("Validation prediction sample")
        st.dataframe(preds.head(1000),width="stretch",hide_index=True)
        st.download_button("Download validation predictions",preds.to_csv(index=False).encode(),
                           "validation_predictions.csv","text/csv")
    st.caption("These are historical holdout validation results, not proof of production-grade recursive multi-day forecasting accuracy.")

elif page == "Stockout Risk":
    st.header("Stockout Risk Analysis")
    a,b,c=st.columns(3)
    a.metric("Risk-flagged records",f"{risk:,}")
    b.metric("Risk percentage",f"{risk/n*100:.2f}%" if n else "0%")
    c.metric("Records analysed",f"{n:,}")
    d=pd.DataFrame({"Risk":["Simulated risk","No simulated risk"],
                    "Records":[risk,n-risk]})
    st.plotly_chart(px.pie(d,names="Risk",values="Records",hole=.55),width="stretch")
    if stockout_comparison is not None:
        st.subheader("Stockout model comparison")
        st.dataframe(stockout_comparison,width="stretch",hide_index=True)
    st.warning("The stockout target was synthetically derived from assumed inventory and demand values. Very high model scores do not demonstrate real-world predictive skill.")
    risk_df=inv[inv["stockout_risk"]==1].copy()
    a,b=st.columns(2)
    if "family" in risk_df:
        opts=["All"]+sorted(risk_df.family.dropna().astype(str).unique())
        cat=a.selectbox("Product category",opts)
        if cat!="All": risk_df=risk_df[risk_df.family.astype(str)==cat]
    if "store_nbr" in risk_df:
        opts=["All"]+sorted(risk_df.store_nbr.dropna().unique())
        store=b.selectbox("Store",opts)
        if store!="All": risk_df=risk_df[risk_df.store_nbr==store]
    cols=[x for x in ["date","store_nbr","family","daily_demand","available_inventory","lead_time","lead_time_demand","recommended_order_quantity"] if x in risk_df]
    st.write(f"Matching risk records: {len(risk_df):,}")
    st.dataframe(risk_df[cols].head(1000),width="stretch",hide_index=True)
    st.download_button("Download filtered risk records",risk_df[cols].to_csv(index=False).encode(),
                       "filtered_simulated_stockout_risk.csv","text/csv")

elif page == "Inventory Optimization":
    st.header("Inventory Optimization")
    st.caption("Reorder point and order quantity calculations are based on modelling assumptions.")
    filtered=inv.copy()
    if "inventory_status" in filtered:
        opts=sorted(filtered.inventory_status.dropna().unique())
        selected=st.multiselect("Inventory status",opts,default=opts)
        filtered=filtered[filtered.inventory_status.isin(selected)]
    a,b=st.columns(2)
    if "family" in filtered:
        opts=["All"]+sorted(filtered.family.dropna().astype(str).unique())
        cat=a.selectbox("Category",opts)
        if cat!="All": filtered=filtered[filtered.family.astype(str)==cat]
    if "store_nbr" in filtered:
        opts=["All"]+sorted(filtered.store_nbr.dropna().unique())
        store=b.selectbox("Store",opts)
        if store!="All": filtered=filtered[filtered.store_nbr==store]
    a,b,c=st.columns(3)
    a.metric("Filtered records",f"{len(filtered):,}")
    b.metric("Order quantity",fmt(filtered.recommended_order_quantity.sum()))
    c.metric("Average safety stock",fmt(filtered.safety_stock.mean()) if "safety_stock" in filtered else "N/A")
    cols=[x for x in ["date","store_nbr","family","average_daily_demand","available_inventory","lead_time","safety_stock","reorder_point","recommended_order_quantity","inventory_status"] if x in filtered]
    st.dataframe(filtered[cols].head(2000),width="stretch",hide_index=True)
    st.caption("Showing up to 2,000 rows. Download to obtain all filtered records.")
    st.download_button("Download filtered inventory",filtered.to_csv(index=False).encode(),
                       "filtered_inventory_optimization.csv","text/csv")
    a,b=st.columns(2)
    with a: chart_image("inventory_status_distribution.png","Inventory status distribution")
    with b: chart_image("top_10_reorder_categories.png","Top reorder categories")
    st.info("Safety stock uses a 1.65 service factor and demand variability assumed at 20% of average demand. These are assumptions, not measured operational values.")

elif page == "Business Recommendations":
    st.header("Business Recommendations")
    if recs is not None and not recs.empty:
        st.metric("Recommendations available",f"{len(recs):,}")
        if "priority" in recs:
            opts=["All"]+sorted(recs.priority.dropna().astype(str).unique())
            p=st.selectbox("Priority filter",opts)
            view=recs if p=="All" else recs[recs.priority.astype(str)==p]
        else: view=recs
        st.dataframe(view,width="stretch",hide_index=True)
        st.download_button("Download recommendations",recs.to_csv(index=False).encode(),
                           "business_recommendations.csv","text/csv")
    else:
        st.warning("reports/business_recommendations.csv not found.")
    a,b=st.columns(2)
    if cats is not None and not cats.empty:
        with a:
            st.subheader("Top categories")
            top=cats.sort_values("total_order_quantity",ascending=False).head(10)
            st.plotly_chart(px.bar(top,x="total_order_quantity",y="family",orientation="h"),width="stretch")
            st.dataframe(top,width="stretch",hide_index=True)
    if stores is not None and not stores.empty:
        with b:
            st.subheader("Top stores")
            top=stores.sort_values("total_order_quantity",ascending=False).head(10)
            top["store_nbr"]=top["store_nbr"].astype(str)
            st.plotly_chart(px.bar(top,x="total_order_quantity",y="store_nbr",orientation="h"),width="stretch")
            st.dataframe(top,width="stretch",hide_index=True)
    summary=REPORTS/"executive_business_summary.txt"
    if summary.exists():
        txt=summary.read_text(encoding="utf-8",errors="replace")
        st.subheader("Executive business summary")
        st.text(txt)
        st.download_button("Download executive summary",txt,"executive_business_summary.txt","text/plain")

st.divider()
st.caption("Retail AI Analytics | Portfolio demonstration | Validate assumptions before operational use.")
