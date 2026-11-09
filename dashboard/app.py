"""
Retail Analytics & AI-Powered Sales Forecasting System
Enterprise-Grade Premium Dashboard

Run:
    python -m streamlit run dashboard/app.py

Author: Kunal | Project: Unlox Capstone
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import sys

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail AI Analytics Pro",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "retail_sales_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

# ── Colors ───────────────────────────────────────────────────────────────────
C = {
    "blue": "#3b82f6",
    "purple": "#8b5cf6",
    "pink": "#ec4899",
    "green": "#10b981",
    "orange": "#f59e0b",
    "red": "#ef4444",
    "cyan": "#06b6d4",
    "yellow": "#eab308",
}


# ── Safe Table (avoids pyarrow DLL issue permanently) ────────────────────────
def safe_table(data):
    """Display table as HTML — completely avoids pyarrow."""
    try:
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to styled HTML table
            html = "<div style='overflow-x:auto;'><table style='width:100%;border-collapse:collapse;color:#e2e8f0;font-size:0.85rem;'>"
            html += "<thead><tr>"
            for col in data.columns:
                html += f"<th style='background:rgba(102,126,234,0.2);padding:10px 12px;text-align:left;border-bottom:2px solid rgba(102,126,234,0.3);font-weight:700;white-space:nowrap;'>{col}</th>"
            html += "</tr></thead><tbody>"
            for _, row in data.iterrows():
                html += "<tr>"
                for col in data.columns:
                    val = row[col]
                    if isinstance(val, float):
                        if abs(val) >= 1e9:
                            val = f"${val/1e9:.2f}B"
                        elif abs(val) >= 1e6:
                            val = f"${val/1e6:.1f}M"
                        elif abs(val) >= 1e3:
                            val = f"{val:,.1f}"
                        else:
                            val = f"{val:.4f}" if abs(val) < 1 else f"{val:,.2f}"
                    html += f"<td style='padding:8px 12px;border-bottom:1px solid rgba(255,255,255,0.05);'>{val}</td>"
                html += "</tr>"
            html += "</tbody></table></div>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            # For non-DataFrame, use code block
            st.code(str(data), language=None)
    except Exception:
        st.code(str(data), language=None)

def download_btn(label, data, filename, mime="text/csv"):
    """Download button that avoids pyarrow."""
    try:
        st.markdown(f"""
        <div style="margin:0.5rem 0;">
            <a href="data:{mime};base64,{__import__('base64').b64encode(data.encode()).decode()}"
               download="{filename}"
               style="display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:0.5rem 1.5rem;border-radius:8px;text-decoration:none;font-weight:600;">
                {label}
            </a>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.code(f"Save as: {filename}")


# ── Dark Plotly Theme ────────────────────────────────────────────────────────
def dark(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    return fig


# ── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%) !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important; border-right: 1px solid rgba(255,255,255,0.08) !important; }

.hero { background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.1), rgba(236,72,153,0.05)); padding: 2.5rem 2rem; border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 2rem; position: relative; overflow: hidden; }
.hero-title { font-size: 2.3rem; font-weight: 900; background: linear-gradient(135deg, #c7d2fe, #e879f9, #fbbf24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px; }
.hero-sub { font-size: 1rem; color: #94a3b8; margin-top: 0.5rem; letter-spacing: 1px; }
.hero-badge { display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 1px; margin-top: 0.8rem; }

.kpi-card { background: rgba(255,255,255,0.04); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 1.5rem; text-align: center; transition: all 0.3s ease; position: relative; overflow: hidden; }
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 8px 30px rgba(102,126,234,0.2); border-color: rgba(102,126,234,0.3); }
.kpi-card::after { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 3px; background: linear-gradient(90deg, #667eea, #764ba2, #f093fb); }
.kpi-icon { font-size: 1.5rem; margin-bottom: 0.3rem; }
.kpi-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 2px; font-weight: 600; }
.kpi-value { font-size: 1.6rem; font-weight: 800; color: #f1f5f9; margin: 0.3rem 0; }
.kpi-delta { font-size: 0.8rem; color: #94a3b8; }

.section-title { font-size: 1.4rem; font-weight: 700; color: #e2e8f0; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid rgba(255,255,255,0.08); }

.insight-box { background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.2); border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0; color: #e2e8f0; transition: all 0.3s ease; }
.insight-box:hover { border-color: rgba(102,126,234,0.4); }

.rec-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.3rem; margin: 0.6rem 0; color: #e2e8f0; transition: all 0.3s ease; }
.rec-card:hover { transform: translateX(4px); box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
.rec-card.critical { border-left: 4px solid #ef4444; }
.rec-card.high { border-left: 4px solid #f59e0b; }
.rec-card.medium { border-left: 4px solid #3b82f6; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: 700; letter-spacing: 1px; }
.badge-critical { background: rgba(239,68,68,0.2); color: #ef4444; }
.badge-high { background: rgba(245,158,11,0.2); color: #f59e0b; }
.badge-medium { background: rgba(59,130,246,0.2); color: #3b82f6; }

.gradient-bar { height: 3px; background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c, #667eea); background-size: 200% 100%; animation: gradMove 3s ease infinite; border-radius: 2px; margin: 1.5rem 0; }
@keyframes gradMove { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: rgba(255,255,255,0.03); border-radius: 12px; padding: 4px; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea, #764ba2) !important; color: white !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: rgba(102,126,234,0.3); border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Data Loading (Cached) ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


@st.cache_data
def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


df = load_data()
forecast_df = load_csv_safe(REPORTS_DIR / "future_forecast_30day.csv")
model_comparison = load_csv_safe(REPORTS_DIR / "forecasting_model_comparison.csv")
store_profiles = load_csv_safe(REPORTS_DIR / "store_profiles.csv")
recommendations = load_csv_safe(REPORTS_DIR / "business_recommendations_v2.csv")
whatif_df = load_csv_safe(REPORTS_DIR / "whatif_scenarios.csv")
hierarchical_df = load_csv_safe(REPORTS_DIR / "hierarchical_forecast.csv")
stockout_risk = load_csv_safe(REPORTS_DIR / "stockout_risk_prediction.csv")
category_deep = load_csv_safe(REPORTS_DIR / "category_deep_dive.csv")
store_deep = load_csv_safe(REPORTS_DIR / "store_deep_dive.csv")
anomaly_report = load_csv_safe(REPORTS_DIR / "anomaly_report.csv")


# ── Helper Functions ─────────────────────────────────────────────────────────
def hero(title, subtitle, badge="AI-Powered Analytics"):
    st.markdown(
        f'<div class="hero"><div class="hero-title">{title}</div>'
        f'<div class="hero-sub">{subtitle}</div>'
        f'<div class="hero-badge">{badge}</div></div>',
        unsafe_allow_html=True,
    )


def kpi(label, value, delta="", icon=""):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-delta">{delta}</div></div>',
        unsafe_allow_html=True,
    )


def section(icon, title):
    st.markdown(
        f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True
    )


def insight(title, content):
    st.markdown(
        f'<div class="insight-box"><strong style="color:#818cf8">{title}</strong><br>{content}</div>',
        unsafe_allow_html=True,
    )


def bar():
    st.markdown('<div class="gradient-bar"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:1rem 0">'
        '<div style="font-size:1.5rem;font-weight:900;'
        "background:linear-gradient(135deg,#c7d2fe,#e879f9);"
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent">'
        "◆ RETAIL AI PRO</div>"
        '<div style="color:#64748b;font-size:0.75rem;letter-spacing:2px;'
        'text-transform:uppercase;margin-top:0.3rem">Enterprise Analytics</div></div>',
        unsafe_allow_html=True,
    )
    bar()

    page = st.radio(
        "Nav",
        [
            "🏠 Executive Dashboard",
            "📈 Sales Intelligence",
            "🏪 Store Performance",
            "📦 Product Intelligence",
            "👥 Customer Insights",
            "🗺️ Store Map",
            "🤖 AI Forecasting",
            "📊 Store Segmentation",
            "📋 Inventory & Risk",
            "⚡ What-If Simulator",
            "💬 AI Assistant",
            "📝 Recommendations",
            "📄 Export & Reports",
        ],
        index=0,
        label_visibility="collapsed",
    )

    bar()
    st.markdown("### 🔍 Filters")
    date_range = st.date_input(
        "Date Range",
        value=(df["Date"].min(), df["Date"].max()),
        min_value=df["Date"].min(),
        max_value=df["Date"].max(),
    )
    selected_stores = st.multiselect(
        "Stores",
        options=sorted(df["Store_ID"].unique()),
        default=sorted(df["Store_ID"].unique()),
    )
    selected_categories = st.multiselect(
        "Categories",
        options=sorted(df["Product_Category"].unique()),
        default=sorted(df["Product_Category"].unique()),
    )
    selected_regions = st.multiselect(
        "Regions",
        options=sorted(df["Region"].unique()),
        default=sorted(df["Region"].unique()),
    )
    bar()
    st.markdown(
        '<div style="text-align:center;color:#64748b;font-size:0.75rem">'
        "<p>73K Records | 8 Stores | 5 Categories</p>"
        "<p>Best Model: Random Forest (MAPE 9.49%)</p></div>",
        unsafe_allow_html=True,
    )

# ── Apply Filters ────────────────────────────────────────────────────────────
mask = (
    (df["Date"] >= pd.Timestamp(date_range[0]))
    & (df["Date"] <= pd.Timestamp(date_range[-1]))
    & (df["Store_ID"].isin(selected_stores))
    & (df["Product_Category"].isin(selected_categories))
    & (df["Region"].isin(selected_regions))
)
fdf = df[mask]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Dashboard":
    hero(
        "Retail AI Analytics Pro",
        "Enterprise Business Intelligence & AI Forecasting Platform",
        "UNLOX CAPSTONE PROJECT",
    )

    rev = fdf["Revenue"].sum()
    units = fdf["Units_Sold"].sum()
    txns = len(fdf)
    rating = fdf["Store_Rating"].mean()
    disc = fdf["Discount_Percentage"].mean()

    c = st.columns(6)
    with c[0]:
        kpi("TOTAL REVENUE", f"${rev / 1e9:.2f}B", "2-Year Total", "💰")
    with c[1]:
        kpi("UNITS SOLD", f"{units / 1e6:.2f}M", "Volume", "📦")
    with c[2]:
        kpi("TRANSACTIONS", f"{txns / 1e3:.1f}K", "Records", "🧾")
    with c[3]:
        kpi("AVG RATING", f"{rating:.2f}★", "Quality", "⭐")
    with c[4]:
        kpi("30-DAY FC", "~$1.63B", "AI Prediction", "🔮")
    with c[5]:
        kpi("BEST MODEL", "Rand. Forest", "MAPE: 9.49%", "🤖")

    bar()
    col1, col2 = st.columns([2, 1])
    with col1:
        daily = fdf.groupby("Date")["Revenue"].sum().reset_index()
        daily["MA7"] = daily["Revenue"].rolling(7, min_periods=1).mean()
        daily["MA30"] = daily["Revenue"].rolling(30, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily["Date"],
                y=daily["Revenue"],
                mode="lines",
                name="Daily",
                line=dict(color=C["blue"], width=1),
                opacity=0.3,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily["Date"],
                y=daily["MA7"],
                mode="lines",
                name="7-Day MA",
                line=dict(color=C["purple"], width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily["Date"],
                y=daily["MA30"],
                mode="lines",
                name="30-Day MA",
                line=dict(color=C["pink"], width=2.5),
                fill="tonexty",
                fillcolor="rgba(236,72,153,0.08)",
            )
        )
        fig.update_layout(
            title="Revenue Trend Analysis",
            height=420,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(dark(fig))

    with col2:
        cat = fdf.groupby("Product_Category")["Revenue"].sum().reset_index()
        fig = go.Figure(
            go.Pie(
                labels=cat["Product_Category"],
                values=cat["Revenue"],
                hole=0.6,
                marker=dict(
                    colors=[C["blue"], C["purple"], C["pink"], C["green"], C["orange"]]
                ),
                textinfo="label+percent",
            )
        )
        fig.update_layout(
            title="Revenue by Category",
            height=420,
            annotations=[
                dict(
                    text=f"${rev / 1e9:.1f}B<br>Total",
                    x=0.5,
                    y=0.5,
                    font_size=13,
                    showarrow=False,
                    font_color="#e2e8f0",
                )
            ],
        )
        st.plotly_chart(dark(fig))

    c = st.columns(3)
    with c[0]:
        sr = (
            fdf.groupby("Store_ID")["Revenue"]
            .sum()
            .sort_values(ascending=True)
            .reset_index()
        )
        fig = px.bar(
            sr,
            x="Revenue",
            y="Store_ID",
            orientation="h",
            color="Revenue",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(title="Store Revenue", height=350, showlegend=False)
        st.plotly_chart(dark(fig))
    with c[1]:
        rr = fdf.groupby("Region")["Revenue"].sum().reset_index()
        fig = go.Figure(
            go.Bar(
                x=rr["Region"],
                y=rr["Revenue"],
                marker_color=[C["blue"], C["purple"], C["pink"], C["green"]][
                    : len(rr)
                ],
            )
        )
        fig.update_layout(title="Revenue by Region", height=350)
        st.plotly_chart(dark(fig))
    with c[2]:
        mr = fdf.groupby("YearMonth")["Revenue"].sum().reset_index()
        fig = go.Figure(
            go.Scatter(
                x=mr["YearMonth"],
                y=mr["Revenue"],
                mode="lines+markers",
                line=dict(color=C["cyan"], width=2.5),
                fill="tonexty",
                fillcolor="rgba(6,182,212,0.1)",
            )
        )
        fig.update_layout(title="Monthly Revenue", height=350)
        st.plotly_chart(dark(fig))

    bar()
    section("🧠", "AI-Generated Business Insights")
    c = st.columns(3)
    with c[0]:
        insight(
            "🏆 REVENUE LEADER",
            "<b>STR_103</b> leads with <b>$5.22B</b>. Replicate its operational model.",
        )
    with c[1]:
        insight(
            "⚠️ PROMOTION ALERT",
            "Promotions reduce avg revenue by <b>4.1%</b>. Shift to targeted promotions.",
        )
    with c[2]:
        insight(
            "📈 GROWTH OPPORTUNITY",
            "<b>February</b> is peak season. Increase inventory +30%.",
        )
    c = st.columns(3)
    with c[0]:
        insight(
            "🌍 REGIONAL INSIGHT",
            "<b>West Region</b> leads with <b>37.5%</b> revenue share.",
        )
    with c[1]:
        insight(
            "📱 CATEGORY FOCUS",
            "<b>Electronics</b> generates <b>20.2%</b> of revenue.",
        )
    with c[2]:
        insight(
            "⭐ QUALITY SIGNAL",
            "Rating +0.3 improvement = <b>+2.4% revenue lift</b>.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: SALES INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Sales Intelligence":
    hero(
        "Sales Intelligence",
        "Trends, Seasonality, Growth Patterns & Anomaly Detection",
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📅 Trends", "🔄 Seasonality", "📊 Growth", "🚨 Anomalies"]
    )

    with tab1:
        daily = fdf.groupby("Date")["Revenue"].sum().reset_index()
        daily["MA7"] = daily["Revenue"].rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily["Date"],
                y=daily["Revenue"],
                mode="lines",
                name="Daily",
                line=dict(color=C["blue"], width=1),
                opacity=0.3,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily["Date"],
                y=daily["MA7"],
                mode="lines",
                name="7-Day MA",
                line=dict(color=C["pink"], width=2.5),
                fill="tonexty",
                fillcolor="rgba(236,72,153,0.1)",
            )
        )
        fig.update_layout(title="Daily Revenue + 7-Day MA", height=400)
        st.plotly_chart(dark(fig))

        monthly = (
            fdf.groupby("YearMonth")
            .agg(Revenue=("Revenue", "sum"), Units=("Units_Sold", "sum"))
            .reset_index()
        )
        c = st.columns(2)
        with c[0]:
            fig = go.Figure(
                go.Bar(
                    x=monthly["YearMonth"],
                    y=monthly["Revenue"],
                    marker_color=C["purple"],
                )
            )
            fig.update_layout(title="Monthly Revenue", height=350)
            st.plotly_chart(dark(fig))
        with c[1]:
            fig = go.Figure(
                go.Scatter(
                    x=monthly["YearMonth"],
                    y=monthly["Units"],
                    mode="lines+markers",
                    line=dict(color=C["cyan"], width=2.5),
                )
            )
            fig.update_layout(title="Monthly Units", height=350)
            st.plotly_chart(dark(fig))

    with tab2:
        wd = (
            fdf.groupby(["DayOfWeek", "DayName"])["Revenue"]
            .mean()
            .reset_index()
            .sort_values("DayOfWeek")
        )
        fig = go.Figure(
            go.Bar(
                x=wd["DayName"],
                y=wd["Revenue"],
                marker_color=[
                    C["blue"],
                    C["purple"],
                    C["pink"],
                    C["orange"],
                    C["green"],
                    C["cyan"],
                    C["red"],
                ],
            )
        )
        fig.update_layout(title="Revenue by Day of Week", height=350)
        st.plotly_chart(dark(fig))

        hm = fdf.groupby(["Year", "Month"])["Revenue"].sum().reset_index()
        pivot = hm.pivot_table(index="Year", columns="Month", values="Revenue")
        fig = px.imshow(
            pivot,
            color_continuous_scale="YlOrRd",
            text_auto=".2s",
            title="Monthly Heatmap",
        )
        fig.update_layout(height=300)
        st.plotly_chart(dark(fig))

    with tab3:
        mr = fdf.groupby("YearMonth")["Revenue"].sum().reset_index()
        mr["Growth"] = mr["Revenue"].pct_change() * 100
        fig = go.Figure(
            go.Bar(
                x=mr["YearMonth"],
                y=mr["Growth"].fillna(0),
                marker_color=[
                    C["green"] if g >= 0 else C["red"]
                    for g in mr["Growth"].fillna(0)
                ],
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
        fig.update_layout(title="MoM Growth Rate (%)", height=400)
        st.plotly_chart(dark(fig))

        yoy = fdf.groupby(["Year", "Month"])["Revenue"].sum().reset_index()
        fig = go.Figure()
        for year in sorted(yoy["Year"].unique()):
            yd = yoy[yoy["Year"] == year]
            fig.add_trace(
                go.Scatter(
                    x=yd["Month"],
                    y=yd["Revenue"],
                    mode="lines+markers",
                    name=str(year),
                    line=dict(width=2.5),
                )
            )
        fig.update_layout(
            title="Year-over-Year Comparison",
            height=400,
            xaxis=dict(
                tickvals=list(range(1, 13)),
                ticktext=[
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ],
            ),
        )
        st.plotly_chart(dark(fig))

    with tab4:
        if anomaly_report is not None and len(anomaly_report) > 0:
            st.markdown(f"### 🚨 {len(anomaly_report)} Anomalies Detected")
            daily = fdf.groupby("Date")["Revenue"].sum().reset_index()
            daily["Anom"] = daily["Date"].isin(pd.to_datetime(anomaly_report["Date"]))
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=daily[~daily["Anom"]]["Date"],
                    y=daily[~daily["Anom"]]["Revenue"],
                    mode="lines",
                    name="Normal",
                    line=dict(color=C["blue"], width=1.5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=daily[daily["Anom"]]["Date"],
                    y=daily[daily["Anom"]]["Revenue"],
                    mode="markers",
                    name="Anomaly",
                    marker=dict(color=C["red"], size=12, symbol="x"),
                )
            )
            fig.update_layout(title="Revenue Anomaly Detection", height=450)
            st.plotly_chart(dark(fig))
            safe_table(anomaly_report)
        else:
            st.info("No anomalies detected.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: STORE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏪 Store Performance":
    hero(
        "Store Performance",
        "Rankings, Deep-Dive Profiles & Regional Analysis",
    )

    if store_deep is not None:
        section("📊", "Store Performance Matrix")
        safe_table(store_deep)

    c = st.columns(2)
    with c[0]:
        sr = (
            fdf.groupby(["Store_ID", "Store_Location"])["Revenue"]
            .sum()
            .reset_index()
            .sort_values("Revenue", ascending=True)
        )
        fig = px.bar(
            sr,
            x="Revenue",
            y="Store_ID",
            orientation="h",
            color="Revenue",
            color_continuous_scale="Viridis",
            hover_data=["Store_Location"],
        )
        fig.update_layout(
            title="Revenue Ranking", height=400, showlegend=False
        )
        st.plotly_chart(dark(fig))
    with c[1]:
        sm = (
            fdf.groupby("Store_ID")
            .agg(
                Rating=("Store_Rating", "mean"),
                Revenue=("Revenue", "sum"),
                Txns=("Date", "count"),
            )
            .reset_index()
        )
        fig = px.scatter(
            sm,
            x="Rating",
            y="Revenue",
            size="Txns",
            color="Revenue",
            color_continuous_scale="RdYlGn",
            hover_name="Store_ID",
            text="Store_ID",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(title="Rating vs Revenue", height=400)
        st.plotly_chart(dark(fig))

    section("🔍", "Store Deep-Dive")
    sel = st.selectbox("Select Store", sorted(df["Store_ID"].unique()))
    sd = fdf[fdf["Store_ID"] == sel]
    if len(sd) > 0:
        c = st.columns(5)
        with c[0]:
            kpi("Revenue", f"${sd['Revenue'].sum() / 1e9:.2f}B", icon="💰")
        with c[1]:
            kpi("Units", f"{sd['Units_Sold'].sum() / 1e6:.2f}M", icon="📦")
        with c[2]:
            kpi("Transactions", f"{len(sd) / 1e3:.1f}K", icon="🧾")
        with c[3]:
            kpi("Rating", f"{sd['Store_Rating'].mean():.2f}★", icon="⭐")
        with c[4]:
            kpi(
                "Avg Discount",
                f"{sd['Discount_Percentage'].mean():.1f}%",
                icon="🏷️",
            )

        c = st.columns(2)
        with c[0]:
            sm2 = sd.groupby("YearMonth")["Revenue"].sum().reset_index()
            fig = go.Figure(
                go.Scatter(
                    x=sm2["YearMonth"],
                    y=sm2["Revenue"],
                    mode="lines+markers",
                    line=dict(color=C["purple"], width=2.5),
                    fill="tonexty",
                    fillcolor="rgba(139,92,246,0.1)",
                )
            )
            fig.update_layout(title=f"{sel} Monthly Trend", height=350)
            st.plotly_chart(dark(fig))
        with c[1]:
            sc = sd.groupby("Product_Category")["Revenue"].sum().reset_index()
            fig = go.Figure(
                go.Pie(
                    labels=sc["Product_Category"],
                    values=sc["Revenue"],
                    hole=0.5,
                    marker=dict(
                        colors=[
                            C["blue"],
                            C["purple"],
                            C["pink"],
                            C["green"],
                            C["orange"],
                        ]
                    ),
                )
            )
            fig.update_layout(title=f"{sel} Category Mix", height=350)
            st.plotly_chart(dark(fig))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PRODUCT INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Product Intelligence":
    hero("Product Intelligence", "Category, Brand & Subcategory Analytics")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Categories", "🏷️ Brands", "📋 Subcategories"]
    )
    with tab1:
        cat = (
            fdf.groupby("Product_Category")
            .agg(Revenue=("Revenue", "sum"), Units=("Units_Sold", "sum"))
            .reset_index()
            .sort_values("Revenue")
        )
        c = st.columns(2)
        with c[0]:
            fig = px.bar(
                cat,
                x="Revenue",
                y="Product_Category",
                orientation="h",
                color="Revenue",
                color_continuous_scale="Plasma",
            )
            fig.update_layout(
                title="Revenue by Category", height=350, showlegend=False
            )
            st.plotly_chart(dark(fig))
        with c[1]:
            fig = go.Figure(
                go.Pie(
                    labels=cat["Product_Category"],
                    values=cat["Units"],
                    hole=0.4,
                    marker=dict(
                        colors=[
                            C["blue"],
                            C["purple"],
                            C["pink"],
                            C["green"],
                            C["orange"],
                        ]
                    ),
                )
            )
            fig.update_layout(title="Units Share", height=350)
            st.plotly_chart(dark(fig))
        if category_deep is not None:
            safe_table(category_deep)

    with tab2:
        brand = (
            fdf.groupby("Brand")["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig = go.Figure(
            go.Bar(
                x=brand["Brand"],
                y=brand["Revenue"],
                marker=dict(color=brand["Revenue"], colorscale="Magma"),
            )
        )
        fig.update_layout(title="Top 15 Brands", height=400)
        st.plotly_chart(dark(fig))

    with tab3:
        sub = (
            fdf.groupby("Product_Subcategory")["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig = px.bar(
            sub,
            x="Revenue",
            y="Product_Subcategory",
            orientation="h",
            color="Revenue",
            color_continuous_scale="Cividis",
        )
        fig.update_layout(
            title="Top 15 Subcategories", height=450, showlegend=False
        )
        st.plotly_chart(dark(fig))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: CUSTOMER INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Insights":
    hero(
        "Customer Analytics",
        "Segments, Payment Behavior & Promotion Effectiveness",
    )

    c = st.columns(2)
    with c[0]:
        cust = fdf.groupby("Customer_Type")["Revenue"].sum().reset_index()
        fig = go.Figure(
            go.Pie(
                labels=cust["Customer_Type"],
                values=cust["Revenue"],
                hole=0.5,
                marker=dict(colors=[C["blue"], C["purple"], C["pink"]]),
            )
        )
        fig.update_layout(title="Revenue by Customer Type", height=350)
        st.plotly_chart(dark(fig))
    with c[1]:
        pay = fdf.groupby("Payment_Mode")["Revenue"].sum().reset_index()
        fig = go.Figure(
            go.Bar(
                x=pay["Payment_Mode"],
                y=pay["Revenue"],
                marker=dict(color=pay["Revenue"], colorscale="Blues"),
            )
        )
        fig.update_layout(title="Revenue by Payment Mode", height=350)
        st.plotly_chart(dark(fig))

    cc = (
        fdf.groupby(["Customer_Type", "Product_Category"])["Revenue"]
        .sum()
        .reset_index()
    )
    fig = px.sunburst(
        cc,
        path=["Customer_Type", "Product_Category"],
        values="Revenue",
        color_discrete_sequence=[
            C["blue"],
            C["purple"],
            C["pink"],
            C["green"],
            C["orange"],
        ],
    )
    fig.update_layout(title="Customer x Category Breakdown", height=500)
    st.plotly_chart(dark(fig))

    section("🎯", "Promotion Effectiveness")
    c = st.columns(2)
    with c[0]:
        promo = fdf.copy()
        promo["Status"] = (
            promo["Promotion_Applied"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "yes": "With Promo",
                    "true": "With Promo",
                    "1": "With Promo",
                }
            )
            .fillna("No Promo")
        )
        pr = promo.groupby("Status")["Revenue"].mean().reset_index()
        fig = go.Figure(
            go.Bar(
                x=pr["Status"],
                y=pr["Revenue"],
                marker_color=[C["red"], C["green"]],
            )
        )
        fig.update_layout(title="Avg Revenue: Promo Impact", height=350)
        st.plotly_chart(dark(fig))
    with c[1]:
        disc = (
            fdf.groupby("Discount_Bucket")
            .agg(Units=("Units_Sold", "sum"), Revenue=("Revenue", "sum"))
            .reset_index()
        )
        fig = px.bar(
            disc,
            x="Discount_Bucket",
            y="Units",
            color="Revenue",
            color_continuous_scale="YlOrRd",
        )
        fig.update_layout(title="Units by Discount Level", height=350)
        st.plotly_chart(dark(fig))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6: STORE MAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Store Map":
    hero(
        "Geographic Store Intelligence",
        "India Store Locations — Revenue & Rating Visualization",
    )

    STORE_COORDS = {
        "STR_101": {"city": "Mumbai", "lat": 19.076, "lon": 72.8777},
        "STR_102": {"city": "Bangalore", "lat": 12.9716, "lon": 77.5946},
        "STR_103": {"city": "Delhi", "lat": 28.7041, "lon": 77.1025},
        "STR_104": {"city": "Chennai", "lat": 13.0827, "lon": 80.2707},
        "STR_105": {"city": "Pune", "lat": 18.5204, "lon": 73.8567},
        "STR_106": {"city": "Hyderabad", "lat": 17.385, "lon": 78.4867},
        "STR_107": {"city": "Kolkata", "lat": 22.5726, "lon": 88.3639},
        "STR_108": {"city": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
    }

    sd = (
        fdf.groupby("Store_ID")
        .agg(
            Revenue=("Revenue", "sum"),
            Units=("Units_Sold", "sum"),
            Rating=("Store_Rating", "mean"),
            Location=("Store_Location", "first"),
            Region=("Region", "first"),
        )
        .reset_index()
    )
    sd["lat"] = sd["Store_ID"].map(
        lambda x: STORE_COORDS.get(x, {}).get("lat", 0)
    )
    sd["lon"] = sd["Store_ID"].map(
        lambda x: STORE_COORDS.get(x, {}).get("lon", 0)
    )

    # Use scatter_geo instead of scatter_mapbox (compatible with all Plotly versions)
    fig = px.scatter_geo(
        sd,
        lat="lat",
        lon="lon",
        size="Revenue",
        color="Rating",
        color_continuous_scale="RdYlGn",
        hover_name="Store_ID",
        hover_data={
            "Location": True,
            "Region": True,
            "Revenue": ":,.0f",
            "Rating": ":.2f",
        },
        size_max=50,
        projection="natural earth",
    )
    fig.update_geos(
        center={"lat": 22.0, "lon": 79.0},
        projection_scale=8,
        showcountries=True,
        countrycolor="rgba(255,255,255,0.1)",
        showland=True,
        landcolor="rgba(255,255,255,0.05)",
        showocean=True,
        oceancolor="rgba(10,14,26,0.8)",
    )
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(dark(fig))

    section("Regional Performance")
    region = (
        sd.groupby("Region")
        .agg(
            Revenue=("Revenue", "sum"),
            Stores=("Store_ID", "count"),
            Avg_Rating=("Rating", "mean"),
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    safe_table(region)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7: AI FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Forecasting":
    hero(
        "AI-Powered Sales Forecasting",
        "Model Comparison, 30-Day Predictions & Hierarchical Forecasts",
    )

    if model_comparison is not None:
        section("📊", "Model Performance")
        c = st.columns(4)
        with c[0]:
            kpi("BEST MODEL", "Random Forest", "R²: 0.14", "🤖")
        with c[1]:
            kpi("MAPE", "9.49%", "Accuracy", "🎯")
        with c[2]:
            kpi("RMSE", "$6.79M", "Error", "📉")
        with c[3]:
            kpi("MODELS TESTED", "6", "Compared", "🧪")

        c = st.columns(2)
        with c[0]:
            fig = px.bar(
                model_comparison,
                x="Model",
                y="MAPE (%)",
                color="MAPE (%)",
                color_continuous_scale="RdYlGn_r",
                title="MAPE (Lower=Better)",
            )
            fig.update_layout(height=350)
            st.plotly_chart(dark(fig))
        with c[1]:
            fig = px.bar(
                model_comparison,
                x="Model",
                y="R2 Score",
                color="R2 Score",
                color_continuous_scale="RdYlGn",
                title="R² Score (Higher=Better)",
            )
            fig.update_layout(height=350)
            st.plotly_chart(dark(fig))

        safe_table(model_comparison)

    if forecast_df is not None:
        section("🔮", "30-Day Revenue Forecast")
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
        c = st.columns(4)
        with c[0]:
            kpi(
                "TOTAL FC",
                f"${forecast_df['Predicted_Revenue'].sum() / 1e9:.2f}B",
                icon="💰",
            )
        with c[1]:
            kpi(
                "DAILY AVG",
                f"${forecast_df['Predicted_Revenue'].mean() / 1e6:.1f}M",
                icon="📊",
            )
        with c[2]:
            kpi(
                "PEAK DAY",
                f"${forecast_df['Predicted_Revenue'].max() / 1e6:.1f}M",
                icon="📈",
            )
        with c[3]:
            kpi(
                "LOW DAY",
                f"${forecast_df['Predicted_Revenue'].min() / 1e6:.1f}M",
                icon="📉",
            )

        hist = df.groupby("Date")["Revenue"].sum().reset_index().tail(60)
        rs = df["Revenue"].tail(30).std()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist["Date"],
                y=hist["Revenue"],
                mode="lines",
                name="Historical",
                line=dict(color=C["blue"], width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Predicted_Revenue"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color=C["pink"], width=2.5, dash="dash"),
                marker=dict(size=5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(forecast_df["Date"]) + list(forecast_df["Date"][::-1]),
                y=list(forecast_df["Predicted_Revenue"] + 1.96 * rs)
                + list((forecast_df["Predicted_Revenue"] - 1.96 * rs)[::-1]),
                fill="toself",
                fillcolor="rgba(236,72,153,0.1)",
                line=dict(width=0),
                name="95% CI",
            )
        )
        fig.update_layout(
            title="30-Day Forecast with Confidence Band",
            height=500,
            hovermode="x unified",
        )
        st.plotly_chart(dark(fig))
        safe_table(forecast_df)

    if hierarchical_df is not None:
        section("📊", "Hierarchical Forecast")
        level = st.selectbox("Level", ["Store", "Category"])
        ma = st.slider("Months Ahead", 1, 3, 1)
        sub = hierarchical_df[
            (hierarchical_df["Level"] == level)
            & (hierarchical_df["Month_Ahead"] == ma)
        ].sort_values("Forecasted_Revenue", ascending=True)
        fig = px.bar(
            sub,
            x="Forecasted_Revenue",
            y="Entity",
            orientation="h",
            color="Forecasted_Revenue",
            color_continuous_scale="Viridis",
            title=f"{level} Forecast ({ma}M Ahead)",
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(dark(fig))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8: STORE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Store Segmentation":
    hero("Store Segmentation", "K-Means Clustering — Performance Groups")

    if store_profiles is not None:
        segs = store_profiles["Segment_Label"].unique()
        cols = st.columns(len(segs))
        for i, seg in enumerate(segs[:5]):
            sd = store_profiles[store_profiles["Segment_Label"] == seg]
            with cols[i]:
                icon = (
                    "🏆"
                    if "High" in seg
                    else "📊"
                    if "Medium" in seg
                    else "⚠️"
                )
                kpi(
                    seg.upper(),
                    f"{len(sd)} Stores",
                    f"Avg Rev: ${sd['Total_Revenue'].mean() / 1e9:.2f}B",
                    icon,
                )

        fig = px.scatter(
            store_profiles,
            x="Total_Revenue",
            y="Revenue_Growth_Rate",
            color="Segment_Label",
            size="Total_Transactions",
            hover_name="Store_ID",
            text="Store_ID",
            color_discrete_sequence=[
                C["green"],
                C["orange"],
                C["red"],
                C["blue"],
                C["purple"],
            ],
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(title="Segmentation: Revenue vs Growth", height=500)
        st.plotly_chart(dark(fig))

        display_cols = [
            c
            for c in [
                "Store_ID",
                "Store_Location",
                "Region",
                "Segment_Label",
                "Total_Revenue",
                "Total_Units",
                "Store_Rating",
                "Revenue_Growth_Rate",
                "Avg_Discount",
            ]
            if c in store_profiles.columns
        ]
        safe_table(store_profiles[display_cols])


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 9: INVENTORY & RISK
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Inventory & Risk":
    hero(
        "Inventory & Stockout Risk",
        "Stock Health Monitoring & Replenishment Intelligence",
    )

    c = st.columns(2)
    with c[0]:
        ss = fdf["Stock_Status"].value_counts().reset_index()
        ss.columns = ["Status", "Count"]
        fig = go.Figure(
            go.Pie(
                labels=ss["Status"],
                values=ss["Count"],
                hole=0.5,
                marker=dict(
                    colors=[C["red"], C["orange"], C["green"], C["blue"]]
                ),
            )
        )
        fig.update_layout(title="Inventory Status", height=350)
        st.plotly_chart(dark(fig))
    with c[1]:
        stk = fdf.groupby("Store_ID")["Stock_On_Hand"].mean().reset_index()
        fig = px.bar(
            stk,
            x="Store_ID",
            y="Stock_On_Hand",
            color="Stock_On_Hand",
            color_continuous_scale="RdYlGn",
            title="Avg Stock by Store",
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(dark(fig))

    if stockout_risk is not None:
        section("⚠️", "Stockout Risk Prediction")
        rs = (
            stockout_risk.groupby("Risk_Level")
            .agg(Count=("Store_ID", "count"), Revenue=("Revenue", "sum"))
            .reset_index()
        )
        c = st.columns(2)
        with c[0]:
            fig = px.bar(
                rs,
                x="Risk_Level",
                y="Count",
                color="Risk_Level",
                color_discrete_sequence=[
                    C["red"],
                    C["orange"],
                    C["yellow"],
                    C["green"],
                    C["cyan"],
                ],
            )
            fig.update_layout(title="Items by Risk Level", height=350)
            st.plotly_chart(dark(fig))
        with c[1]:
            fig = go.Figure(
                go.Pie(
                    labels=rs["Risk_Level"], values=rs["Revenue"], hole=0.4
                )
            )
            fig.update_layout(title="Revenue at Risk", height=350)
            st.plotly_chart(dark(fig))
        safe_table(stockout_risk.head(20))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 10: WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ What-If Simulator":
    hero(
        "What-If Business Simulator",
        "Interactive Scenario Planning & Revenue Impact",
    )

    section("🎛️", "Adjust Parameters")
    c = st.columns(3)
    with c[0]:
        dc = st.slider("🏷️ Discount Change (%)", -10, 10, 0)
    with c[1]:
        pc = st.slider("🎯 Promotion Increase (%)", 0, 20, 0, step=5)
    with c[2]:
        rc = st.slider("⭐ Rating Improvement", 0.0, 0.5, 0.0, step=0.1)

    base = 54205210
    di = -0.5 * dc + 1.2 * dc * 0.3
    pi_val = pc * 0.15
    ri = rc * 8
    ti = di + pi_val + ri
    new = base * (1 + ti / 100)
    mi = (new - base) * 30

    bar()
    c = st.columns(4)
    with c[0]:
        kpi("BASE DAILY", f"${base / 1e6:.1f}M", icon="📊")
    with c[1]:
        kpi("PROJECTED", f"${new / 1e6:.1f}M", f"{ti:+.2f}%", "🔮")
    with c[2]:
        kpi("MONTHLY IMPACT", f"${abs(mi) / 1e6:.1f}M", f"{ti:+.2f}%", "📈")
    with c[3]:
        kpi(
            "ANNUAL IMPACT",
            f"${abs(mi) * 12 / 1e6:.0f}M",
            f"{ti:+.2f}%",
            "💰",
        )

    section("📊", "Impact Breakdown")
    fig = go.Figure(
        go.Bar(
            x=["Discount", "Promotion", "Rating", "TOTAL"],
            y=[di, pi_val, ri, ti],
            marker_color=[C["blue"], C["purple"], C["green"], C["orange"]],
        )
    )
    fig.update_layout(title="Revenue Impact by Factor (%)", height=400)
    st.plotly_chart(dark(fig))

    if whatif_df is not None:
        section("📋", "Pre-Computed Scenarios")
        safe_table(whatif_df)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 11: AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💬 AI Assistant":
    hero("AI Business Assistant", "Ask Questions in Natural Language")

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from src.ai_assistant import RetailAIAssistant

        assistant = RetailAIAssistant(df)
    except Exception as e:
        st.error(f"Error: {e}")
        assistant = None

    if assistant:
        section("🚀", "Quick Questions")
        c = st.columns(3)
        with c[0]:
            if st.button("📊 Business Summary"):
                st.code(
                    assistant.answer("Give me a business summary"),
                    language=None,
                )
        with c[1]:
            if st.button("🏆 Best Store?"):
                st.code(
                    assistant.answer("Which store performs best?"),
                    language=None,
                )
        with c[2]:
            if st.button("📈 Seasonal Trends?"):
                st.code(
                    assistant.answer("What are the seasonal trends?"),
                    language=None,
                )

        c = st.columns(3)
        with c[0]:
            if st.button("💰 Total Revenue?"):
                st.code(
                    assistant.answer("What is the total revenue?"),
                    language=None,
                )
        with c[1]:
            if st.button("🎯 Promotion Impact?"):
                st.code(
                    assistant.answer("How do promotions impact sales?"),
                    language=None,
                )
        with c[2]:
            if st.button("🔮 Forecast?"):
                st.code(
                    assistant.answer("What will revenue be next month?"),
                    language=None,
                )

        bar()
        section("💬", "Ask Your Own Question")
        q = st.text_input(
            "Type here:", placeholder="e.g., Which region has highest growth?"
        )
        if q:
            with st.spinner("Analyzing..."):
                ans = assistant.answer(q)
            st.code(ans, language=None)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 12: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Recommendations":
    hero("Business Recommendations", "AI-Powered Strategic Insights")

    if recommendations is not None:
        for _, row in recommendations.iterrows():
            p = row["Priority"]
            css = (
                p.lower() if p in ["CRITICAL", "HIGH", "MEDIUM"] else "medium"
            )
            st.markdown(
                f"""
            <div class="rec-card {css}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <strong style="font-size:1.1rem;color:#f1f5f9">{row['Category']}</strong>
                    <span class="badge badge-{css}">{p}</span>
                </div>
                <p style="color:#94a3b8;margin:0.5rem 0"><strong style="color:#e2e8f0">Finding:</strong> {row['Finding']}</p>
                <p style="color:#94a3b8;margin:0.5rem 0"><strong style="color:#e2e8f0">Action:</strong> {row['Recommendation']}</p>
                <p style="color:#94a3b8;margin:0.5rem 0"><strong style="color:#e2e8f0">Impact:</strong> {row['Expected_Impact']}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    section("📄", "Executive Summary")
    try:
        summary = (REPORTS_DIR / "executive_summary.txt").read_text(
            encoding="utf-8"
        )
        st.code(summary, language=None)
    except Exception:
        st.info("Run Phase 6 to generate summary.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 13: EXPORT & REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Export & Reports":
    hero("Reports & Data Export", "Download All Reports & Data")

    section("📥", "Download Center")
    c = st.columns(2)
    with c[0]:
        download_btn(
            "📥 Filtered Data (CSV)",
            fdf.to_csv(index=False),
            "retail_filtered.csv",
            "text/csv",
        )
        if store_profiles is not None:
            download_btn(
                "📥 Store Profiles (CSV)",
                store_profiles.to_csv(index=False),
                "store_profiles.csv",
                "text/csv",
            )
        if model_comparison is not None:
            download_btn(
                "📥 Model Comparison (CSV)",
                model_comparison.to_csv(index=False),
                "model_comparison.csv",
                "text/csv",
            )
    with c[1]:
        if forecast_df is not None:
            download_btn(
                "📥 30-Day Forecast (CSV)",
                forecast_df.to_csv(index=False),
                "forecast_30day.csv",
                "text/csv",
            )
        if hierarchical_df is not None:
            download_btn(
                "📥 Hierarchical Forecast (CSV)",
                hierarchical_df.to_csv(index=False),
                "hierarchical.csv",
                "text/csv",
            )
        if whatif_df is not None:
            download_btn(
                "📥 What-If Scenarios (CSV)",
                whatif_df.to_csv(index=False),
                "whatif.csv",
                "text/csv",
            )

    bar()
    section("📋", "Generated Report Files")
    for f in sorted(REPORTS_DIR.glob("*")):
        if f.is_file():
            st.markdown(
                f'<div class="insight-box" style="display:flex;justify-content:space-between">'
                f'<span>📄 <strong>{f.name}</strong></span>'
                f'<span style="color:#64748b">{f.stat().st_size / 1024:.1f} KB</span></div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
bar()
st.markdown(
    """
<div style="text-align:center;padding:1.5rem;color:#64748b;font-size:0.85rem">
<strong style="color:#94a3b8">Retail Analytics & AI-Powered Sales Forecasting System</strong><br>
Enterprise Analytics Platform | Python + Streamlit + Scikit-learn + Plotly<br>
Unlox Capstone Project | 73,000 Records | 8 Stores | 2023-2024
</div>
""",
    unsafe_allow_html=True,
)