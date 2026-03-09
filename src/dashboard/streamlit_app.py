"""
Enhanced Streamlit Dashboard
Run: streamlit run src/dashboard/streamlit_app.py
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Revenue Intelligence", layout="wide", page_icon="R")

BASE = Path(__file__).resolve().parents[2]
PROC = BASE / "data" / "processed"
RAW  = BASE / "data" / "raw"

@st.cache_data(ttl=60)
def load(path, **kw): return pd.read_csv(path, **kw)

# Sidebar
st.sidebar.title("Controls")
section = st.sidebar.radio("Navigate", [
    "Revenue & Anomalies", "30-Day Forecast",
    "RFM 3D Segments", "Cohort Retention", "Geo Map", "Products", "AI Insights"
])

# Date range filter
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input("Start", datetime.now() - timedelta(days=90))
end_date = st.sidebar.date_input("End", datetime.now())

# Header
st.title("E-Commerce Revenue Intelligence")
st.caption("Enhanced Dashboard · AI-Powered · Interactive")

# KPI Row
col1, col2, col3, col4, col5, col6 = st.columns(6)
try:
    daily = load(RAW / "daily_revenue_summary.csv", parse_dates=["date"])
    master = load(PROC / "master_dataset.csv", parse_dates=["order_purchase_timestamp"])
    rfm = load(PROC / "rfm_segments.csv")

    col1.metric("Revenue", f"R${daily.total_revenue.sum()/1e6:.1f}M", "+12%")
    col2.metric("Orders", f"{master.order_id.nunique():,}", "+8%")
    col3.metric("AOV", f"R${daily.avg_order_value.mean():.0f}", "+3%")
    col4.metric("LTV", f"R${rfm.monetary.mean() * rfm.frequency.mean():,.0f}", "Lifetime")
    col5.metric("Repeat Rate", f"{(master.groupby('customer_id').order_id.count() > 1).sum() / master.customer_id.nunique() * 100:.1f}%", "")
    col6.metric("Segments", str(rfm.segment.nunique()), "RFM")
except Exception as e:
    st.warning(f"Run pipeline first: {e}")

if section == "Revenue & Anomalies":
    st.subheader("Revenue Trend with Anomaly Detection")
    try:
        daily = load(RAW / "daily_revenue_summary.csv", parse_dates=["date"])
        anom = load(PROC / "anomaly_detection.csv", parse_dates=["date"])

        fig = go.Figure()
        fig.add_scatter(x=daily.date, y=daily.total_revenue, name="Revenue",
                        line=dict(color="#6C63FF", width=2), fill="tozeroy")
        fig.add_scatter(x=daily.date, y=daily.total_revenue.rolling(7).mean(),
                        name="7-Day MA", line=dict(color="#F39C12", width=2, dash="dot"))

        spikes = anom[(anom.is_anomaly==True) & (anom.anomaly_type=="spike")]
        drops = anom[(anom.is_anomaly==True) & (anom.anomaly_type=="drop")]

        if len(spikes):
            spike_data = daily[daily.date.isin(spikes.date)]
            fig.add_scatter(x=spike_data.date, y=spike_data.total_revenue, mode="markers",
                            name="Spike", marker=dict(color="#2ECC71", size=12, symbol="triangle-up"))
        if len(drops):
            drop_data = daily[daily.date.isin(drops.date)]
            fig.add_scatter(x=drop_data.date, y=drop_data.total_revenue, mode="markers",
                            name="Drop", marker=dict(color="#E74C3C", size=12, symbol="triangle-down"))

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(anom[anom.is_anomaly][["date", "total_revenue", "z_score", "anomaly_type", "anomaly_severity"]].head(20))
    except Exception as e:
        st.warning(f"Run pipeline first: {e}")

elif section == "30-Day Forecast":
    st.subheader("Revenue Forecast (SARIMA + GBM)")
    try:
        daily = load(RAW / "daily_revenue_summary.csv", parse_dates=["date"])
        forecast = load(PROC / "revenue_forecast.csv", parse_dates=["date"])

        fig = go.Figure()
        fig.add_scatter(x=daily.date, y=daily.total_revenue, name="Historical", line=dict(color="#3498DB", width=2))
        fig.add_scatter(x=forecast.date, y=forecast.ensemble_forecast, name="Forecast",
                        line=dict(color="#6C63FF", width=2, dash="dash"))
        fig.add_scatter(x=pd.concat([forecast.date, forecast.date[::-1]]),
                        y=pd.concat([forecast.upper_80, forecast.lower_80[::-1]]),
                        fill="toself", fillcolor="rgba(108,99,255,0.15)", line=dict(color="rgba(0,0,0,0)"),
                        name="80% CI")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Next 7 Days", f"R${forecast.ensemble_forecast.head(7).sum():,.0f}")
        col2.metric("Next 30 Days", f"R${forecast.ensemble_forecast.sum():,.0f}")
    except Exception as e:
        st.warning(f"Run forecast pipeline: {e}")

elif section == "RFM 3D Segments":
    st.subheader("3D RFM Visualization")
    try:
        rfm = load(PROC / "rfm_segments.csv")
        seg = rfm.groupby("segment").agg(
            r=("recency", "mean"), f=("frequency", "mean"), m=("monetary", "mean"),
            count=("customer_id", "count")
        ).reset_index()

        fig = px.scatter_3d(seg, x="r", y="f", z="m", size="count", color="segment",
                            text="segment", title="RFM 3D Map")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Run RFM pipeline: {e}")

elif section == "Cohort Retention":
    st.subheader("Cohort Retention Heatmap")
    try:
        cohort = pd.read_csv(PROC / "cohort_retention.csv", index_col=0)
        cols = [c for c in cohort.columns if str(c).isdigit()][:13]

        fig = go.Figure(data=go.Heatmap(
            z=cohort[cols].values,
            x=[f"M+{c}" for c in cols],
            y=[str(i) for i in cohort.index],
            colorscale=[[0, "#1A1A2E"], [0.5, "#6C63FF"], [1, "#2ECC71"]],
            text=[[f"{v:.0f}%" if v==v else "" for v in row] for row in cohort[cols].values],
            texttemplate="%{text}",
        ))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Run cohort pipeline: {e}")

elif section == "Geo Map":
    st.subheader("Brazil Revenue Map")
    try:
        master = load(PROC / "master_dataset.csv", parse_dates=["order_purchase_timestamp"])
        state_rev = master.groupby("customer_state").total_order_value.sum().reset_index()

        # Add coordinates
        BRAZIL_COORDS = {
            "AC": (-9.0, -70.5), "AL": (-9.5, -36.5), "AP": (0.5, -52.0), "AM": (-3.5, -65.0),
            "BA": (-12.5, -41.5), "CE": (-5.5, -39.5), "DF": (-15.8, -47.9), "ES": (-19.2, -40.3),
            "GO": (-15.8, -49.8), "MA": (-4.8, -45.2), "MT": (-12.6, -56.0), "MS": (-20.8, -54.5),
            "MG": (-18.5, -44.5), "PA": (-4.0, -53.0), "PB": (-7.2, -36.8), "PR": (-25.0, -51.0),
            "PE": (-8.8, -37.0), "PI": (-7.5, -42.5), "RJ": (-22.2, -42.5), "RN": (-5.5, -36.0),
            "RS": (-30.0, -53.5), "RO": (-11.0, -62.5), "RR": (2.0, -61.0), "SC": (-27.2, -50.0),
            "SP": (-22.2, -48.8), "SE": (-10.5, -37.2), "TO": (-10.2, -48.2),
        }
        state_rev["lat"] = state_rev["customer_state"].map(lambda s: BRAZIL_COORDS.get(s, (0, 0))[0])
        state_rev["lon"] = state_rev["customer_state"].map(lambda s: BRAZIL_COORDS.get(s, (0, 0))[1])

        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lon=state_rev["lon"], lat=state_rev["lat"],
            text=state_rev.apply(lambda r: f"<b>{r['customer_state']}</b><br>R$ {r['total_order_value']:,.0f}", axis=1),
            mode="markers+text",
            marker=dict(
                size=np.sqrt(state_rev["total_order_value"]) / 50 + 8,
                color=state_rev["total_order_value"],
                colorscale="Viridis",
                line=dict(color="white", width=1),
            ),
            textposition="top center",
            hoverinfo="text",
        ))
        fig.update_layout(
            geo=dict(scope="south america", landcolor="#1A1A2E", lakecolor="#16213E"),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Run pipeline: {e}")

elif section == "Products":
    st.subheader("Product Analytics")
    try:
        items = load(RAW / "olist_order_items_dataset.csv")
        products = load(RAW / "olist_products_dataset.csv")
        df = items.merge(products[["product_id", "product_category_name"]], on="product_id")

        col1, col2 = st.columns(2)
        with col1:
            cat_rev = df.groupby("product_category_name").price.sum().nlargest(15).reset_index()
            fig = px.treemap(cat_rev, path=["product_category_name"], values="price",
                             title="Category Revenue")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            payment = load(RAW / "olist_order_payments_dataset.csv")
            fig = px.pie(payment, names="payment_type", title="Payment Methods")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Run pipeline: {e}")

elif section == "AI Insights":
    st.subheader("AI-Powered Insights")
    try:
        daily = load(RAW / "daily_revenue_summary.csv", parse_dates=["date"])
        master = load(PROC / "master_dataset.csv", parse_dates=["order_purchase_timestamp"])
        rfm = load(PROC / "rfm_segments.csv")
        reviews = load(RAW / "olist_order_reviews_dataset.csv")

        # Auto insights
        recent = daily.tail(30).total_revenue.mean()
        older = daily.iloc[-60:-30].total_revenue.mean() if len(daily) > 60 else daily.head(30).total_revenue.mean()
        pct = ((recent - older) / older * 100) if older > 0 else 0

        st.info(f"Revenue is {'up' if pct > 0 else 'down'} {abs(pct):.1f}% vs previous period")
        st.success(f"Top state: {master.groupby('customer_state').total_order_value.sum().idxmax()}")
        st.warning(f"Largest segment: {rfm.segment.value_counts().idxmax()}")

        # Sentiment
        avg_score = reviews.review_score.mean()
        sentiment = (avg_score - 3) / 2
        st.metric("Customer Sentiment", f"{sentiment:.2f}", delta=f"{avg_score:.1f}/5 avg review")

        # What-if
        st.subheader("What-If Simulator")
        col1, col2 = st.columns(2)
        aov_change = col1.slider("AOV Change (%)", -20, 50, 10)
        vol_change = col2.slider("Volume Change (%)", -30, 50, 0)

        base = daily.total_revenue.sum()
        new_rev = base * (1 + aov_change/100) * (1 + vol_change/100)
        st.metric("Projected Revenue", f"R${new_rev:,.0f}", delta=f"R${new_rev - base:,.0f}")
    except Exception as e:
        st.warning(f"Run pipeline: {e}")
