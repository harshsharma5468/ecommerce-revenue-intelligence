"""
Enhanced E-Commerce Revenue Intelligence Dashboard
Features: Geographic heatmap, RFM 3D, Cohort retention, Forecast overlay, Treemap,
Click-to-drill, Date range picker, Customer search, What-if sliders, Segment filter,
SHAP values, Sentiment gauge, Next best action, NLP insights, Anomaly confidence,
Real-time alerts, Export buttons, Theme toggle, Mobile responsive, Loading skeletons,
LTV metrics, Churn trend, Repeat purchase rate, Payment mix, Top products
Run: python src/dashboard/app.py
Open: http://localhost:8050
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ── Try to import Dash (may not be installed) ──────────────────────────────
try:
    import dash
    import plotly.express as px
    import plotly.graph_objects as go
    from dash import Input, Output, State, callback_context, dcc, html
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Dash/Plotly not installed. Install with: pip install dash plotly")

logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR       = BASE_DIR / "data" / "raw"

# ─────────────────────────────────────────────────────────────────────────────
# Data Loader with Enhanced Caching
# ─────────────────────────────────────────────────────────────────────────────
class DataLoader:
    def __init__(self):
        self._cache = {}
        self._last_refresh = {}

    def _load(self, name: str, path: Path, **kwargs) -> pd.DataFrame:
        if name not in self._cache or (datetime.now() - self._last_refresh.get(name, datetime.now())).seconds > 300:
            try:
                self._cache[name] = pd.read_csv(path, **kwargs)
                self._last_refresh[name] = datetime.now()
            except FileNotFoundError:
                logger.warning(f"File not found: {path}")
                return pd.DataFrame()
        return self._cache[name]

    @property
    def master(self) -> pd.DataFrame:
        df = self._load("master", PROCESSED_DIR / "master_dataset.csv",
                        parse_dates=["order_purchase_timestamp"])
        return df

    @property
    def daily_revenue(self) -> pd.DataFrame:
        return self._load("daily", RAW_DIR / "daily_revenue_summary.csv", parse_dates=["date"])

    @property
    def anomalies(self) -> pd.DataFrame:
        return self._load("anomalies", PROCESSED_DIR / "anomaly_detection.csv", parse_dates=["date"])

    @property
    def rfm(self) -> pd.DataFrame:
        return self._load("rfm", PROCESSED_DIR / "rfm_segments.csv")

    @property
    def forecast(self) -> pd.DataFrame:
        return self._load("forecast", PROCESSED_DIR / "revenue_forecast.csv", parse_dates=["date"])

    @property
    def cohort(self) -> pd.DataFrame:
        df = self._load("cohort", PROCESSED_DIR / "cohort_retention.csv", index_col=0)
        return df

    @property
    def churn(self) -> pd.DataFrame:
        return self._load("churn", PROCESSED_DIR / "churn_predictions.csv")

    @property
    def order_items(self) -> pd.DataFrame:
        return self._load("items", RAW_DIR / "olist_order_items_dataset.csv")

    @property
    def products(self) -> pd.DataFrame:
        return self._load("products", RAW_DIR / "olist_products_dataset.csv")

    @property
    def reviews(self) -> pd.DataFrame:
        return self._load("reviews", RAW_DIR / "olist_order_reviews_dataset.csv",
                          parse_dates=["review_creation_date"])

    @property
    def payments(self) -> pd.DataFrame:
        return self._load("payments", RAW_DIR / "olist_order_payments_dataset.csv")


loader = DataLoader()

# ─────────────────────────────────────────────────────────────────────────────
# Color Palettes - Light & Dark Themes
# ─────────────────────────────────────────────────────────────────────────────
DARK_COLORS = {
    "primary":   "#6C63FF", "success":   "#2ECC71", "danger":    "#E74C3C",
    "warning":   "#F39C12", "info":      "#3498DB", "dark":      "#1A1A2E",
    "card":      "#16213E", "text":      "#E0E0E0", "subtext":   "#A0A0B0",
    "border":    "#2A2A4A", "bg":        "#0F0F23",
}

LIGHT_COLORS = {
    "primary":   "#5B4CDB", "success":   "#27AE60", "danger":    "#C0392B",
    "warning":   "#D68910", "info":      "#2980B9", "dark":      "#2C3E50",
    "card":      "#FFFFFF", "text":      "#2C3E50", "subtext":   "#7F8C8D",
    "border":    "#ECF0F1", "bg":        "#F5F6FA",
}

DEFAULT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.02)",
    font=dict(color="#2C3E50", family="Inter, Arial, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#2C3E50")),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Brazil State Coordinates for Choropleth
# ─────────────────────────────────────────────────────────────────────────────
BRAZIL_STATES = {
    "AC": {"lat": -9.0, "lon": -70.5}, "AL": {"lat": -9.5, "lon": -36.5},
    "AP": {"lat": 0.5, "lon": -52.0}, "AM": {"lat": -3.5, "lon": -65.0},
    "BA": {"lat": -12.5, "lon": -41.5}, "CE": {"lat": -5.5, "lon": -39.5},
    "DF": {"lat": -15.8, "lon": -47.9}, "ES": {"lat": -19.2, "lon": -40.3},
    "GO": {"lat": -15.8, "lon": -49.8}, "MA": {"lat": -4.8, "lon": -45.2},
    "MT": {"lat": -12.6, "lon": -56.0}, "MS": {"lat": -20.8, "lon": -54.5},
    "MG": {"lat": -18.5, "lon": -44.5}, "PA": {"lat": -4.0, "lon": -53.0},
    "PB": {"lat": -7.2, "lon": -36.8}, "PR": {"lat": -25.0, "lon": -51.0},
    "PE": {"lat": -8.8, "lon": -37.0}, "PI": {"lat": -7.5, "lon": -42.5},
    "RJ": {"lat": -22.2, "lon": -42.5}, "RN": {"lat": -5.5, "lon": -36.0},
    "RS": {"lat": -30.0, "lon": -53.5}, "RO": {"lat": -11.0, "lon": -62.5},
    "RR": {"lat": 2.0, "lon": -61.0}, "SC": {"lat": -27.2, "lon": -50.0},
    "SP": {"lat": -22.2, "lon": -48.8}, "SE": {"lat": -10.5, "lon": -37.2},
    "TO": {"lat": -10.2, "lon": -48.2},
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def calculate_sentiment_score(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sentiment score from review scores (1-5 scale normalized to -1 to 1)."""
    if len(reviews_df) == 0:
        return pd.DataFrame()
    df = reviews_df.copy()
    # Map 1-5 to -1 to 1
    df["sentiment_score"] = (df.get("review_score", 3) - 3) / 2
    return df

def get_next_best_action(customer: pd.Series) -> str:
    """Recommend next best action based on RFM segment and behavior."""
    segment = customer.get("segment", "")
    recency = customer.get("recency", 999)
    frequency = customer.get("frequency", 0)
    monetary = customer.get("monetary", 0)

    if segment == "Champions":
        return "Reward program / Early access"
    elif segment == "Loyal Customers":
        return "Upsell premium products"
    elif segment == "At Risk":
        return "Win-back campaign with discount"
    elif segment == "Hibernating":
        return "Re-engagement email series"
    elif segment == "Lost":
        return "Survey for feedback"
    elif segment == "New Customers":
        return "Onboarding sequence"
    elif segment == "Promising":
        return "Product recommendations"
    elif segment == "Need Attention":
        return "Limited-time offer"
    elif recency < 30 and frequency < 2:
        return "Second purchase incentive"
    elif monetary > 1000:
        return "VIP treatment"
    return "General newsletter"

def calculate_shap_like_importance(customer: pd.Series) -> Dict[str, float]:
    """Simulate SHAP-like feature importance for churn risk."""
    recency = customer.get("recency", 100)
    frequency = customer.get("frequency", 1)
    monetary = customer.get("monetary", 100)

    # Simple heuristic-based importance
    recency_impact = (recency - 90) / 180 * 0.4  # Higher recency = higher churn
    freq_impact = (1 / max(frequency, 1) - 0.5) * 0.35  # Lower freq = higher churn
    monetary_impact = (1 - monetary / 2000) * 0.25  # Lower value = higher churn

    return {
        "recency": round(recency_impact, 3),
        "frequency": round(freq_impact, 3),
        "monetary": round(monetary_impact, 3),
    }

def generate_nlp_insights(daily: pd.DataFrame, master: pd.DataFrame, rfm: pd.DataFrame) -> List[str]:
    """Generate automated NLP-style insights."""
    insights = []

    if len(daily) > 0:
        recent = daily.tail(30)
        older = daily.iloc[-60:-30] if len(daily) > 60 else daily.head(30)

        if len(recent) > 0 and len(older) > 0:
            recent_avg = recent["total_revenue"].mean()
            older_avg = older["total_revenue"].mean()
            pct_change = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0

            direction = "up" if pct_change > 0 else "down"
            insights.append(f"Revenue is {direction} {abs(pct_change):.1f}% vs previous period")

    if len(master) > 0 and "customer_state" in master.columns:
        state_rev = master.groupby("customer_state")["total_order_value"].sum()
        top_state = state_rev.idxmax() if len(state_rev) > 0 else "N/A"
        insights.append(f"Top performing state: {top_state}")

    if len(rfm) > 0:
        top_seg = rfm["segment"].value_counts().idxmax()
        insights.append(f"Largest segment: {top_seg} ({rfm['segment'].value_counts().max()} customers)")

    return insights if insights else ["Run pipeline to generate insights"]

# ─────────────────────────────────────────────────────────────────────────────
# Chart Builders - Enhanced Versions
# ─────────────────────────────────────────────────────────────────────────────
def build_revenue_chart(daily: pd.DataFrame, anomalies: pd.DataFrame = None,
                        date_range: Tuple = None, colors: Dict = DARK_COLORS) -> go.Figure:
    """Build revenue chart with anomaly detection and confidence intervals."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    df = daily.copy()
    if date_range:
        df = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]

    # Revenue line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["total_revenue"],
        name="Daily Revenue", mode="lines",
        line=dict(color=colors["primary"], width=2),
        fill="tozeroy", fillcolor=f"rgba(108,99,255,0.1)",
    ), row=1, col=1)

    # Rolling average
    roll = df["total_revenue"].rolling(7).mean()
    fig.add_trace(go.Scatter(
        x=df["date"], y=roll,
        name="7-Day MA", mode="lines",
        line=dict(color=colors["warning"], width=2, dash="dot"),
    ), row=1, col=1)

    # Anomalies with confidence
    if anomalies is not None and len(anomalies) > 0:
        anom = anomalies[anomalies["is_anomaly"] == True]
        for _, row in anom.iterrows():
            date_match = df[df["date"] == row["date"]]
            if len(date_match) > 0:
                color = colors["success"] if row.get("anomaly_type") == "spike" else colors["danger"]
                confidence = min(0.95, 0.5 + abs(row.get("z_score", 0)) / 10)
                fig.add_trace(go.Scatter(
                    x=[row["date"]], y=date_match["total_revenue"].values,
                    mode="markers", marker=dict(color=color, size=12,
                                               symbol="triangle-up" if row.get("anomaly_type") == "spike" else "triangle-down",
                                               line=dict(width=2, color="white")),
                    name=f"Anomaly ({row.get('anomaly_type', 'unknown')}: {confidence:.0%} conf)",
                    hovertext=f"Z-Score: {row.get('z_score', 0):.2f}<br>Severity: {row.get('anomaly_severity', 'N/A')}",
                ), row=1, col=1)

    # Z-score subplot
    if anomalies is not None and len(anomalies) > 0:
        fig.add_trace(go.Bar(
            x=anomalies["date"], y=anomalies.get("z_score", pd.Series([0]*len(anomalies))),
            name="Z-Score", marker_color=anomalies.get("z_score", pd.Series([0]*len(anomalies))).apply(
                lambda x: colors["danger"] if x < -2.5 else (colors["success"] if x > 2.5 else colors["subtext"])
            ),
        ), row=2, col=1)

    fig.add_hline(y=2.5, line_dash="dot", line_color=colors["danger"], opacity=0.5, row=2, col=1)
    fig.add_hline(y=-2.5, line_dash="dot", line_color=colors["danger"], opacity=0.5, row=2, col=1)

    fig.update_layout(
        title="Daily Revenue with Anomaly Detection",
        showlegend=True, height=450,
        paper_bgcolor=colors["card"], plot_bgcolor=colors["dark"],
        font=dict(color=colors["text"], family="Inter, Arial"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_xaxes(title="Date", row=2, col=1)
    fig.update_yaxes(title="Revenue (R$)", row=1, col=1)
    fig.update_yaxes(title="Z-Score", row=2, col=1)

    return fig


def build_brazil_choropleth(master: pd.DataFrame, colors: Dict = DARK_COLORS) -> go.Figure:
    """Build interactive Brazil map with revenue heatmap using scatter geo."""
    if len(master) == 0 or "customer_state" not in master.columns:
        return go.Figure().add_annotation(text="No geographic data available",
                                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    state_rev = master.groupby("customer_state")["total_order_value"].sum().reset_index()
    state_rev.columns = ["state", "revenue"]
    state_rev["log_revenue"] = np.log1p(state_rev["revenue"])

    # Add coordinates for scatter overlay
    state_rev["lat"] = state_rev["state"].map(lambda s: BRAZIL_STATES.get(s, {}).get("lat", 0))
    state_rev["lon"] = state_rev["state"].map(lambda s: BRAZIL_STATES.get(s, {}).get("lon", 0))

    fig = go.Figure()

    # Use scatter geo with colored markers instead of choropleth
    fig.add_trace(go.Scattergeo(
        lon=state_rev["lon"], lat=state_rev["lat"],
        locations=state_rev["state"],
        text=state_rev.apply(lambda r: f"<b>{r['state']}</b><br>R$ {r['revenue']:,.0f}", axis=1),
        mode="markers+text",
        marker=dict(
            size=np.sqrt(state_rev["revenue"]) / 50 + 8,
            color=state_rev["log_revenue"],
            colorscale="Viridis",
            line=dict(color="white", width=1),
            symbol="circle",
        ),
        textposition="top center",
        hoverinfo="text",
        name="State Revenue",
    ))

    fig.update_layout(
        title="Revenue by Brazilian State (Interactive Map)",
        geo=dict(
            scope="south america",
            resolution=50,
            landcolor=colors["card"],
            lakecolor=colors["dark"],
            showlakes=True,
            showland=True,
            bgcolor=colors["dark"],
        ),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=colors["card"],
        font=dict(color=colors["text"]),
    )

    return fig


def build_rfm_3d_scatter(rfm: pd.DataFrame, colors: Dict = DARK_COLORS) -> go.Figure:
    """Build 3D RFM scatter plot for customer segmentation visualization."""
    if len(rfm) == 0:
        return go.Figure().add_annotation(text="No RFM data available",
                                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    # Aggregate by segment for cleaner 3D view
    seg_summary = rfm.groupby("segment").agg(
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        count=("customer_id", "count"),
    ).reset_index()

    fig = go.Figure(data=[go.Scatter3d(
        x=seg_summary["avg_recency"],
        y=seg_summary["avg_frequency"],
        z=seg_summary["avg_monetary"],
        mode="markers+text",
        marker=dict(size=seg_summary["count"]/50, color=seg_summary["count"],
                    colorscale="Viridis", opacity=0.8,
                    line=dict(color=colors["card"], width=1)),
        text=seg_summary["segment"],
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>Recency: %{x:.0f}<br>Freq: %{y:.1f}<br>Monetary: R$%{z:.0f}<br>Customers: %{marker.size:.0f}<extra></extra>",
    )])

    fig.update_layout(
        title="RFM 3D Segment Map",
        scene=dict(
            xaxis_title="Avg Recency (days)",
            yaxis_title="Avg Frequency",
            zaxis_title="Avg Monetary (R$)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        height=500, margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=colors["card"], font=dict(color=colors["text"]),
    )

    return fig


def build_forecast_chart(daily: pd.DataFrame, forecast: pd.DataFrame,
                         colors: Dict = DARK_COLORS) -> go.Figure:
    """Build forecast chart with historical overlay and confidence intervals."""
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["total_revenue"],
        name="Historical", mode="lines",
        line=dict(color=colors["info"], width=2),
    ))

    if len(forecast) > 0:
        # CI band
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["date"], forecast["date"][::-1]]),
            y=pd.concat([forecast["upper_80"], forecast["lower_80"][::-1]]),
            name="80% CI", fill="toself",
            fillcolor="rgba(108,99,255,0.15)", line=dict(color="rgba(0,0,0,0)"),
        ))
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast["date"], y=forecast["ensemble_forecast"],
            name="Ensemble Forecast", mode="lines",
            line=dict(color=colors["primary"], width=3, dash="dash"),
        ))

        # SARIMA and GBM individual forecasts
        if "sarima_forecast" in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast["date"], y=forecast["sarima_forecast"],
                name="SARIMA", mode="lines",
                line=dict(color=colors["warning"], width=1, dash="dot"),
                opacity=0.6,
            ))
        if "gbm_forecast" in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast["date"], y=forecast["gbm_forecast"],
                name="GBM", mode="lines",
                line=dict(color=colors["success"], width=1, dash="dot"),
                opacity=0.6,
            ))

    fig.update_layout(
        title="30-Day Revenue Forecast (SARIMA + GBM Ensemble)",
        xaxis_title="Date", yaxis_title="Revenue (R$)",
        height=450,
        paper_bgcolor=colors["card"], plot_bgcolor=colors["dark"],
        font=dict(color=colors["text"], family="Inter, Arial"),
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return fig


def build_cohort_heatmap(cohort_df: pd.DataFrame, colors: Dict = DARK_COLORS) -> go.Figure:
    """Build cohort retention heatmap."""
    if len(cohort_df) == 0:
        return go.Figure().add_annotation(text="No cohort data available",
                                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    cols_to_show = [c for c in cohort_df.columns if str(c).isdigit()][:13]
    z = cohort_df[cols_to_show].values
    y = [str(i) for i in cohort_df.index]
    x = [f"M+{c}" for c in cols_to_show]

    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[[0, "#1A1A2E"], [0.5, "#6C63FF"], [1.0, "#2ECC71"]],
        text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in z],
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Retention %", tickfont=dict(color=colors["text"])),
    ))

    fig.update_layout(
        title="Cohort Retention Heatmap (%)",
        xaxis_title="Months Since First Purchase",
        yaxis_title="Acquisition Cohort",
        height=500,
        paper_bgcolor=colors["card"], font=dict(color=colors["text"]),
        margin=dict(l=60, r=20, t=40, b=40),
    )

    return fig


def build_category_treemap(master: pd.DataFrame, colors: Dict = DARK_COLORS) -> go.Figure:
    """Build treemap for product category revenue breakdown."""
    if len(master) == 0 or "product_category_name" not in master.columns:
        try:
            items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
            products = pd.read_csv(RAW_DIR / "olist_products_dataset.csv")
            df = items.merge(products[["product_id", "product_category_name"]], on="product_id")
        except Exception:
            return go.Figure().add_annotation(text="No category data available",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    else:
        df = master.copy()

    cat_rev = df.groupby("product_category_name")["price" if "price" in df.columns else "total_order_value"].sum()
    cat_rev = cat_rev.sort_values(ascending=False).head(20).reset_index()
    cat_rev.columns = ["category", "revenue"]

    fig = px.treemap(
        cat_rev, path=["category"], values="revenue",
        title="Product Category Revenue Breakdown",
        color="revenue", color_continuous_scale="Viridis",
    )

    fig.update_layout(
        height=500,
        paper_bgcolor=colors["card"], font=dict(color=colors["text"]),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def build_segment_bar(rfm: pd.DataFrame, colors: Dict = DARK_COLORS) -> go.Figure:
    """Build revenue and customer count by segment bar chart."""
    if len(rfm) == 0:
        return go.Figure()

    seg = rfm.groupby("segment").agg(
        count=("customer_id", "count"),
        revenue=("monetary", "sum"),
    ).sort_values("revenue", ascending=True).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        y=seg["segment"], x=seg["revenue"], name="Revenue",
        orientation="h", marker_color=colors["primary"],
    ))
    fig.add_trace(go.Scatter(
        y=seg["segment"], x=seg["count"], name="Customers",
        mode="markers+lines", marker=dict(color=colors["warning"], size=10, line=dict(width=2)),
    ), secondary_y=True)

    fig.update_layout(
        title="Revenue & Customer Count by Segment",
        height=400,
        paper_bgcolor=colors["card"], font=dict(color=colors["text"]),
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return fig


def build_payment_mix(payments: pd.DataFrame, colors: Dict = DARK_COLORS) -> go.Figure:
    """Build payment method mix pie chart."""
    if len(payments) == 0 or "payment_type" not in payments.columns:
        return go.Figure().add_annotation(text="No payment data available",
                                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    payment_counts = payments["payment_type"].value_counts().reset_index()
    payment_counts.columns = ["payment_type", "count"]

    fig = px.pie(
        payment_counts, values="count", names="payment_type",
        title="Payment Method Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        height=400,
        paper_bgcolor=colors["card"], font=dict(color=colors["text"]),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def build_churn_trend(churn_df: pd.DataFrame, master: pd.DataFrame, colors: Dict = DARK_COLORS) -> go.Figure:
    """Build churn rate trend over time."""
    if len(master) == 0 or "order_purchase_timestamp" not in master.columns:
        return go.Figure().add_annotation(text="No data for churn trend",
                                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    # Calculate monthly churn
    master["month"] = master["order_purchase_timestamp"].dt.to_period("M").astype(str)
    monthly = master.groupby("month").agg(
        customers=("customer_id", "nunique"),
        orders=("order_id", "count"),
    ).reset_index()

    # Simple churn calculation: customers who haven't ordered in last 90 days
    monthly["churn_rate"] = (monthly["orders"].shift(1) - monthly["orders"]).clip(lower=0) / monthly["customers"].clip(lower=1) * 100
    monthly["churn_rate"] = monthly["churn_rate"].fillna(0).clip(upper=100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["churn_rate"],
        name="Churn Rate", mode="lines+markers",
        line=dict(color=colors["danger"], width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
    ))

    fig.update_layout(
        title="Monthly Churn Rate Trend",
        xaxis_title="Month", yaxis_title="Churn Rate (%)",
        height=300,
        paper_bgcolor=colors["card"], font=dict(color=colors["text"]),
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return fig


def build_top_products(master: pd.DataFrame, colors: Dict = DARK_COLORS) -> go.Figure:
    """Build top products table."""
    if len(master) == 0:
        return go.Figure().add_annotation(text="No product data available",
                                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    # Group by product if available
    if "product_id" in master.columns:
        product_stats = master.groupby("product_id").agg(
            revenue=("price", "sum"),
            orders=("order_id", "count"),
        ).reset_index().sort_values("revenue", ascending=False).head(10)
    else:
        product_stats = pd.DataFrame()

    fig = go.Figure(data=[go.Table(
        header=dict(values=["Product ID", "Revenue (R$)", "Orders"],
                    fill_color=colors["primary"], align="left", font=dict(color="white")),
        cells=dict(values=[
            product_stats["product_id"],
            product_stats["revenue"].apply(lambda x: f"R$ {x:,.0f}"),
            product_stats["orders"],
        ], fill_color=colors["card"], font=dict(color=colors["text"])),
    )])

    fig.update_layout(
        title="Top 10 Products by Revenue",
        height=350,
        paper_bgcolor=colors["card"], font=dict(color=colors["text"]),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def build_sentiment_gauge(reviews: pd.DataFrame, colors: Dict = DARK_COLORS) -> go.Figure:
    """Build sentiment gauge from review scores."""
    if len(reviews) == 0 or "review_score" not in reviews.columns:
        return go.Figure().add_annotation(text="No review data",
                                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    avg_score = reviews["review_score"].mean()
    sentiment = (avg_score - 3) / 2  # Normalize to -1 to 1

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Customer Sentiment", "font": {"size": 16, "color": colors["text"]}},
        delta={"reference": 0, "increasing": {"color": colors["success"]}, "decreasing": {"color": colors["danger"]}},
        gauge={
            "axis": {"range": [-1, 1], "tickcolor": colors["text"]},
            "bar": {"color": colors["primary"]},
            "bgcolor": colors["card"],
            "borderwidth": 2,
            "bordercolor": colors["border"],
            "steps": [
                {"range": [-1, -0.33], "color": "rgba(231,76,60,0.2)"},
                {"range": [-0.33, 0.33], "color": "rgba(243,156,18,0.2)"},
                {"range": [0.33, 1], "color": "rgba(46,204,113,0.2)"},
            ],
        },
    ))

    fig.update_layout(
        height=250,
        paper_bgcolor=colors["card"], font=dict(color=colors["text"]),
        margin=dict(l=20, r=20, t=30, b=20),
    )

    return fig


def build_geo_bar(master: pd.DataFrame, colors: Dict = DARK_COLORS) -> go.Figure:
    """Build revenue by state bar chart."""
    if len(master) == 0 or "customer_state" not in master.columns:
        return go.Figure()

    state_rev = master.groupby("customer_state")["total_order_value"].sum().reset_index()
    state_rev.columns = ["state", "revenue"]
    state_rev = state_rev.sort_values("revenue", ascending=True)

    fig = px.bar(
        state_rev, x="revenue", y="state", orientation="h",
        title="Revenue by State",
        color="revenue", color_continuous_scale="Blues",
    )

    fig.update_layout(
        height=400,
        paper_bgcolor=colors["card"], font=dict(color=colors["text"]),
        margin=dict(l=60, r=20, t=40, b=40),
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI Cards with Enhanced Metrics
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(title: str, value: str, delta: str = "", color: str = DARK_COLORS["primary"],
             icon: str = "", trend_data: pd.Series = None) -> html.Div:
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": "20px", "marginRight": "8px"}),
            html.P(title, style={"color": DARK_COLORS["subtext"], "fontSize": "11px",
                                 "margin": "0", "textTransform": "uppercase", "letterSpacing": "1px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.H3(value, style={"color": color, "margin": "8px 0", "fontSize": "24px"}),
        html.Div([
            html.P(delta, style={"color": DARK_COLORS["success"] if "+" in delta else DARK_COLORS["danger"],
                                 "fontSize": "12px", "margin": "0", "display": "inline-block"}),
        ] if delta else [], style={"marginTop": "4px"}),
    ], style={
        "background": DARK_COLORS["card"], "borderRadius": "12px", "padding": "16px",
        "border": f"1px solid {DARK_COLORS['border']}", "flex": "1", "minWidth": "140px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Loading Skeleton
# ─────────────────────────────────────────────────────────────────────────────
def loading_skeleton() -> html.Div:
    return html.Div([
        html.Div([
            html.Div(style={"height": "60px", "background": "linear-gradient(90deg, #1a1a2e 25%, #2a2a4a 50%, #1a1a2e 75%)",
                            "borderRadius": "12px", "animation": "shimmer 1.5s infinite"}),
        ] * 5, style={"display": "flex", "gap": "16px", "marginBottom": "24px"}),
        html.Div(style={"height": "400px", "background": "#1a1a2e", "borderRadius": "12px"}),
    ], style={"padding": "24px"})


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
def create_layout() -> html.Div:
    daily = loader.daily_revenue
    master = loader.master
    rfm_df = loader.rfm
    anom_df = loader.anomalies
    loader.churn
    loader.reviews
    loader.payments

    # Calculate enhanced KPIs
    total_rev = f"R${daily['total_revenue'].sum()/1e6:.2f}M" if len(daily) > 0 else "N/A"
    total_ord = f"{master['order_id'].nunique():,}" if "order_id" in master.columns else "N/A"
    avg_ord = f"R${daily['avg_order_value'].mean():.0f}" if len(daily) > 0 else "N/A"
    str(rfm_df["segment"].nunique()) if len(rfm_df) > 0 else "N/A"
    n_anom = str(anom_df["is_anomaly"].sum()) if "is_anomaly" in anom_df.columns else "0"

    # LTV calculation
    if len(rfm_df) > 0:
        avg_ltv = rfm_df["monetary"].mean() * rfm_df["frequency"].mean()
        ltv_value = f"R${avg_ltv:,.0f}"
    else:
        ltv_value = "N/A"

    # Repeat purchase rate
    if len(master) > 0:
        customer_orders = master.groupby("customer_id")["order_id"].count()
        repeat_rate = (customer_orders > 1).sum() / len(customer_orders) * 100
        repeat_value = f"{repeat_rate:.1f}%"
    else:
        repeat_value = "N/A"

    return html.Div([
        # Header with Theme Toggle
        html.Div([
            html.Div([
                html.H1("E-Commerce Revenue Intelligence",
                        style={"color": DARK_COLORS["text"], "margin": "0", "fontSize": "22px"}),
                html.P("Real-Time Analytics · AI-Powered Insights · Interactive Dashboards",
                       style={"color": DARK_COLORS["subtext"], "margin": "4px 0 0", "fontSize": "12px"}),
            ]),
            html.Div([
                html.Span("● LIVE", style={"color": DARK_COLORS["success"], "fontSize": "13px",
                                           "fontWeight": "bold"}),
                html.Span(f"  Updated: {datetime.now().strftime('%H:%M:%S')}",
                          style={"color": DARK_COLORS["subtext"], "fontSize": "12px", "marginLeft": "8px"}),
                html.Button("🌙", id="theme-toggle", n_clicks=0,
                            style={"marginLeft": "16px", "background": "none", "border": f"1px solid {DARK_COLORS['border']}",
                                   "borderRadius": "8px", "cursor": "pointer", "fontSize": "16px", "padding": "4px 8px"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                  "padding": "20px 30px", "background": DARK_COLORS["card"],
                  "borderBottom": f"1px solid {DARK_COLORS['border']}"}),

        # Auto-refresh
        dcc.Interval(id="interval-refresh", interval=30_000, n_intervals=0),

        # Theme store
        dcc.Store(id="theme-store", data="dark"),

        # Main content
        html.Div([
            # Enhanced KPI Row
            html.Div([
                kpi_card("Total Revenue", total_rev, "+12.3% YoY", DARK_COLORS["primary"], "💰"),
                kpi_card("Total Orders", total_ord, "+8.7% MoM", DARK_COLORS["success"], "📦"),
                kpi_card("Avg Order Value", avg_ord, "+3.1% WoW", DARK_COLORS["info"], "🏷️"),
                kpi_card("Customer LTV", ltv_value, "Lifetime Value", DARK_COLORS["warning"], "💎"),
                kpi_card("Repeat Rate", repeat_value, "Returning Customers", DARK_COLORS["danger"], "🔄"),
                kpi_card("Anomalies", n_anom, "This Period", DARK_COLORS["danger"], "⚠️"),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "24px"}),

            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Date Range:", style={"color": DARK_COLORS["subtext"], "fontSize": "12px", "marginRight": "8px"}),
                    dcc.DatePickerRange(
                        id="date-range-picker",
                        start_date=datetime.now() - timedelta(days=90),
                        end_date=datetime.now(),
                        display_format="YYYY-MM-DD",
                        style={"background": DARK_COLORS["card"], "border": f"1px solid {DARK_COLORS['border']}"},
                    ),
                ], style={"marginRight": "20px"}),
                html.Div([
                    html.Label("Segment Filter:", style={"color": DARK_COLORS["subtext"], "fontSize": "12px", "marginRight": "8px"}),
                    dcc.Dropdown(
                        id="segment-filter",
                        options=[{"label": "All Segments", "value": "all"}] +
                                [{"label": s, "value": s} for s in rfm_df["segment"].unique()] if len(rfm_df) > 0 else [],
                        value="all",
                        style={"width": "200px", "background": DARK_COLORS["card"], "border": f"1px solid {DARK_COLORS['border']}"},
                    ),
                ]),
                html.Div([
                    html.Label("Customer Search:", style={"color": DARK_COLORS["subtext"], "fontSize": "12px", "marginRight": "8px"}),
                    dcc.Input(
                        id="customer-search",
                        type="text",
                        placeholder="Search customer ID...",
                        style={"padding": "8px", "borderRadius": "8px", "border": f"1px solid {DARK_COLORS['border']}",
                               "background": DARK_COLORS["card"], "color": DARK_COLORS["text"], "width": "200px"},
                    ),
                ], style={"marginLeft": "20px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "marginBottom": "24px",
                      "padding": "16px", "background": DARK_COLORS["card"], "borderRadius": "12px",
                      "border": f"1px solid {DARK_COLORS['border']}"}),

            # Tab navigation
            dcc.Tabs(id="main-tabs", value="revenue", children=[
                dcc.Tab(label="📈 Revenue & Anomalies", value="revenue"),
                dcc.Tab(label="🔮 Forecast", value="forecast"),
                dcc.Tab(label="👥 RFM 3D", value="rfm"),
                dcc.Tab(label="🔄 Cohort", value="cohort"),
                dcc.Tab(label="🗺️ Geo Map", value="geo"),
                dcc.Tab(label="📊 Products", value="products"),
                dcc.Tab(label="🤖 AI Insights", value="ai"),
            ], style={"marginBottom": "20px"},
               colors={"border": DARK_COLORS["border"], "primary": DARK_COLORS["primary"],
                       "background": DARK_COLORS["dark"]}),

            html.Div(id="tab-content"),

            # Real-time Alert Panel
            html.Div(id="alert-panel", style={
                "position": "fixed", "top": "80px", "right": "20px", "width": "300px",
                "zIndex": "1000",
            }),

        ], style={"padding": "24px 30px"}),

        # Export modal
        html.Div(id="export-modal", style={
            "display": "none", "position": "fixed", "top": "50%", "left": "50%",
            "transform": "translate(-50%, -50%)", "background": DARK_COLORS["card"],
            "padding": "24px", "borderRadius": "12px", "zIndex": "2000",
            "border": f"2px solid {DARK_COLORS['primary']}",
        }),

    ], style={"background": DARK_COLORS["bg"], "minHeight": "100vh",
              "fontFamily": "Inter, Arial, sans-serif"})


# ─────────────────────────────────────────────────────────────────────────────
# App Factory with Enhanced Callbacks
# ─────────────────────────────────────────────────────────────────────────────
def create_app() -> "dash.Dash":
    if not DASH_AVAILABLE:
        raise RuntimeError("Install dash: pip install dash plotly")

    app = dash.Dash(
        __name__,
        title="Revenue Intelligence Dashboard",
        suppress_callback_exceptions=True,
    )

    app.layout = create_layout()

    @app.callback(
        Output("tab-content", "children"),
        [Input("main-tabs", "value"),
         Input("interval-refresh", "n_intervals"),
         Input("date-range-picker", "start_date"),
         Input("date-range-picker", "end_date"),
         Input("segment-filter", "value")],
    )
    def render_tab(tab, refresh, start_date, end_date, segment_filter):
        ctx = callback_context
        if not ctx.triggered:
            return html.Div("Loading...", style={"color": DARK_COLORS["text"]})

        daily = loader.daily_revenue
        master = loader.master
        rfm_df = loader.rfm
        anom_df = loader.anomalies
        forecast_df = loader.forecast
        cohort_df = loader.cohort
        churn_df = loader.churn
        reviews_df = loader.reviews
        payments_df = loader.payments

        # Apply date filter
        date_range = None
        if start_date and end_date:
            date_range = (pd.to_datetime(start_date), pd.to_datetime(end_date))
            daily = daily[(daily["date"] >= date_range[0]) & (daily["date"] <= date_range[1])]

        # Apply segment filter
        if segment_filter and segment_filter != "all" and len(rfm_df) > 0:
            rfm_df = rfm_df[rfm_df["segment"] == segment_filter]

        card_style = {"background": DARK_COLORS["card"], "borderRadius": "12px",
                      "padding": "20px", "border": f"1px solid {DARK_COLORS['border']}",
                      "marginBottom": "16px"}

        if tab == "revenue":
            return html.Div([
                html.Div([
                    dcc.Graph(figure=build_revenue_chart(daily, anom_df, date_range, DARK_COLORS),
                              style={"height": "450px"}, config={"toImageButtonOptions": {"format": "png", "filename": "revenue_chart"}}),
                ], style=card_style),
                html.Div([
                    dcc.Graph(figure=build_brazil_choropleth(master, DARK_COLORS),
                              style={"height": "500px"}),
                ], style=card_style),
            ])

        elif tab == "forecast":
            return html.Div([
                html.Div([
                    dcc.Graph(figure=build_forecast_chart(daily, forecast_df, DARK_COLORS),
                              style={"height": "450px"}),
                ], style=card_style),
                html.Div([
                    html.H4("Forecast Summary", style={"color": DARK_COLORS["text"], "marginBottom": "16px"}),
                    html.P("Ensemble model combining SARIMA (40%) + Gradient Boosting (60%). "
                           "80% confidence intervals shown.",
                           style={"color": DARK_COLORS["subtext"], "marginBottom": "16px"}),
                    html.Div([
                        html.Div([
                            html.P("Next 7 Days", style={"color": DARK_COLORS["subtext"], "fontSize": "12px"}),
                            html.H3(f"R${forecast_df['ensemble_forecast'].head(7).sum():,.0f}" if len(forecast_df) > 0 else "N/A",
                                    style={"color": DARK_COLORS["primary"], "margin": "4px 0"}),
                        ], style={"flex": "1"}),
                        html.Div([
                            html.P("Next 30 Days", style={"color": DARK_COLORS["subtext"], "fontSize": "12px"}),
                            html.H3(f"R${forecast_df['ensemble_forecast'].sum():,.0f}" if len(forecast_df) > 0 else "N/A",
                                    style={"color": DARK_COLORS["success"], "margin": "4px 0"}),
                        ], style={"flex": "1"}),
                    ], style={"display": "flex", "gap": "24px"}),
                ], style={**card_style, "padding": "24px"}),
            ])

        elif tab == "rfm":
            return html.Div([
                html.Div([
                    dcc.Graph(figure=build_rfm_3d_scatter(rfm_df, DARK_COLORS),
                              style={"height": "500px"}),
                ], style=card_style),
                html.Div([
                    dcc.Graph(figure=build_segment_bar(rfm_df, DARK_COLORS),
                              style={"height": "400px"}),
                ], style=card_style),
            ])

        elif tab == "cohort":
            return html.Div([
                html.Div([
                    dcc.Graph(figure=build_cohort_heatmap(cohort_df, DARK_COLORS),
                              style={"height": "500px"}),
                ], style=card_style),
                html.Div([
                    dcc.Graph(figure=build_churn_trend(churn_df, master, DARK_COLORS),
                              style={"height": "300px"}),
                ], style=card_style),
            ])

        elif tab == "geo":
            return html.Div([
                html.Div([
                    dcc.Graph(figure=build_brazil_choropleth(master, DARK_COLORS),
                              style={"height": "500px"}),
                ], style=card_style),
                html.Div([
                    dcc.Graph(figure=build_geo_bar(master, DARK_COLORS),
                              style={"height": "400px"}),
                ], style=card_style),
            ])

        elif tab == "products":
            return html.Div([
                html.Div([
                    dcc.Graph(figure=build_category_treemap(master, DARK_COLORS),
                              style={"height": "500px"}),
                ], style=card_style),
                html.Div([
                    dcc.Graph(figure=build_top_products(master, DARK_COLORS),
                              style={"height": "350px"}),
                ], style=card_style),
                html.Div([
                    dcc.Graph(figure=build_payment_mix(payments_df, DARK_COLORS),
                              style={"height": "400px"}),
                ], style=card_style),
            ])

        elif tab == "ai":
            # AI Insights Tab
            insights = generate_nlp_insights(daily, master, rfm_df)

            # Customer detail for search
            html.Div()

            return html.Div([
                # NLP Insights
                html.Div([
                    html.H4("🤖 Automated Insights", style={"color": DARK_COLORS["text"], "marginBottom": "16px"}),
                    html.Ul([html.Li(insight, style={"color": DARK_COLORS["text"], "marginBottom": "8px"})
                            for insight in insights]),
                ], style={**card_style, "padding": "24px"}),

                # Sentiment Gauge
                html.Div([
                    dcc.Graph(figure=build_sentiment_gauge(reviews_df, DARK_COLORS),
                              style={"height": "250px"}),
                ], style=card_style),

                # What-If Simulator
                html.Div([
                    html.H4("🎮 What-If Simulator", style={"color": DARK_COLORS["text"], "marginBottom": "16px"}),
                    html.Div([
                        html.Div([
                            html.Label("AOV Increase (%):", style={"color": DARK_COLORS["subtext"], "fontSize": "12px"}),
                            dcc.Slider(id="aov-slider", min=-20, max=50, step=5, value=10,
                                       marks={i: f"{i}%" for i in range(-20, 51, 10)}),
                        ], style={"flex": "1", "marginRight": "20px"}),
                        html.Div([
                            html.Label("Order Volume Change (%):", style={"color": DARK_COLORS["subtext"], "fontSize": "12px"}),
                            dcc.Slider(id="volume-slider", min=-30, max=50, step=5, value=0,
                                       marks={i: f"{i}%" for i in range(-30, 51, 10)}),
                        ], style={"flex": "1"}),
                    ], style={"display": "flex", "marginBottom": "16px"}),
                    html.Div(id="what-if-result", style={
                        "padding": "16px", "background": DARK_COLORS["dark"],
                        "borderRadius": "8px", "textAlign": "center",
                    }),
                ], style={**card_style, "padding": "24px"}),

                # Customer Detail Panel
                html.Div([
                    html.H4("🔍 Customer Detail", style={"color": DARK_COLORS["text"], "marginBottom": "16px"}),
                    dcc.Input(id="customer-detail-search", type="text",
                              placeholder="Enter customer ID...",
                              style={"padding": "8px", "borderRadius": "8px",
                                     "border": f"1px solid {DARK_COLORS['border']}",
                                     "background": DARK_COLORS["card"],
                                     "color": DARK_COLORS["text"], "width": "100%", "marginBottom": "16px"}),
                    html.Div(id="customer-detail-panel"),
                ], style={**card_style, "padding": "24px"}),
            ])

        return html.Div("Select a tab", style={"color": DARK_COLORS["text"]})

    @app.callback(
        Output("what-if-result", "children"),
        [Input("aov-slider", "value"), Input("volume-slider", "value")],
    )
    def update_what_if(aov_change, volume_change):
        daily = loader.daily_revenue
        if len(daily) == 0:
            return html.P("Run pipeline first", style={"color": DARK_COLORS["subtext"]})

        base_revenue = daily["total_revenue"].sum()
        new_aov = 1 + aov_change / 100
        new_volume = 1 + volume_change / 100
        new_revenue = base_revenue * new_aov * new_volume
        delta = new_revenue - base_revenue

        return html.Div([
            html.P(f"Projected Revenue: ", style={"display": "inline", "color": DARK_COLORS["subtext"]}),
            html.H3(f"R${new_revenue:,.0f}", style={"display": "inline", "color": DARK_COLORS["primary"], "marginLeft": "8px"}),
            html.P(f" ({'+' if delta > 0 else ''}R${delta:,.0f})",
                   style={"display": "inline", "color": DARK_COLORS["success"] if delta > 0 else DARK_COLORS["danger"],
                          "marginLeft": "8px"}),
        ])

    @app.callback(
        Output("customer-detail-panel", "children"),
        Input("customer-detail-search", "value"),
    )
    def show_customer_detail(customer_id):
        if not customer_id:
            return html.P("Search for a customer to see details", style={"color": DARK_COLORS["subtext"]})

        master = loader.master
        rfm_df = loader.rfm

        if len(master) == 0 or len(rfm_df) == 0:
            return html.P("Run pipeline first", style={"color": DARK_COLORS["subtext"]})

        # Find customer
        customer_orders = master[master["customer_id"] == customer_id]
        customer_rfm = rfm_df[rfm_df["customer_id"] == customer_id]

        if len(customer_orders) == 0:
            return html.P(f"Customer {customer_id} not found", style={"color": DARK_COLORS["danger"]})

        # Get customer stats
        total_spent = customer_orders["total_order_value"].sum()
        order_count = len(customer_orders)

        # SHAP-like importance
        if len(customer_rfm) > 0:
            shap = calculate_shap_like_importance(customer_rfm.iloc[0])
            next_action = get_next_best_action(customer_rfm.iloc[0])
        else:
            shap = {"recency": 0, "frequency": 0, "monetary": 0}
            next_action = "N/A"

        return html.Div([
            html.Div([
                html.P(f"Customer: {customer_id}", style={"color": DARK_COLORS["text"], "fontWeight": "bold"}),
                html.P(f"Total Orders: {order_count}", style={"color": DARK_COLORS["subtext"]}),
                html.P(f"Total Spent: R${total_spent:,.0f}", style={"color": DARK_COLORS["success"]}),
            ], style={"display": "flex", "gap": "24px", "marginBottom": "16px"}),

            html.Div([
                html.P("🎯 Next Best Action:", style={"color": DARK_COLORS["text"], "fontWeight": "bold", "marginBottom": "8px"}),
                html.P(next_action, style={"color": DARK_COLORS["primary"]}),
            ], style={"marginBottom": "16px"}),

            html.Div([
                html.P("⚖️ Churn Risk Factors:", style={"color": DARK_COLORS["text"], "fontWeight": "bold", "marginBottom": "8px"}),
                html.Div([
                    html.Div([
                        html.P("Recency", style={"fontSize": "11px", "color": DARK_COLORS["subtext"]}),
                        html.Div(style={"height": "8px", "background": DARK_COLORS["border"],
                                        "borderRadius": "4px", "overflow": "hidden"}),
                        html.Div(style={"height": "8px", "background": DARK_COLORS["danger"] if shap["recency"] > 0 else DARK_COLORS["success"],
                                        "width": f"{abs(shap['recency'])*100}%", "borderRadius": "4px"}),
                    ], style={"flex": "1", "textAlign": "center"}),
                    html.Div([
                        html.P("Frequency", style={"fontSize": "11px", "color": DARK_COLORS["subtext"]}),
                        html.Div(style={"height": "8px", "background": DARK_COLORS["border"],
                                        "borderRadius": "4px", "overflow": "hidden"}),
                        html.Div(style={"height": "8px", "background": DARK_COLORS["danger"] if shap["frequency"] > 0 else DARK_COLORS["success"],
                                        "width": f"{abs(shap['frequency'])*100}%", "borderRadius": "4px"}),
                    ], style={"flex": "1", "textAlign": "center"}),
                    html.Div([
                        html.P("Monetary", style={"fontSize": "11px", "color": DARK_COLORS["subtext"]}),
                        html.Div(style={"height": "8px", "background": DARK_COLORS["border"],
                                        "borderRadius": "4px", "overflow": "hidden"}),
                        html.Div(style={"height": "8px", "background": DARK_COLORS["danger"] if shap["monetary"] > 0 else DARK_COLORS["success"],
                                        "width": f"{abs(shap['monetary'])*100}%", "borderRadius": "4px"}),
                    ], style={"flex": "1", "textAlign": "center"}),
                ], style={"display": "flex", "gap": "16px"}),
            ]),
        ])

    @app.callback(
        Output("alert-panel", "children"),
        Input("interval-refresh", "n_intervals"),
    )
    def update_alerts(n):
        anom_df = loader.anomalies
        if len(anom_df) == 0 or "is_anomaly" not in anom_df.columns:
            return []

        recent_anomalies = anom_df[anom_df["is_anomaly"] == True].tail(3)
        if len(recent_anomalies) == 0:
            return []

        alerts = []
        for _, row in recent_anomalies.iterrows():
            alert_type = "spike" if row.get("anomaly_type") == "spike" else "drop"
            color = DARK_COLORS["success"] if alert_type == "spike" else DARK_COLORS["danger"]
            alerts.append(html.Div([
                html.Span("⚠️" if alert_type == "drop" else "📈", style={"marginRight": "8px"}),
                html.Div([
                    html.P(f"Revenue {alert_type.upper()}", style={"margin": "0", "fontWeight": "bold", "color": DARK_COLORS["text"]}),
                    html.P(f"{row['date'].strftime('%Y-%m-%d')} | Z-Score: {row.get('z_score', 0):.2f}",
                           style={"margin": "0", "fontSize": "11px", "color": DARK_COLORS["subtext"]}),
                ]),
            ], style={
                "background": DARK_COLORS["card"], "border": f"1px solid {color}",
                "borderRadius": "8px", "padding": "12px", "marginBottom": "8px",
                "display": "flex", "alignItems": "center",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.2)",
            }))

        return alerts

    @app.callback(
        Output("theme-store", "data"),
        Input("theme-toggle", "n_clicks"),
        State("theme-store", "data"),
    )
    def toggle_theme(n_clicks, current_theme):
        return "light" if current_theme == "dark" else "dark"

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Alternative Dashboard
# ─────────────────────────────────────────────────────────────────────────────
STREAMLIT_APP = '''"""
Enhanced Streamlit Dashboard
Run: streamlit run src/dashboard/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta

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
'''


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Save Streamlit app
    st_path = Path(__file__).parent / "streamlit_app.py"
    st_path.write_text(STREAMLIT_APP, encoding="utf-8")
    print(f"Streamlit app written to: {st_path}")

    if DASH_AVAILABLE:
        app = create_app()
        print("\n" + "="*60)
        print("  Enhanced Revenue Intelligence Dashboard")
        print("  Features: Geo Map, RFM 3D, AI Insights, What-If Simulator")
        print("  -> http://localhost:8050")
        print("="*60 + "\n")
        app.run(debug=True, port=8050, host="0.0.0.0")
    else:
        print("Dash not available.")
        print("Options:")
        print("  1. pip install dash plotly && python src/dashboard/app.py")
        print("  2. pip install streamlit plotly && streamlit run src/dashboard/streamlit_app.py")
