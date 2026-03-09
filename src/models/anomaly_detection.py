"""
Layer 2: Real-Time Anomaly Detection
Detects unusual spikes/drops in revenue using:
  - Z-Score (statistical baseline)
  - IQR Fence method
  - Isolation Forest (ML-based)
  - Rolling Mean ± 3σ control chart
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"


class AnomalyDetector:
    """
    Multi-method anomaly detector for revenue time series.

    Methods:
        z_score          - Flag points beyond ±N standard deviations
        iqr              - Flag points beyond Q1–1.5×IQR or Q3+1.5×IQR
        isolation_forest - Unsupervised ML anomaly scoring
        control_chart    - Rolling mean ± 3σ SPC chart
        ensemble         - Majority vote across all methods
    """

    def __init__(self, z_threshold: float = 2.5, contamination: float = 0.05,
                 rolling_window: int = 7):
        self.z_threshold     = z_threshold
        self.contamination   = contamination
        self.rolling_window  = rolling_window
        self.iso_forest      = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=200)
        self.scaler          = StandardScaler()
        self.anomalies_df: Optional[pd.DataFrame] = None

    # ─── Public API ───────────────────────────────────────────────────────────

    def detect(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all detection methods on a daily revenue time series.

        Args:
            daily_df: DataFrame with columns [date, total_revenue, total_orders, avg_order_value]
        Returns:
            Annotated DataFrame with anomaly flags and scores
        """
        df = daily_df.copy().sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])

        df = self._rolling_features(df)
        df = self._z_score(df)
        df = self._iqr(df)
        df = self._control_chart(df)
        df = self._isolation_forest(df)
        df = self._ensemble(df)
        df = self._classify_anomaly(df)

        self.anomalies_df = df
        n_anom = df["is_anomaly"].sum()
        logger.info(f"Anomaly detection complete: {n_anom}/{len(df)} days flagged "
                    f"({n_anom/len(df):.1%})")
        return df

    def detect_realtime(self, event: dict, history: list[dict]) -> dict:
        """
        Lightweight real-time anomaly scoring for a single streaming event.

        Args:
            event:   Current event {"revenue": float, "timestamp": str, ...}
            history: Recent events list for context
        Returns:
            event enriched with anomaly_score and is_spike/is_drop flags
        """
        revenues = [e["revenue"] for e in history] if history else [event["revenue"]]
        mu  = np.mean(revenues)
        sig = np.std(revenues) if len(revenues) > 1 else 1.0
        z   = (event["revenue"] - mu) / max(sig, 0.01)
        event["z_score"]    = round(float(z), 3)
        event["is_spike"]   = bool(z > self.z_threshold)
        event["is_drop"]    = bool(z < -self.z_threshold)
        event["is_anomaly"] = event["is_spike"] or event["is_drop"]
        event["anomaly_severity"] = self._severity(abs(z))
        return event

    def get_anomaly_report(self) -> pd.DataFrame:
        """Return only anomalous rows with root cause hints."""
        if self.anomalies_df is None:
            raise RuntimeError("Run detect() first.")
        anom = self.anomalies_df[self.anomalies_df["is_anomaly"]].copy()
        anom = anom.sort_values("anomaly_magnitude", ascending=False)
        return anom[[
            "date", "total_revenue", "revenue_change_pct",
            "z_score", "anomaly_type", "anomaly_severity", "anomaly_magnitude",
            "is_anomaly",
        ]]

    # ─── Detection Methods ────────────────────────────────────────────────────

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self.rolling_window
        df["rolling_mean"]   = df["total_revenue"].rolling(w, min_periods=3).mean()
        df["rolling_std"]    = df["total_revenue"].rolling(w, min_periods=3).std()
        df["rolling_median"] = df["total_revenue"].rolling(w, min_periods=3).median()
        df["revenue_change_pct"] = df["total_revenue"].pct_change() * 100
        df["revenue_lag1"]   = df["total_revenue"].shift(1)
        df["revenue_lag7"]   = df["total_revenue"].shift(7)
        return df

    def _z_score(self, df: pd.DataFrame) -> pd.DataFrame:
        mu  = df["total_revenue"].mean()
        sig = df["total_revenue"].std()
        df["z_score"]          = (df["total_revenue"] - mu) / max(sig, 1.0)
        df["anomaly_zscore"]   = df["z_score"].abs() > self.z_threshold
        return df

    def _iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        Q1, Q3 = df["total_revenue"].quantile(0.25), df["total_revenue"].quantile(0.75)
        IQR    = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df["anomaly_iqr"] = (df["total_revenue"] < lower) | (df["total_revenue"] > upper)
        df["iqr_lower"]   = lower
        df["iqr_upper"]   = upper
        return df

    def _control_chart(self, df: pd.DataFrame) -> pd.DataFrame:
        """SPC Western Electric Rule 1: point beyond 3σ."""
        df["ucl"] = df["rolling_mean"] + 3 * df["rolling_std"]
        df["lcl"] = df["rolling_mean"] - 3 * df["rolling_std"]
        df["anomaly_cc"] = (
            (df["total_revenue"] > df["ucl"]) |
            (df["total_revenue"] < df["lcl"].clip(lower=0))
        )
        return df

    def _isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Isolation Forest on multi-variate features."""
        feat_cols = ["total_revenue", "total_orders", "avg_order_value"]
        # Use only available columns
        available = [c for c in feat_cols if c in df.columns]
        X = df[available].fillna(df[available].mean())
        X_scaled = self.scaler.fit_transform(X)
        preds = self.iso_forest.fit_predict(X_scaled)
        scores = self.iso_forest.decision_function(X_scaled)
        df["anomaly_iforest"] = preds == -1
        df["iforest_score"]   = -scores  # Higher = more anomalous
        return df

    def _ensemble(self, df: pd.DataFrame) -> pd.DataFrame:
        """Majority vote: flag if ≥2 methods agree."""
        cols = ["anomaly_zscore", "anomaly_iqr", "anomaly_cc", "anomaly_iforest"]
        available = [c for c in cols if c in df.columns]
        df["anomaly_votes"] = df[available].sum(axis=1)
        df["is_anomaly"]    = df["anomaly_votes"] >= 2
        return df

    def _classify_anomaly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add type (spike/drop) and severity labels."""
        df["anomaly_type"] = np.where(
            ~df["is_anomaly"], "normal",
            np.where(df["z_score"] > 0, "spike", "drop")
        )
        df["anomaly_magnitude"] = (df["total_revenue"] - df["rolling_mean"]).abs()
        df["anomaly_severity"]  = df["z_score"].abs().apply(self._severity)
        return df

    @staticmethod
    def _severity(z_abs: float) -> str:
        if z_abs < 2.0:   return "none"
        if z_abs < 2.5:   return "low"
        if z_abs < 3.5:   return "medium"
        if z_abs < 5.0:   return "high"
        return "critical"


# ─────────────────────────────────────────────────────────────────────────────
# Root Cause Attribution
# ─────────────────────────────────────────────────────────────────────────────
class RootCauseAttributor:
    """
    Layer 5: When an anomaly is detected, pinpoint which
    SKU/category, region, or channel drove the change.
    """

    def __init__(self):
        self.master_df: Optional[pd.DataFrame] = None

    def load(self, master_df: pd.DataFrame):
        self.master_df = master_df.copy()
        self.master_df["order_purchase_timestamp"] = pd.to_datetime(
            self.master_df["order_purchase_timestamp"])
        self.master_df["date"] = self.master_df["order_purchase_timestamp"].dt.date

    def explain(self, anomaly_date: str, window_days: int = 7) -> dict:
        """
        For a given anomaly date, compare it to the prior window
        and return top contributors by dimension.

        Returns:
            dict with keys: category, state, payment_type — each a ranked DataFrame
        """
        if self.master_df is None:
            raise RuntimeError("Call load() first.")

        anom_dt  = pd.to_datetime(anomaly_date).date()
        prior_start = anom_dt - timedelta(days=window_days)

        anom_df  = self.master_df[self.master_df["date"] == anom_dt]
        prior_df = self.master_df[
            (self.master_df["date"] >= prior_start) &
            (self.master_df["date"] < anom_dt)
        ]

        results = {}
        for dim in ["customer_state", "payment_type"]:
            if dim not in self.master_df.columns:
                continue
            anom_agg  = anom_df.groupby(dim)["total_order_value"].sum().rename("anomaly_revenue")
            prior_agg = prior_df.groupby(dim)["total_order_value"].sum().rename("prior_avg_revenue")
            prior_agg = prior_agg / max(window_days, 1)

            comp = pd.concat([anom_agg, prior_agg], axis=1).fillna(0)
            comp["delta"]    = comp["anomaly_revenue"] - comp["prior_avg_revenue"]
            comp["delta_pct"] = (comp["delta"] / comp["prior_avg_revenue"].replace(0, np.nan) * 100).round(1)
            comp = comp.sort_values("delta", ascending=False).reset_index()
            results[dim] = comp

        # Order items dimension: category
        try:
            items    = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
            products = pd.read_csv(RAW_DIR / "olist_products_dataset.csv")
            orders_date = self.master_df[["order_id","date"]].copy()
            enriched = items.merge(products[["product_id","product_category_name"]], on="product_id")
            enriched = enriched.merge(orders_date, on="order_id")

            anom_cat  = enriched[enriched["date"] == anom_dt].groupby("product_category_name")["price"].sum()
            prior_cat = enriched[
                (enriched["date"] >= prior_start) &
                (enriched["date"] < anom_dt)
            ].groupby("product_category_name")["price"].sum() / max(window_days, 1)

            cat_comp = pd.concat([anom_cat.rename("anomaly_revenue"),
                                   prior_cat.rename("prior_avg_revenue")], axis=1).fillna(0)
            cat_comp["delta"]     = cat_comp["anomaly_revenue"] - cat_comp["prior_avg_revenue"]
            cat_comp["delta_pct"] = (cat_comp["delta"] / cat_comp["prior_avg_revenue"].replace(0,np.nan)*100).round(1)
            cat_comp = cat_comp.sort_values("delta", ascending=False).reset_index()
            results["product_category"] = cat_comp
        except Exception as e:
            logger.warning(f"Category attribution skipped: {e}")

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    daily = pd.read_csv(RAW_DIR / "daily_revenue_summary.csv", parse_dates=["date"])
    detector = AnomalyDetector(z_threshold=2.5, contamination=0.05)
    result   = detector.detect(daily)

    result.to_csv(PROCESSED_DIR / "anomaly_detection.csv", index=False)

    print("\n=== Anomaly Report ===")
    report = detector.get_anomaly_report()
    print(report.head(15).to_string(index=False))

    print(f"\nTotal anomalies: {report.shape[0]}")
    print(f"Spikes: {(report.anomaly_type=='spike').sum()}")
    print(f"Drops:  {(report.anomaly_type=='drop').sum()}")

    # Root Cause
    if len(report) > 0:
        master = pd.read_csv(PROCESSED_DIR / "master_dataset.csv",
                             parse_dates=["order_purchase_timestamp"])
        attributor = RootCauseAttributor()
        attributor.load(master)
        top_anomaly = str(report.iloc[0]["date"])[:10]
        print(f"\n=== Root Cause Attribution: {top_anomaly} ===")
        causes = attributor.explain(top_anomaly)
        for dim, df in causes.items():
            print(f"\n  By {dim}:")
            print(df.head(5).to_string(index=False))
