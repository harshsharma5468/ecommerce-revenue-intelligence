"""
Layer 1: RFM Customer Segmentation
Recency–Frequency–Monetary scoring with K-Means dynamic clustering.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


# ─────────────────────────────────────────────────────────────────────────────
# RFM Calculator
# ─────────────────────────────────────────────────────────────────────────────
class RFMAnalyzer:
    """
    Computes RFM scores, assigns segment labels, and clusters customers.

    Segments (based on quintile scoring):
        Champions       | R=5, F=5, M=5
        Loyal Customers | R≥4, F≥4
        Potential Loyal | R≥3, F≥2
        At Risk         | R≤2, F≥3
        Lost            | R=1, F=1
        New Customers   | R=5, F=1
        Promising       | R=4, F=1
    """

    SEGMENT_MAP = {
        (5, 5): "Champions",
        (5, 4): "Champions",
        (4, 5): "Loyal Customers",
        (4, 4): "Loyal Customers",
        (5, 3): "Potential Loyalist",
        (4, 3): "Potential Loyalist",
        (3, 5): "At Risk",
        (3, 4): "At Risk",
        (2, 5): "Cannot Lose",
        (2, 4): "Cannot Lose",
        (1, 5): "Lost",
        (1, 4): "Lost",
        (5, 1): "New Customers",
        (5, 2): "New Customers",
        (4, 1): "Promising",
        (4, 2): "Promising",
        (3, 3): "Need Attention",
        (3, 2): "Need Attention",
        (3, 1): "About to Sleep",
        (2, 3): "About to Sleep",
        (2, 2): "Hibernating",
        (2, 1): "Hibernating",
        (1, 3): "Hibernating",
        (1, 2): "Hibernating",
        (1, 1): "Lost",
    }

    def __init__(self, reference_date: datetime = None):
        self.reference_date = reference_date or datetime.now()
        self.rfm_df: pd.DataFrame = None
        self.scaler = StandardScaler()
        self.kmeans: KMeans = None
        self.optimal_k: int = 4

    def compute(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RFM table from master dataset.

        Args:
            master_df: Order-level dataset with customer_id, order_purchase_timestamp, total_order_value
        Returns:
            Customer-level RFM DataFrame with scores and segment labels
        """
        df = master_df.copy()
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

        # Only include delivered orders for monetary calculation
        delivered = df[df["order_status"] == "delivered"].copy()

        rfm = delivered.groupby("customer_id").agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("total_order_value", "sum"),
        ).reset_index()

        rfm["recency"] = (self.reference_date - rfm["last_purchase"]).dt.days

        # Quintile scoring (1–5)
        rfm["R_score"] = pd.qcut(rfm["recency"], q=5, labels=[5,4,3,2,1]).astype(int)
        rfm["F_score"] = pd.qcut(rfm["frequency"].clip(upper=rfm["frequency"].quantile(0.99)),
                                  q=5, labels=[1,2,3,4,5], duplicates="drop").astype(int)
        rfm["M_score"] = pd.qcut(rfm["monetary"].clip(upper=rfm["monetary"].quantile(0.99)),
                                  q=5, labels=[1,2,3,4,5], duplicates="drop").astype(int)

        rfm["RFM_score"]  = rfm["R_score"] * 100 + rfm["F_score"] * 10 + rfm["M_score"]
        rfm["RFM_avg"]    = (rfm["R_score"] + rfm["F_score"] + rfm["M_score"]) / 3

        # Segment labels
        rfm["segment"] = rfm.apply(
            lambda row: self.SEGMENT_MAP.get(
                (row["R_score"], row["F_score"]),
                "Need Attention"
            ),
            axis=1,
        )

        self.rfm_df = rfm
        logger.info(f"RFM computed for {len(rfm):,} customers")
        logger.info(f"Segment distribution:\n{rfm['segment'].value_counts()}")
        return rfm

    def cluster(self, n_clusters: int = None) -> pd.DataFrame:
        """Apply K-Means clustering on scaled RFM features."""
        if self.rfm_df is None:
            raise RuntimeError("Run compute() first.")

        features = self.rfm_df[["recency", "frequency", "monetary"]].copy()
        # Log-transform monetary & frequency to reduce skew
        features["frequency"] = np.log1p(features["frequency"])
        features["monetary"]  = np.log1p(features["monetary"])
        features["recency"]   = np.log1p(features["recency"])

        X = self.scaler.fit_transform(features)

        if n_clusters is None:
            n_clusters = self._find_optimal_k(X)

        self.optimal_k = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.rfm_df["cluster"] = self.kmeans.fit_predict(X)

        # Silhouette score
        sil = silhouette_score(X, self.rfm_df["cluster"])
        logger.info(f"K-Means k={n_clusters} | Silhouette: {sil:.3f}")

        # Label clusters by mean monetary value
        cluster_means = self.rfm_df.groupby("cluster")["monetary"].mean().sort_values(ascending=False)
        label_map = {old: f"Cluster_{i+1}_{'High' if i==0 else 'Mid' if i==1 else 'Low'}"
                     for i, old in enumerate(cluster_means.index)}
        self.rfm_df["cluster_label"] = self.rfm_df["cluster"].map(label_map)
        return self.rfm_df

    def _find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 9)) -> int:
        """Elbow + Silhouette method to find optimal K."""
        inertias, silhouettes = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X, labels))

        # Pick k with highest silhouette
        best_k = list(k_range)[int(np.argmax(silhouettes))]
        logger.info(f"Optimal k={best_k} (silhouette={max(silhouettes):.3f})")
        return best_k

    def segment_summary(self) -> pd.DataFrame:
        """Return per-segment aggregated statistics."""
        if self.rfm_df is None:
            raise RuntimeError("Run compute() first.")
        summary = self.rfm_df.groupby("segment").agg(
            customer_count=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
            avg_rfm=("RFM_avg", "mean"),
        ).round(2).reset_index()
        summary["revenue_share"] = (summary["total_revenue"] / summary["total_revenue"].sum() * 100).round(1)
        summary.sort_values("total_revenue", ascending=False, inplace=True)
        return summary

    def save(self, path: Path = None):
        """Save RFM table to processed data directory."""
        if path is None:
            path = PROCESSED_DIR / "rfm_segments.csv"
        self.rfm_df.to_csv(path, index=False)
        logger.info(f"RFM saved to {path}")

    def get_segment_actions(self) -> dict:
        """Return recommended actions per segment."""
        return {
            "Champions":         "Reward them. Be early adopters for new products. Ask for reviews.",
            "Loyal Customers":   "Upsell higher value products. Ask for reviews. Engage them.",
            "Potential Loyalist":"Offer membership/loyalty program. Recommend related products.",
            "At Risk":           "Send personalized emails. Offer discounts. Re-engage them.",
            "Cannot Lose":       "Win them back via renewals. Give freebies. Don't lose them.",
            "New Customers":     "Provide onboarding support. Start building relationship.",
            "Promising":         "Create brand awareness. Offer free trial.",
            "Need Attention":    "Make limited-time offers. Recommend popular products.",
            "About to Sleep":    "Share valuable resources. Recommend popular products. Offer discounts.",
            "Hibernating":       "Offer popular products and discounts. Reconnect with them.",
            "Lost":              "Revive interest with reach-out campaigns. Ignore otherwise.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    master = pd.read_csv(PROCESSED_DIR / "master_dataset.csv", parse_dates=["order_purchase_timestamp"])
    reference = pd.to_datetime(master["order_purchase_timestamp"].max()) + pd.Timedelta(days=1)

    analyzer = RFMAnalyzer(reference_date=reference.to_pydatetime())
    rfm = analyzer.compute(master)
    rfm = analyzer.cluster()
    analyzer.save()

    print("\n=== RFM Segment Summary ===")
    print(analyzer.segment_summary().to_string(index=False))

    print("\n=== Segment Actions ===")
    for seg, action in analyzer.get_segment_actions().items():
        count = len(rfm[rfm.segment == seg])
        if count > 0:
            print(f"  [{seg}] ({count} customers): {action}")
