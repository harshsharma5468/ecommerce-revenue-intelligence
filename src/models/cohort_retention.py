"""
Layer 4: Cohort Retention Analysis
Month-over-month retention heatmaps and LTV calculation.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"


class CohortAnalyzer:
    """
    Computes cohort-based retention rates and customer lifetime value.

    A cohort = all customers who made their FIRST purchase in a given month.
    Retention = % of cohort still ordering in subsequent months.
    """

    def __init__(self):
        self.retention_matrix: pd.DataFrame = None
        self.revenue_matrix:   pd.DataFrame = None
        self.cohort_sizes:     pd.Series    = None
        self.ltv_df:           pd.DataFrame = None

    def compute(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build cohort retention matrix.

        Args:
            master_df: Master dataset with customer_id, order_purchase_timestamp, total_order_value
        Returns:
            Retention rate matrix (cohorts × months)
        """
        df = master_df.copy()
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M")

        # First purchase month per customer
        first_purchase = df.groupby("customer_id")["order_month"].min().rename("cohort_month")
        df = df.merge(first_purchase, on="customer_id")

        # Cohort period = months since first purchase
        df["cohort_period"] = (df["order_month"] - df["cohort_month"]).apply(
            lambda x: x.n if hasattr(x, "n") else 0
        )

        # Retention count matrix
        cohort_data = df.groupby(["cohort_month", "cohort_period"])["customer_id"].nunique()
        cohort_data = cohort_data.reset_index()
        cohort_pivot = cohort_data.pivot_table(
            index="cohort_month", columns="cohort_period", values="customer_id"
        )

        # Cohort sizes (period=0)
        self.cohort_sizes = cohort_pivot[0]

        # Retention rates
        retention = cohort_pivot.divide(self.cohort_sizes, axis=0) * 100
        self.retention_matrix = retention.round(1)

        logger.info(f"Cohort retention matrix: {retention.shape}")

        # Revenue per cohort
        rev_data = df.groupby(["cohort_month","cohort_period"])["total_order_value"].sum()
        rev_pivot = rev_data.reset_index().pivot_table(
            index="cohort_month", columns="cohort_period", values="total_order_value"
        )
        self.revenue_matrix = rev_pivot.round(2)

        return self.retention_matrix

    def compute_ltv(self, master_df: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
        """
        Compute Customer Lifetime Value by cohort over N periods.
        LTV = Cumulative revenue / Initial cohort size
        """
        if self.revenue_matrix is None:
            self.compute(master_df)

        ltv = pd.DataFrame(index=self.revenue_matrix.index)
        cumrev = self.revenue_matrix.iloc[:, :periods].cumsum(axis=1)

        for col in cumrev.columns:
            ltv[f"LTV_M{col}"] = (cumrev[col] / self.cohort_sizes).round(2)

        self.ltv_df = ltv
        return ltv

    def get_summary_stats(self) -> dict:
        """Return key retention metrics."""
        if self.retention_matrix is None:
            raise RuntimeError("Run compute() first.")

        m1_retention = self.retention_matrix.get(1, pd.Series()).dropna()
        m3_retention = self.retention_matrix.get(3, pd.Series()).dropna()
        m6_retention = self.retention_matrix.get(6, pd.Series()).dropna()

        return {
            "total_cohorts":         len(self.retention_matrix),
            "avg_m1_retention_pct":  round(m1_retention.mean(), 1) if len(m1_retention) > 0 else 0,
            "avg_m3_retention_pct":  round(m3_retention.mean(), 1) if len(m3_retention) > 0 else 0,
            "avg_m6_retention_pct":  round(m6_retention.mean(), 1) if len(m6_retention) > 0 else 0,
            "best_cohort_m1":        str(m1_retention.idxmax()) if len(m1_retention) > 0 else "N/A",
            "worst_cohort_m1":       str(m1_retention.idxmin()) if len(m1_retention) > 0 else "N/A",
            "avg_cohort_size":       round(self.cohort_sizes.mean(), 0),
            "total_customers":       int(self.cohort_sizes.sum()),
        }

    def get_retention_heatmap_data(self) -> pd.DataFrame:
        """Return retention matrix formatted for heatmap plotting."""
        df = self.retention_matrix.copy()
        df.index = df.index.astype(str)
        df.columns = [f"Month {c}" for c in df.columns]
        return df

    def compute_churn_prediction_features(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build churn-ready feature matrix.
        Label: customer has NOT ordered in the last 90 days.
        """
        df = master_df.copy()
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        reference = df["order_purchase_timestamp"].max()

        features = df.groupby("customer_id").agg(
            order_count=("order_id",       "nunique"),
            total_revenue=("total_order_value", "sum"),
            avg_order_value=("total_order_value","mean"),
            avg_review=("review_score",    "mean"),
            days_since_first=("order_purchase_timestamp",
                               lambda x: (reference - x.min()).days),
            days_since_last=("order_purchase_timestamp",
                              lambda x: (reference - x.max()).days),
            unique_categories=("total_order_value", "count"),
        ).reset_index()

        features["purchase_frequency"]  = features["order_count"] / (
            features["days_since_first"].clip(lower=1) / 30)
        features["is_churned"]          = (features["days_since_last"] > 90).astype(int)
        features["avg_inter_order_days"] = features["days_since_first"] / features["order_count"].clip(lower=1)

        logger.info(f"Churn features: {len(features)} customers | "
                    f"Churned: {features.is_churned.mean():.1%}")
        return features

    def save(self):
        """Save matrices to processed directory."""
        if self.retention_matrix is not None:
            ret = self.retention_matrix.copy()
            ret.index = ret.index.astype(str)
            ret.to_csv(PROCESSED_DIR / "cohort_retention.csv")
            logger.info("Retention matrix saved.")

        if self.ltv_df is not None:
            ltv = self.ltv_df.copy()
            ltv.index = ltv.index.astype(str)
            ltv.to_csv(PROCESSED_DIR / "cohort_ltv.csv")
            logger.info("LTV matrix saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Churn Predictor
# ─────────────────────────────────────────────────────────────────────────────
class ChurnPredictor:
    """
    Binary churn classifier using Gradient Boosting.
    Churn label: no purchase in last 90 days.
    """

    def __init__(self):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        self.model  = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42)
        self.scaler = StandardScaler()
        self.feature_cols = [
            "order_count","total_revenue","avg_order_value","avg_review",
            "days_since_first","days_since_last","purchase_frequency",
            "avg_inter_order_days",
        ]
        self.is_fitted = False

    def fit(self, features_df: pd.DataFrame) -> dict:
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        X = features_df[self.feature_cols].fillna(0)
        y = features_df["is_churned"]

        X_s = self.scaler.fit_transform(X)
        cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = cross_val_score(self.model, X_s, y, cv=cv, scoring="roc_auc")
        self.model.fit(X_s, y)
        self.is_fitted = True

        self.feature_importance_ = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        metrics = {
            "cv_auc_mean": round(auc_scores.mean(), 3),
            "cv_auc_std":  round(auc_scores.std(), 3),
            "churn_rate":  round(y.mean(), 3),
        }
        logger.info(f"Churn model: CV AUC={metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
        return metrics

    def predict_proba(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        X = features_df[self.feature_cols].fillna(0)
        X_s = self.scaler.transform(X)
        probs = self.model.predict_proba(X_s)[:, 1]
        result = features_df[["customer_id"]].copy()
        result["churn_probability"] = probs.round(3)
        result["churn_risk"] = pd.cut(probs, bins=[0, 0.3, 0.6, 0.8, 1.0],
                                       labels=["Low","Medium","High","Critical"])
        return result.sort_values("churn_probability", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    master = pd.read_csv(PROCESSED_DIR / "master_dataset.csv",
                         parse_dates=["order_purchase_timestamp"])

    cohort = CohortAnalyzer()
    ret    = cohort.compute(master)
    ltv    = cohort.compute_ltv(master)
    cohort.save()

    print("\n=== Retention Matrix (first 6 months) ===")
    print(ret.iloc[:, :7].to_string())

    print("\n=== Summary Stats ===")
    for k, v in cohort.get_summary_stats().items():
        print(f"  {k}: {v}")

    # Churn
    churn_features = cohort.compute_churn_prediction_features(master)
    churn_features.to_csv(PROCESSED_DIR / "churn_features.csv", index=False)

    predictor = ChurnPredictor()
    metrics   = predictor.fit(churn_features)
    print(f"\n=== Churn Model Metrics ===\n{metrics}")

    preds = predictor.predict_proba(churn_features)
    preds.to_csv(PROCESSED_DIR / "churn_predictions.csv", index=False)
    print(f"\nTop 10 at-risk customers:")
    print(preds.head(10).to_string(index=False))
    print(f"\nRisk distribution:\n{preds.churn_risk.value_counts()}")
