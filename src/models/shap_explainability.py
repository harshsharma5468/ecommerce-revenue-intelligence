"""
SHAP Model Explainability for Churn Predictions
Generates SHAP values and visualization for churn model interpretability.

Usage:
    from src.models.shap_explainability import ChurnExplainer

    explainer = ChurnExplainer()
    explainer.fit(churn_features_df)

    # Get explanations for specific customer
    explanation = explainer.explain_customer(customer_id="abc123")

    # Generate SHAP summary plot
    fig = explainer.plot_shap_summary()
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Install with: pip install shap")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ChurnExplainer:
    """
    SHAP-based explainability for churn prediction model.
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize churn explainer.

        Args:
            model_type: Model type ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.explainer = None
        self.shap_values = None
        self.customer_data: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame, target_col: str = "churned") -> Dict:
        """
        Fit churn prediction model and compute SHAP values.

        Args:
            df: DataFrame with features and target
            target_col: Name of target column

        Returns:
            Model metrics dictionary
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Using feature importance fallback.")
            return self._fit_fallback(df, target_col)

        logger.info("Fitting churn prediction model with SHAP explainability...")

        # Prepare features
        feature_cols = [c for c in df.columns if c not in [target_col, "customer_id"]]
        self.feature_cols = feature_cols

        X = df[feature_cols].fillna(0)
        y = df[target_col]
        self.customer_data = df.copy()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create model
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42,
                class_weight="balanced", n_jobs=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )

        # Fit model
        self.model.fit(self.X_train_scaled, self.y_train)

        # Evaluate
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        auc_score = roc_auc_score(self.y_test, y_pred_proba)

        logger.info(f"Model AUC: {auc_score:.4f}")

        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X_test_scaled)

        metrics = {
            "auc": round(auc_score, 4),
            "n_features": len(feature_cols),
            "n_train": len(self.y_train),
            "n_test": len(self.y_test),
        }

        logger.info(f"SHAP explainer fitted | {metrics}")
        return metrics

    def _fit_fallback(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Fallback without SHAP - uses feature importance only."""
        feature_cols = [c for c in df.columns if c not in [target_col, "customer_id"]]
        self.feature_cols = feature_cols

        X = df[feature_cols].fillna(0)
        y = df[target_col]
        self.customer_data = df.copy()

        X_scaled = self.scaler.fit_transform(X)

        self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train_scaled, self.y_train)

        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        auc_score = roc_auc_score(self.y_test, y_pred_proba)

        return {
            "auc": round(auc_score, 4),
            "n_features": len(feature_cols),
            "shap_available": False,
        }

    def explain_customer(self, customer_id: str) -> Dict:
        """
        Get SHAP explanation for a specific customer.

        Args:
            customer_id: Customer ID to explain

        Returns:
            Dictionary with explanation details
        """
        if self.customer_data is None:
            raise RuntimeError("Call fit() first")

        customer = self.customer_data[self.customer_data["customer_id"] == customer_id]

        if len(customer) == 0:
            return {"error": f"Customer {customer_id} not found"}

        # Get features
        features = customer[self.feature_cols].fillna(0).values[0]
        features_scaled = self.scaler.transform([features])[0]

        # Get prediction
        churn_proba = self.model.predict_proba([features_scaled])[0][1]

        # Get SHAP values
        if SHAP_AVAILABLE and self.explainer:
            shap_vals = self.explainer.shap_values([features_scaled])
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # Get values for positive class
            shap_vals = shap_vals[0]
        else:
            # Fallback: use feature importance as proxy
            shap_vals = self.model.feature_importances_ * (features_scaled - self.scaler.mean_)

        # Create explanation
        feature_importance = list(zip(self.feature_cols, shap_vals))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            "customer_id": customer_id,
            "churn_probability": round(float(churn_proba), 4),
            "churn_risk": self._risk_category(churn_proba),
            "shap_values": {k: round(float(v), 4) for k, v in feature_importance},
            "top_positive_factors": [k for k, v in feature_importance if v > 0][:5],
            "top_negative_factors": [k for k, v in feature_importance if v < 0][:5],
            "next_best_action": self._get_next_best_action(churn_proba, feature_importance),
        }

    def _risk_category(self, probability: float) -> str:
        """Categorize churn risk level."""
        if probability >= 0.8:
            return "Critical"
        elif probability >= 0.6:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        elif probability >= 0.2:
            return "Low"
        else:
            return "Very Low"

    def _get_next_best_action(self, probability: float, factors: List[Tuple]) -> str:
        """Recommend next best action based on churn risk and factors."""
        if probability < 0.3:
            return "Maintain engagement - send newsletter"
        elif probability < 0.5:
            return "Monitor closely - check recent activity"
        elif probability < 0.7:
            return "Proactive outreach - personalized offer"
        else:
            # High risk - check top factors
            top_factor = factors[0][0] if factors else ""
            if "recency" in top_factor.lower():
                return "Win-back campaign - time-sensitive discount"
            elif "frequency" in top_factor.lower():
                return "Engagement campaign - new product recommendations"
            elif "monetary" in top_factor.lower():
                return "VIP treatment - exclusive offers"
            else:
                return "Urgent win-back - personal contact recommended"

    def plot_shap_summary(self) -> go.Figure:
        """
        Create SHAP summary plot showing feature importance.

        Returns:
            Plotly figure
        """
        if not SHAP_AVAILABLE or self.shap_values is None:
            return self._plot_feature_importance()

        # Get mean absolute SHAP values
        shap_sum = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": shap_sum,
        }).sort_values("importance", ascending=True)

        fig = go.Figure(go.Bar(
            x=importance_df["importance"],
            y=importance_df["feature"],
            orientation="h",
            marker=dict(color=importance_df["importance"], colorscale="Viridis"),
        ))

        fig.update_layout(
            title="SHAP Feature Importance (Mean |SHAP Value|)",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400,
            showlegend=False,
        )

        return fig

    def _plot_feature_importance(self) -> go.Figure:
        """Fallback feature importance plot."""
        if self.model is None:
            return go.Figure().add_annotation(text="Model not fitted")

        importance_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=True)

        fig = go.Figure(go.Bar(
            x=importance_df["importance"],
            y=importance_df["feature"],
            orientation="h",
            marker=dict(color=importance_df["importance"], colorscale="Blues"),
        ))

        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            height=400,
            showlegend=False,
        )

        return fig

    def plot_shap_beeswarm(self, max_display: int = 20) -> go.Figure:
        """
        Create SHAP beeswarm plot.

        Args:
            max_display: Maximum features to display

        Returns:
            Plotly figure
        """
        if not SHAP_AVAILABLE or self.shap_values is None:
            return go.Figure().add_annotation(text="SHAP not available")

        # Get top features
        shap_sum = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(shap_sum)[-max_display:]

        # Create plot data
        feature_names = [self.feature_cols[i] for i in top_indices]
        shap_data = self.shap_values[:, top_indices]

        fig = make_subplots(rows=len(feature_names), cols=1, shared_xaxes=True,
                            vertical_spacing=0.02, row_heights=[1] * len(feature_names))

        for i, (feature, col_idx) in enumerate(zip(feature_names, top_indices)):
            shap_vals = self.shap_values[:, col_idx]
            feature_vals = self.X_test_scaled[:, col_idx]

            # Color by feature value
            colors = []
            for val in feature_vals:
                if val < np.median(feature_vals):
                    colors.append("blue")
                else:
                    colors.append("red")

            fig.add_trace(go.Scatter(
                x=shap_vals,
                y=[feature] * len(shap_vals),
                mode="markers",
                marker=dict(size=6, color=colors, opacity=0.6),
                showlegend=False,
            ), row=i+1, col=1)

        fig.update_layout(
            title="SHAP Beeswarm Plot",
            xaxis_title="SHAP Value (impact on model output)",
            height=max_display * 30 + 100,
            showlegend=False,
        )

        return fig

    def get_feature_direction(self, feature: str) -> str:
        """
        Get direction of feature impact on churn.

        Args:
            feature: Feature name

        Returns:
            Direction string
        """
        if feature not in self.feature_cols:
            return "Unknown"

        idx = self.feature_cols.index(feature)

        if SHAP_AVAILABLE and self.shap_values is not None:
            avg_shap = np.mean(self.shap_values[:, idx])
            if avg_shap > 0:
                return "Positive (increases churn)"
            else:
                return "Negative (decreases churn)"

        return "Unknown (SHAP not available)"

    def save(self, path: str = "data/processed/churn_explainer.pkl"):
        """Save explainer to file."""
        import pickle

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Churn explainer saved to {path}")

    @classmethod
    def load(cls, path: str = "data/processed/churn_explainer.pkl") -> "ChurnExplainer":
        """Load explainer from file."""
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)


# Global explainer instance
explainer = ChurnExplainer()


def get_explainer() -> ChurnExplainer:
    """Get global explainer instance."""
    return explainer
