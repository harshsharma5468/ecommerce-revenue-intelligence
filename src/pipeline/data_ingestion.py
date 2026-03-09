"""
Data Ingestion Pipeline
Loads, validates, and merges the Olist e-commerce datasets.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from src.config import DATA_DIR

DATA_DIR = DATA_DIR
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


class DataIngestionPipeline:
    """Orchestrates loading and merging of all Olist CSV datasets."""

    SCHEMAS = {
        "orders": {
            "file": "olist_orders_dataset.csv",
            "date_cols": [
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        },
        "customers": {"file": "olist_customers_dataset.csv", "date_cols": []},
        "order_items": {"file": "olist_order_items_dataset.csv", "date_cols": ["shipping_limit_date"]},
        "payments": {"file": "olist_order_payments_dataset.csv", "date_cols": []},
        "reviews": {
            "file": "olist_order_reviews_dataset.csv",
            "date_cols": ["review_creation_date", "review_answer_timestamp"],
        },
        "products": {"file": "olist_products_dataset.csv", "date_cols": []},
        "sellers": {"file": "olist_sellers_dataset.csv", "date_cols": []},
        "geolocation": {"file": "olist_geolocation_dataset.csv", "date_cols": []},
        "category_translation": {"file": "product_category_name_translation.csv", "date_cols": []},
        "daily_revenue": {"file": "daily_revenue_summary.csv", "date_cols": ["date"]},
    }

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.dataframes = {}
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all dataset files."""
        for name, schema in self.SCHEMAS.items():
            path = self.data_dir / schema["file"]
            if not path.exists():
                logger.warning(f"File not found: {path}")
                continue
            try:
                df = pd.read_csv(path, parse_dates=schema["date_cols"])
                self._validate(df, name)
                self.dataframes[name] = df
                logger.info(f"Loaded '{name}': {len(df):,} rows × {len(df.columns)} cols")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
        return self.dataframes

    def _validate(self, df: pd.DataFrame, name: str):
        """Basic validation checks."""
        if df.empty:
            raise ValueError(f"{name} is empty!")
        null_pct = df.isnull().mean().max()
        if null_pct > 0.5:
            logger.warning(f"{name} has column(s) with >{null_pct:.0%} nulls")

    def build_master_dataset(self) -> pd.DataFrame:
        """
        Join all tables into a single analysis-ready master dataset.
        Returns order-level fact table enriched with dimensions.
        """
        if not self.dataframes:
            self.load_all()

        df = (
            self.dataframes["orders"]
            .merge(self.dataframes["customers"], on="customer_id", how="left")
            .merge(
                self.dataframes["order_items"]
                .groupby("order_id")
                .agg(
                    item_count=("order_item_id", "count"),
                    revenue=("price", "sum"),
                    freight=("freight_value", "sum"),
                    unique_products=("product_id", "nunique"),
                )
                .reset_index(),
                on="order_id",
                how="left",
            )
            .merge(
                self.dataframes["payments"][["order_id", "payment_type", "payment_installments", "payment_value"]],
                on="order_id",
                how="left",
            )
            .merge(
                self.dataframes["reviews"][["order_id", "review_score"]],
                on="order_id",
                how="left",
            )
        )

        # Feature Engineering
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        df["year"]        = df["order_purchase_timestamp"].dt.year
        df["month"]       = df["order_purchase_timestamp"].dt.month
        df["day_of_week"] = df["order_purchase_timestamp"].dt.dayofweek
        df["hour"]        = df["order_purchase_timestamp"].dt.hour
        df["quarter"]     = df["order_purchase_timestamp"].dt.quarter
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
        df["total_order_value"] = df["revenue"].fillna(0) + df["freight"].fillna(0)

        # Delivery time
        df["delivery_days"] = (
            pd.to_datetime(df["order_delivered_customer_date"])
            - df["order_purchase_timestamp"]
        ).dt.days

        logger.info(f"Master dataset built: {len(df):,} rows × {len(df.columns)} cols")
        out_path = PROCESSED_DIR / "master_dataset.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved to {out_path}")
        return df

    def get_category_revenue(self) -> pd.DataFrame:
        """Revenue breakdown by product category."""
        if not self.dataframes:
            self.load_all()
        items    = self.dataframes["order_items"]
        products = self.dataframes["products"]
        orders   = self.dataframes["orders"]
        df = items.merge(products[["product_id","product_category_name"]], on="product_id")
        df = df.merge(orders[["order_id","order_purchase_timestamp"]], on="order_id")
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        df["month_year"] = df["order_purchase_timestamp"].dt.to_period("M")
        cat_rev = df.groupby(["product_category_name","month_year"]).agg(
            revenue=("price","sum"),
            orders=("order_id","nunique"),
        ).reset_index()
        cat_rev.to_csv(PROCESSED_DIR / "category_revenue.csv", index=False)
        return cat_rev


if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    dfs = pipeline.load_all()
    master = pipeline.build_master_dataset()
    cat_rev = pipeline.get_category_revenue()
    print(f"\nMaster dataset shape: {master.shape}")
    print(master.dtypes)
    print("\nCategory Revenue sample:")
    print(cat_rev.head(10))
