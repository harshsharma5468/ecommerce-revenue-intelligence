"""
Database Integration Layer
SQLite/PostgreSQL database handler for revenue intelligence data.
Replaces CSV file loading with database queries.

Usage:
    from src.database.db_manager import DatabaseManager

    db = DatabaseManager("sqlite:///data/ecommerce.db")
    db.load_from_csvs()  # Initial load from CSV files
    df = db.query("SELECT * FROM daily_revenue")
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager for e-commerce revenue intelligence.
    Supports SQLite (default) and PostgreSQL.
    """

    def __init__(self, db_uri: str = "sqlite:///data/ecommerce.db"):
        """
        Initialize database connection.

        Args:
            db_uri: Database URI. Examples:
                - SQLite: "sqlite:///data/ecommerce.db"
                - PostgreSQL: "postgresql://user:pass@localhost:5432/ecommerce"
        """
        self.db_uri = db_uri
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        """Initialize SQLAlchemy engine."""
        try:
            from sqlalchemy import create_engine
            self.engine = create_engine(self.db_uri)
            logger.info(f"Database initialized: {self.db_uri}")
        except ImportError:
            # Fallback to sqlite3 if SQLAlchemy not available
            self.engine = None
            if self.db_uri.startswith("sqlite"):
                self.db_path = self.db_uri.replace("sqlite:///", "")
                logger.info(f"Using sqlite3 fallback: {self.db_path}")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        if self.engine:
            conn = self.engine.connect()
            try:
                yield conn
            finally:
                conn.close()
        else:
            # SQLite fallback
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
            finally:
                conn.close()

    def load_from_csvs(self, raw_dir: Path, processed_dir: Path = None):
        """
        Load all CSV files into database tables.

        Args:
            raw_dir: Path to raw CSV files directory
            processed_dir: Path to processed CSV files (optional)
        """
        logger.info("Loading CSV files into database...")

        # Define table mappings
        tables = {
            "orders": raw_dir / "olist_orders_dataset.csv",
            "customers": raw_dir / "olist_customers_dataset.csv",
            "order_items": raw_dir / "olist_order_items_dataset.csv",
            "payments": raw_dir / "olist_order_payments_dataset.csv",
            "reviews": raw_dir / "olist_order_reviews_dataset.csv",
            "products": raw_dir / "olist_products_dataset.csv",
            "sellers": raw_dir / "olist_sellers_dataset.csv",
            "geolocation": raw_dir / "olist_geolocation_dataset.csv",
            "category_translation": raw_dir / "product_category_name_translation.csv",
            "daily_revenue": raw_dir / "daily_revenue_summary.csv",
        }

        # Add processed tables if available
        if processed_dir:
            processed_tables = {
                "master_dataset": processed_dir / "master_dataset.csv",
                "rfm_segments": processed_dir / "rfm_segments.csv",
                "anomaly_detection": processed_dir / "anomaly_detection.csv",
                "revenue_forecast": processed_dir / "revenue_forecast.csv",
                "cohort_retention": processed_dir / "cohort_retention.csv",
                "churn_predictions": processed_dir / "churn_predictions.csv",
            }
            tables.update(processed_tables)

        # Load each table
        for table_name, csv_path in tables.items():
            if csv_path.exists():
                self.load_table(table_name, csv_path)
            else:
                logger.warning(f"CSV not found: {csv_path}")

        logger.info(f"Loaded {len(tables)} tables into database")

    def load_table(self, table_name: str, csv_path: Path):
        """Load a single CSV file into a database table."""
        try:
            df = pd.read_csv(csv_path)

            # Clean column names
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            if self.engine:
                df.to_sql(table_name, self.engine, if_exists="replace", index=False)
            else:
                # SQLite fallback
                with self.get_connection() as conn:
                    df.to_sql(table_name, conn, if_exists="replace", index=False)

            logger.info(f"Loaded {table_name}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading {table_name}: {e}")

    def query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """
        Execute SQL query and return DataFrame.

        Args:
            sql: SQL query string
            params: Query parameters (optional)

        Returns:
            pandas DataFrame with query results
        """
        try:
            if self.engine:
                return pd.read_sql(sql, self.engine, params=params)
            else:
                with self.get_connection() as conn:
                    return pd.read_sql(sql, conn, params=params)
        except Exception as e:
            logger.error(f"Query error: {e}")
            return pd.DataFrame()

    def get_table(self, table_name: str, columns: str = "*",
                  where: str = None, order_by: str = None,
                  limit: int = None) -> pd.DataFrame:
        """
        Get data from a table with optional filtering.

        Args:
            table_name: Name of the table
            columns: Columns to select (default: *)
            where: WHERE clause (optional)
            order_by: ORDER BY clause (optional)
            limit: LIMIT clause (optional)

        Returns:
            pandas DataFrame with table data
        """
        sql = f"SELECT {columns} FROM {table_name}"
        if where:
            sql += f" WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {limit}"

        return self.query(sql)

    def get_daily_revenue(self, start_date: str = None,
                          end_date: str = None) -> pd.DataFrame:
        """Get daily revenue data with date filtering."""
        sql = "SELECT * FROM daily_revenue WHERE 1=1"
        params = []

        if start_date:
            sql += " AND date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND date <= ?"
            params.append(end_date)

        return self.query(sql, tuple(params))

    def get_rfm_segments(self) -> pd.DataFrame:
        """Get RFM segmentation data."""
        return self.get_table("rfm_segments")

    def get_anomalies(self, only_anomalies: bool = True) -> pd.DataFrame:
        """Get anomaly detection results."""
        if only_anomalies:
            return self.get_table("anomaly_detection", where="is_anomaly = 1")
        return self.get_table("anomaly_detection")

    def get_forecast(self, days: int = 30) -> pd.DataFrame:
        """Get revenue forecast data."""
        return self.get_table("revenue_forecast", limit=days)

    def get_cohort_retention(self) -> pd.DataFrame:
        """Get cohort retention matrix."""
        return self.get_table("cohort_retention")

    def get_customer_detail(self, customer_id: str) -> pd.DataFrame:
        """Get detailed data for a specific customer."""
        sql = """
            SELECT * FROM master_dataset
            WHERE customer_id = ?
        """
        return self.query(sql, (customer_id,))

    def get_state_revenue(self) -> pd.DataFrame:
        """Get revenue grouped by state for geo visualization."""
        sql = """
            SELECT customer_state as state,
                   SUM(total_order_value) as revenue,
                   COUNT(DISTINCT customer_id) as customers
            FROM master_dataset
            GROUP BY customer_state
            ORDER BY revenue DESC
        """
        return self.query(sql)

    def get_category_revenue(self) -> pd.DataFrame:
        """Get revenue by product category."""
        sql = """
            SELECT product_category_name as category,
                   SUM(price) as revenue,
                   COUNT(DISTINCT order_id) as orders
            FROM master_dataset
            WHERE product_category_name IS NOT NULL
            GROUP BY product_category_name
            ORDER BY revenue DESC
        """
        return self.query(sql)

    def get_payment_mix(self) -> pd.DataFrame:
        """Get payment method distribution."""
        sql = """
            SELECT payment_type,
                   COUNT(*) as count,
                   SUM(payment_value) as total_value
            FROM payments
            GROUP BY payment_type
            ORDER BY total_value DESC
        """
        return self.query(sql)

    def get_review_sentiment(self) -> pd.DataFrame:
        """Get review data with sentiment scores."""
        sql = """
            SELECT review_id, order_id, review_score,
                   review_comment_title, review_comment_message,
                   review_creation_date,
                   (review_score - 3) / 2.0 as sentiment_score
            FROM reviews
            WHERE review_score IS NOT NULL
        """
        return self.query(sql)

    def get_churn_data(self) -> pd.DataFrame:
        """Get churn prediction data."""
        return self.get_table("churn_predictions")

    def update_table(self, table_name: str, df: pd.DataFrame):
        """Update a table with new data."""
        try:
            if self.engine:
                df.to_sql(table_name, self.engine, if_exists="replace", index=False)
            else:
                with self.get_connection() as conn:
                    df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info(f"Updated table {table_name}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Error updating {table_name}: {e}")

    def get_tables(self) -> List[str]:
        """Get list of all tables in database."""
        if self.engine:
            sql = "SELECT name FROM sqlite_master WHERE type='table'"
        else:
            sql = "SELECT name FROM sqlite_master WHERE type='table'"

        result = self.query(sql)
        return result["name"].tolist() if "name" in result.columns else []

    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# Default database instance
DEFAULT_DB = DatabaseManager("sqlite:///data/ecommerce.db")


def get_db() -> DatabaseManager:
    """Get default database instance."""
    return DEFAULT_DB
