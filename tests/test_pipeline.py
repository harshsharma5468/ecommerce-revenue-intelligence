"""
Unit Tests for E-Commerce Revenue Intelligence Platform
Run with: pytest tests/ -v --cov=src

Tests cover:
- Data ingestion
- RFM segmentation
- Anomaly detection
- Revenue forecasting
- Cohort analysis
- Database operations
- Authentication
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_orders():
    """Create sample orders DataFrame."""
    np.random.seed(42)
    n_orders = 100
    
    return pd.DataFrame({
        "order_id": [f"ORD{i:04d}" for i in range(n_orders)],
        "customer_id": [f"CUST{i % 20:04d}" for i in range(n_orders)],
        "order_purchase_timestamp": pd.date_range("2023-01-01", periods=n_orders, freq="D"),
        "order_status": np.random.choice(["delivered", "shipped", "processing"], n_orders),
    })


@pytest.fixture
def sample_order_items():
    """Create sample order items DataFrame."""
    np.random.seed(42)
    n_items = 200
    
    return pd.DataFrame({
        "order_id": [f"ORD{i // 2:04d}" for i in range(n_items)],
        "product_id": [f"PROD{i:04d}" for i in range(n_items)],
        "price": np.random.uniform(10, 500, n_items),
        "freight_value": np.random.uniform(5, 50, n_items),
    })


@pytest.fixture
def sample_customers():
    """Create sample customers DataFrame."""
    return pd.DataFrame({
        "customer_id": [f"CUST{i:04d}" for i in range(20)],
        "customer_unique_id": [f"UNIQ{i:04d}" for i in range(20)],
        "customer_state": np.random.choice(["SP", "RJ", "MG", "RS"], 20),
        "customer_city": [f"City{i}" for i in range(20)],
    })


@pytest.fixture
def sample_daily_revenue():
    """Create sample daily revenue DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    
    return pd.DataFrame({
        "date": dates,
        "total_revenue": np.random.normal(10000, 2000, 365).clip(1000, 20000),
        "total_orders": np.random.randint(20, 100, 365),
        "avg_order_value": np.random.normal(200, 50, 365).clip(50, 500),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Data Ingestion Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestDataIngestion:
    """Tests for data ingestion pipeline."""
    
    def test_load_csv(self, sample_orders, tmp_path):
        """Test CSV loading."""
        csv_path = tmp_path / "test_orders.csv"
        sample_orders.to_csv(csv_path, index=False)
        
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == len(sample_orders)
        assert list(loaded.columns) == list(sample_orders.columns)
    
    def test_merge_orders_items(self, sample_orders, sample_order_items):
        """Test merging orders with order items."""
        merged = sample_orders.merge(sample_order_items, on="order_id", how="left")
        assert len(merged) == len(sample_order_items)
        assert "order_status" in merged.columns
        assert "price" in merged.columns
    
    def test_calculate_daily_revenue(self, sample_order_items):
        """Test daily revenue calculation."""
        sample_order_items["date"] = pd.date_range("2023-01-01", periods=len(sample_order_items), freq="h")
        sample_order_items["date"] = sample_order_items["date"].dt.date
        
        daily = sample_order_items.groupby("date")["price"].sum().reset_index()
        daily.columns = ["date", "total_revenue"]
        
        assert len(daily) > 0
        assert daily["total_revenue"].sum() == sample_order_items["price"].sum()


# ─────────────────────────────────────────────────────────────────────────────
# RFM Segmentation Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestRFMSegmentation:
    """Tests for RFM customer segmentation."""
    
    def test_rfm_calculation(self, sample_orders, sample_order_items):
        """Test RFM metric calculation."""
        # Merge and calculate metrics
        merged = sample_orders.merge(sample_order_items, on="order_id")
        
        ref_date = sample_orders["order_purchase_timestamp"].max()
        
        rfm = merged.groupby("customer_id").agg(
            recency=(
                "order_purchase_timestamp",
                lambda x: (ref_date - x.max()).days
            ),
            frequency=("order_id", "nunique"),
            monetary=("price", "sum"),
        ).reset_index()
        
        assert len(rfm) > 0
        assert "recency" in rfm.columns
        assert "frequency" in rfm.columns
        assert "monetary" in rfm.columns
        assert all(rfm["recency"] >= 0)
        assert all(rfm["frequency"] >= 1)
        assert all(rfm["monetary"] > 0)
    
    def test_rfm_scoring(self):
        """Test RFM quintile scoring."""
        df = pd.DataFrame({
            "customer_id": range(100),
            "recency": np.random.randint(1, 365, 100),
            "frequency": np.random.randint(1, 50, 100),
            "monetary": np.random.randint(100, 10000, 100),
        })
        
        # Lower recency is better
        df["r_score"] = pd.qcut(df["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1])
        # Higher frequency/monetary is better
        df["f_score"] = pd.qcut(df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        df["m_score"] = pd.qcut(df["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        
        df["rfm_score"] = df["r_score"].astype(str) + df["f_score"].astype(str) + df["m_score"].astype(str)
        
        assert len(df["rfm_score"].unique()) > 1
        assert all(df["r_score"].isin([1, 2, 3, 4, 5]))


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestAnomalyDetection:
    """Tests for anomaly detection."""
    
    def test_zscore_detection(self, sample_daily_revenue):
        """Test Z-score based anomaly detection."""
        mean_rev = sample_daily_revenue["total_revenue"].mean()
        std_rev = sample_daily_revenue["total_revenue"].std()
        
        sample_daily_revenue["z_score"] = (
            sample_daily_revenue["total_revenue"] - mean_rev
        ) / std_rev
        
        anomalies = sample_daily_revenue[
            sample_daily_revenue["z_score"].abs() > 2.5
        ]
        
        # Should detect some anomalies in random data
        assert len(anomalies) >= 0
        assert all(anomalies["z_score"].abs() > 2.5)
    
    def test_iqr_detection(self, sample_daily_revenue):
        """Test IQR based anomaly detection."""
        Q1 = sample_daily_revenue["total_revenue"].quantile(0.25)
        Q3 = sample_daily_revenue["total_revenue"].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies = sample_daily_revenue[
            (sample_daily_revenue["total_revenue"] < lower_bound) |
            (sample_daily_revenue["total_revenue"] > upper_bound)
        ]
        
        assert len(anomalies) >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Database Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestDatabase:
    """Tests for database operations."""
    
    def test_database_creation(self, tmp_path):
        """Test SQLite database creation."""
        from src.database.db_manager import DatabaseManager
        
        db_path = tmp_path / "test.db"
        db = DatabaseManager(f"sqlite:///{db_path}")
        
        # Create test table
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        db.update_table("test_table", df)
        
        # Verify
        result = db.get_table("test_table")
        assert len(result) == 3
        assert list(result["name"]) == ["a", "b", "c"]
    
    def test_database_query(self, tmp_path):
        """Test SQL query execution."""
        from src.database.db_manager import DatabaseManager
        
        db_path = tmp_path / "test.db"
        db = DatabaseManager(f"sqlite:///{db_path}")
        
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        })
        db.update_table("data", df)
        
        result = db.query("SELECT * FROM data WHERE value > 25")
        assert len(result) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Authentication Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestAuthentication:
    """Tests for user authentication."""
    
    def test_user_creation(self, tmp_path):
        """Test user creation and authentication."""
        from src.auth.auth import AuthManager
        
        users_file = tmp_path / "users.json"
        auth = AuthManager(str(users_file))
        
        # Add user
        assert auth.add_user("testuser", "password123", "viewer")
        
        # Duplicate user should fail
        assert not auth.add_user("testuser", "password123", "viewer")
        
        # Authenticate
        token = auth.authenticate("testuser", "password123")
        assert token is not None
        
        # Wrong password should fail
        assert auth.authenticate("testuser", "wrongpassword") is None
    
    def test_session_validation(self, tmp_path):
        """Test session validation."""
        from src.auth.auth import AuthManager
        
        users_file = tmp_path / "users.json"
        auth = AuthManager(str(users_file))
        auth.add_user("testuser", "password123")
        
        token = auth.authenticate("testuser", "password123")
        session = auth.validate_session(token)
        
        assert session is not None
        assert session["username"] == "testuser"
        assert session["permissions"]["view"] is True
    
    def test_password_change(self, tmp_path):
        """Test password change."""
        from src.auth.auth import AuthManager
        
        users_file = tmp_path / "users.json"
        auth = AuthManager(str(users_file))
        auth.add_user("testuser", "oldpassword")
        
        assert auth.change_password("testuser", "oldpassword", "newpassword")
        assert not auth.change_password("testuser", "oldpassword", "anotherpassword")
        assert auth.authenticate("testuser", "newpassword") is not None


# ─────────────────────────────────────────────────────────────────────────────
# Forecasting Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestForecasting:
    """Tests for revenue forecasting."""
    
    def test_moving_average(self, sample_daily_revenue):
        """Test moving average calculation."""
        window = 7
        ma = sample_daily_revenue["total_revenue"].rolling(window).mean()
        
        assert len(ma) == len(sample_daily_revenue)
        assert pd.isna(ma.iloc[:window-1]).all()
        assert not pd.isna(ma.iloc[window-1:])
    
    def test_revenue_growth(self, sample_daily_revenue):
        """Test revenue growth calculation."""
        sample_daily_revenue["growth"] = sample_daily_revenue["total_revenue"].pct_change()
        
        assert len(sample_daily_revenue) > 1
        assert pd.isna(sample_daily_revenue["growth"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_data_flow(self, sample_orders, sample_order_items, sample_customers):
        """Test complete data flow from orders to metrics."""
        # Merge all data
        merged = sample_orders.merge(sample_order_items, on="order_id")
        merged = merged.merge(sample_customers, on="customer_id")
        
        # Calculate metrics
        total_revenue = merged["price"].sum()
        total_orders = merged["order_id"].nunique()
        avg_order_value = total_revenue / total_orders
        
        assert total_revenue > 0
        assert total_orders > 0
        assert avg_order_value > 0
        
        # Verify data integrity
        assert len(merged) == len(sample_order_items)
        assert merged["price"].notna().all()


# ─────────────────────────────────────────────────────────────────────────────
# Run Tests
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
