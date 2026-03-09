"""
Layer 3: 30-Day Revenue Forecasting
Ensemble of:
  - SARIMA (seasonal ARIMA via statsmodels)
  - Gradient Boosting (XGBoost-style via sklearn)
  - Simple Exponential Smoothing baseline
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering for Time Series
# ─────────────────────────────────────────────────────────────────────────────
def build_ts_features(df: pd.DataFrame, target_col: str = "total_revenue",
                       lags: list = None, rolling_windows: list = None) -> pd.DataFrame:
    """Create lag + rolling features for supervised time series learning."""
    if lags is None:
        lags = [1, 2, 3, 7, 14, 28]
    if rolling_windows is None:
        rolling_windows = [3, 7, 14, 30]

    df = df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Calendar features
    df["day_of_week"]    = df["date"].dt.dayofweek
    df["day_of_month"]   = df["date"].dt.day
    df["week_of_year"]   = df["date"].dt.isocalendar().week.astype(int)
    df["month"]          = df["date"].dt.month
    df["quarter"]        = df["date"].dt.quarter
    df["year"]           = df["date"].dt.year
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)

    # Cyclic encoding
    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]        = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]        = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag features
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # Rolling statistics
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = df[target_col].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df[target_col].shift(1).rolling(w).std()
        df[f"roll_max_{w}"]  = df[target_col].shift(1).rolling(w).max()
        df[f"roll_min_{w}"]  = df[target_col].shift(1).rolling(w).min()

    # Momentum
    df["mom_7"]  = df[target_col].shift(1) / df[target_col].shift(8).replace(0, np.nan) - 1
    df["mom_30"] = df[target_col].shift(1) / df[target_col].shift(31).replace(0, np.nan) - 1

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SARIMA Wrapper (pure Python / statsmodels)
# ─────────────────────────────────────────────────────────────────────────────
class SARIMAForecaster:
    """SARIMA model wrapper. Falls back to ETS if statsmodels unavailable."""

    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,7)):
        self.order          = order
        self.seasonal_order = seasonal_order
        self.model_fit      = None
        self._backend       = None

    def fit(self, series: pd.Series):
        """Fit SARIMA model to a time series."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_fit = model.fit(disp=False, maxiter=200)
            self._backend  = "sarima"
            logger.info(f"SARIMA fitted | AIC={self.model_fit.aic:.1f}")
        except ImportError:
            logger.warning("statsmodels not available, using ETS fallback")
            self._fit_ets(series)
        except Exception as e:
            logger.warning(f"SARIMA failed: {e}. Using ETS fallback.")
            self._fit_ets(series)

    def _fit_ets(self, series: pd.Series):
        """Simple exponential smoothing fallback."""
        self._ets_alpha  = 0.3
        self._ets_series = series.values.copy()
        self._backend    = "ets"

    def forecast(self, steps: int = 30) -> np.ndarray:
        """Return point forecasts."""
        if self._backend == "sarima":
            fc = self.model_fit.forecast(steps=steps)
            return np.maximum(fc.values, 0)
        else:
            return self._ets_forecast(steps)

    def forecast_with_ci(self, steps: int = 30) -> pd.DataFrame:
        """Return forecasts with confidence intervals."""
        if self._backend == "sarima":
            pred = self.model_fit.get_forecast(steps=steps)
            ci   = pred.conf_int(alpha=0.1)
            return pd.DataFrame({
                "forecast":  np.maximum(pred.predicted_mean.values, 0),
                "lower_80":  np.maximum(ci.iloc[:, 0].values, 0),
                "upper_80":  np.maximum(ci.iloc[:, 1].values, 0),
            })
        else:
            preds = self._ets_forecast(steps)
            std   = np.std(self._ets_series[-30:]) if len(self._ets_series) > 30 else 0
            return pd.DataFrame({
                "forecast": preds,
                "lower_80": np.maximum(preds - 1.28 * std, 0),
                "upper_80": preds + 1.28 * std,
            })

    def _ets_forecast(self, steps: int) -> np.ndarray:
        """Simple exponential smoothing forecast."""
        alpha   = self._ets_alpha
        history = list(self._ets_series)
        level   = history[-1]
        forecasts = []
        for _ in range(steps):
            forecasts.append(level)
            level = alpha * level + (1 - alpha) * level
        return np.array(forecasts)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Boosting Forecaster
# ─────────────────────────────────────────────────────────────────────────────
class GBMForecaster:
    """
    Supervised gradient boosting approach for multi-step forecasting.
    Uses recursive strategy: predict one step, feed into next.
    """

    def __init__(self, n_estimators: int = 300, max_depth: int = 5,
                 learning_rate: float = 0.05, subsample: float = 0.8):
        self.model  = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            random_state=42, loss="huber"
        )
        self.scaler     = StandardScaler()
        self.feature_cols: list = []
        self.target_col         = "total_revenue"
        self._train_df: pd.DataFrame = None

    def fit(self, df: pd.DataFrame) -> dict:
        """Fit on feature-engineered DataFrame. Returns train metrics."""
        df_feat  = build_ts_features(df, self.target_col)
        df_clean = df_feat.dropna()

        self.feature_cols = [c for c in df_clean.columns
                              if c not in [self.target_col, "date"]]
        self._train_df = df_clean.copy()

        X = df_clean[self.feature_cols]
        y = df_clean[self.target_col]

        # Walk-forward CV: last 30 days as hold-out
        split = max(len(X) - 30, int(len(X) * 0.8))
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)

        self.model.fit(X_train_s, y_train)

        val_preds = np.maximum(self.model.predict(X_val_s), 0)
        mae  = mean_absolute_error(y_val, val_preds)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        mape = np.mean(np.abs((y_val - val_preds) / np.maximum(y_val, 1))) * 100

        metrics = {"MAE": round(mae,2), "RMSE": round(rmse,2), "MAPE": round(mape,2)}
        logger.info(f"GBM fitted | {metrics}")

        # Feature importance
        self.feature_importance_ = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return metrics

    def forecast(self, steps: int = 30) -> np.ndarray:
        """Recursive multi-step forecast."""
        if self._train_df is None:
            raise RuntimeError("Call fit() first.")

        history = self._train_df.copy()
        last_date = pd.to_datetime(history["date"].max())
        preds = []

        for i in range(steps):
            next_date = last_date + pd.Timedelta(days=i+1)
            # Create synthetic next row with only required columns
            new_row = {"date": next_date, self.target_col: np.nan}
            temp = pd.concat([history[["date", self.target_col]],
                               pd.DataFrame([new_row])], ignore_index=True)
            feat_df  = build_ts_features(temp, self.target_col)
            # Ensure all feature columns exist, fill missing with 0
            for col in self.feature_cols:
                if col not in feat_df.columns:
                    feat_df[col] = 0
            last_row = feat_df.iloc[[-1]][self.feature_cols].fillna(0)
            last_scaled = self.scaler.transform(last_row)
            pred = float(np.maximum(self.model.predict(last_scaled)[0], 0))
            preds.append(pred)
            # Feed prediction back
            history = pd.concat([history, pd.DataFrame([{
                "date": next_date, self.target_col: pred}])], ignore_index=True)

        return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Forecaster
# ─────────────────────────────────────────────────────────────────────────────
class EnsembleForecaster:
    """
    Combines SARIMA + GBM forecasts via weighted average.
    Weights determined by inverse validation MAE.
    """

    def __init__(self, forecast_horizon: int = 30):
        self.horizon  = forecast_horizon
        self.sarima   = SARIMAForecaster()
        self.gbm      = GBMForecaster()
        self.weights  = {"sarima": 0.40, "gbm": 0.60}
        self.metrics  = {}

    def fit(self, daily_df: pd.DataFrame):
        """Fit both models."""
        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        series = df.set_index("date")["total_revenue"]

        logger.info("Fitting SARIMA...")
        self.sarima.fit(series)

        logger.info("Fitting GBM...")
        gbm_metrics = self.gbm.fit(df)
        self.metrics["gbm"] = gbm_metrics
        logger.info("Ensemble fitted.")

    def forecast(self) -> pd.DataFrame:
        """
        Generate H-step ahead forecasts.
        Returns DataFrame with date, each model's forecast, and ensemble.
        """
        sarima_fc = self.sarima.forecast(self.horizon)
        gbm_fc    = self.gbm.forecast(self.horizon)

        ensemble = (
            self.weights["sarima"] * sarima_fc +
            self.weights["gbm"]   * gbm_fc
        )

        # Confidence intervals (from SARIMA)
        sarima_ci = self.sarima.forecast_with_ci(self.horizon)
        # Scale CI to ensemble
        ratio = ensemble / np.maximum(sarima_fc, 1)
        lower = sarima_ci["lower_80"].values * ratio
        upper = sarima_ci["upper_80"].values * ratio

        last_date = pd.Timestamp.now().normalize()
        dates     = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                   periods=self.horizon, freq="D")

        result = pd.DataFrame({
            "date":            dates,
            "sarima_forecast": sarima_fc,
            "gbm_forecast":    gbm_fc,
            "ensemble_forecast": ensemble,
            "lower_80":        np.maximum(lower, 0),
            "upper_80":        upper,
        })
        return result

    def backtest(self, daily_df: pd.DataFrame, test_days: int = 30) -> pd.DataFrame:
        """
        Walk-forward backtest on last N days.
        Returns comparison of actual vs predicted.
        """
        df  = daily_df.sort_values("date").copy()
        train = df.iloc[:-test_days]
        test  = df.iloc[-test_days:]

        bt_sarima = SARIMAForecaster()
        bt_gbm    = GBMForecaster()
        series    = train.set_index("date")["total_revenue"]
        bt_sarima.fit(series)
        bt_gbm.fit(train)

        sarima_fc = bt_sarima.forecast(test_days)
        gbm_fc    = bt_gbm.forecast(test_days)
        ensemble  = 0.4 * sarima_fc + 0.6 * gbm_fc

        result = test[["date","total_revenue"]].copy().reset_index(drop=True)
        result["sarima_forecast"]   = sarima_fc
        result["gbm_forecast"]      = gbm_fc
        result["ensemble_forecast"] = ensemble

        actual = result["total_revenue"].values
        for col in ["sarima_forecast","gbm_forecast","ensemble_forecast"]:
            preds = result[col].values
            mae   = mean_absolute_error(actual, preds)
            mape  = np.mean(np.abs((actual - preds) / np.maximum(actual, 1))) * 100
            logger.info(f"  {col}: MAE={mae:.0f} | MAPE={mape:.1f}%")

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    daily = pd.read_csv(RAW_DIR / "daily_revenue_summary.csv", parse_dates=["date"])
    daily = daily.sort_values("date")

    forecaster = EnsembleForecaster(forecast_horizon=30)
    forecaster.fit(daily)

    forecast_df = forecaster.forecast()
    print("\n=== 30-Day Revenue Forecast ===")
    print(forecast_df.to_string(index=False))

    # Backtest
    print("\n=== Backtest (last 30 days) ===")
    bt = forecaster.backtest(daily, test_days=30)
    print(bt.to_string(index=False))

    forecast_df.to_csv(PROCESSED_DIR / "revenue_forecast.csv", index=False)
    bt.to_csv(PROCESSED_DIR / "forecast_backtest.csv", index=False)
    logger.info("Forecasts saved.")
