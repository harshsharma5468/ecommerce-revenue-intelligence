"""
Main Pipeline Runner
Executes all 5 analysis layers in sequence.

Usage:
    python main_pipeline.py                  # Run all layers
    python main_pipeline.py --layer rfm      # Run specific layer
    python main_pipeline.py --layer forecast
    python main_pipeline.py --stream         # Start stream simulator
"""

import os
import sys

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None


import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger("main_pipeline")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))


def run_ingestion():
    """Load and merge all raw CSV datasets."""
    logger.info("=" * 60)
    logger.info("LAYER 0: Data Ingestion")
    logger.info("=" * 60)
    from pipeline.data_ingestion import DataIngestionPipeline
    pipeline = DataIngestionPipeline()
    pipeline.load_all()
    master = pipeline.build_master_dataset()
    pipeline.get_category_revenue()
    logger.info(f"Master dataset: {master.shape}")
    return master


def run_rfm(master):
    """Layer 1: RFM Segmentation."""
    logger.info("=" * 60)
    logger.info("LAYER 1: RFM Customer Segmentation")
    logger.info("=" * 60)
    from models.rfm_segmentation import RFMAnalyzer

    ref_date = master["order_purchase_timestamp"].max()
    if hasattr(ref_date, "to_pydatetime"):
        ref_date = ref_date.to_pydatetime()

    analyzer = RFMAnalyzer(reference_date=ref_date)
    rfm = analyzer.compute(master)
    rfm = analyzer.cluster()
    analyzer.save()

    summary = analyzer.segment_summary()
    logger.info(f"\n{summary.to_string(index=False)}")
    return rfm


def run_anomaly_detection():
    """Layer 2: Anomaly Detection + Layer 5: Root Cause."""
    logger.info("=" * 60)
    logger.info("LAYER 2: Anomaly Detection | LAYER 5: Root Cause")
    logger.info("=" * 60)
    import pandas as pd

    from models.anomaly_detection import AnomalyDetector, RootCauseAttributor

    raw_dir   = BASE_DIR / "data" / "raw"
    proc_dir  = BASE_DIR / "data" / "processed"

    daily    = pd.read_csv(raw_dir / "daily_revenue_summary.csv", parse_dates=["date"])
    detector = AnomalyDetector(z_threshold=2.5, contamination=0.05)
    result   = detector.detect(daily)
    result.to_csv(proc_dir / "anomaly_detection.csv", index=False)

    report = detector.get_anomaly_report()
    logger.info(f"Anomalies detected: {len(report)}")
    if len(report) > 0:
        logger.info(f"Top anomaly: {report.iloc[0]['date']} "
                    f"(type={report.iloc[0]['anomaly_type']}, "
                    f"severity={report.iloc[0]['anomaly_severity']})")

        # Root cause attribution
        master = pd.read_csv(proc_dir / "master_dataset.csv",
                             parse_dates=["order_purchase_timestamp"])
        attributor = RootCauseAttributor()
        attributor.load(master)
        top_date = str(report.iloc[0]["date"])[:10]
        try:
            causes = attributor.explain(top_date)
            logger.info(f"Root cause for {top_date}:")
            for dim, df in causes.items():
                if len(df) > 0:
                    top_driver = df.iloc[0]
                    logger.info(f"  [{dim}] Top driver: {top_driver.iloc[0]} "
                                f"(d R${top_driver.get('delta', 0):.0f})")
        except Exception as e:
            logger.warning(f"Root cause attribution partial: {e}")

    return result


def run_forecasting():
    """Layer 3: Revenue Forecasting."""
    logger.info("=" * 60)
    logger.info("LAYER 3: 30-Day Revenue Forecast (SARIMA + GBM)")
    logger.info("=" * 60)
    import pandas as pd

    from models.revenue_forecasting import EnsembleForecaster

    raw_dir  = BASE_DIR / "data" / "raw"
    proc_dir = BASE_DIR / "data" / "processed"

    daily = pd.read_csv(raw_dir / "daily_revenue_summary.csv", parse_dates=["date"])

    forecaster = EnsembleForecaster(forecast_horizon=30)
    forecaster.fit(daily)

    forecast_df = forecaster.forecast()
    forecast_df.to_csv(proc_dir / "revenue_forecast.csv", index=False)
    logger.info(f"Forecast next 7d: R${forecast_df['ensemble_forecast'].head(7).sum():,.0f}")
    logger.info(f"Forecast next 30d: R${forecast_df['ensemble_forecast'].sum():,.0f}")

    # Backtest
    bt = forecaster.backtest(daily, test_days=30)
    bt.to_csv(proc_dir / "forecast_backtest.csv", index=False)

    return forecast_df


def run_cohort(master):
    """Layer 4: Cohort Retention + Churn."""
    logger.info("=" * 60)
    logger.info("LAYER 4: Cohort Retention + Churn Prediction")
    logger.info("=" * 60)
    from models.cohort_retention import ChurnPredictor, CohortAnalyzer

    cohort = CohortAnalyzer()
    ret    = cohort.compute(master)
    cohort.compute_ltv(master)
    cohort.save()

    stats = cohort.get_summary_stats()
    logger.info(f"Cohort stats: {stats}")

    # Churn
    churn_features = cohort.compute_churn_prediction_features(master)
    proc_dir = BASE_DIR / "data" / "processed"
    churn_features.to_csv(proc_dir / "churn_features.csv", index=False)

    predictor = ChurnPredictor()
    metrics   = predictor.fit(churn_features)
    logger.info(f"Churn model: {metrics}")

    preds = predictor.predict_proba(churn_features)
    preds.to_csv(proc_dir / "churn_predictions.csv", index=False)
    logger.info(f"High/Critical risk: {(preds.churn_risk.isin(['High','Critical'])).sum()} customers")

    return ret


def run_stream_demo():
    """Simulate real-time data stream."""
    logger.info("=" * 60)
    logger.info("STREAM: Real-Time Event Simulator (30 seconds)")
    logger.info("=" * 60)
    from pipeline.stream_simulator import (LocalStreamSimulator,
                                           SlidingWindowAggregator)

    agg = SlidingWindowAggregator(window_seconds=60)
    sim = LocalStreamSimulator(events_per_second=5)

    received = []
    def on_event(evt):
        received.append(evt)
        agg.add(evt)
        if len(received) % 25 == 0:
            stats = agg.aggregates()
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                  f"Events: {stats.get('event_count',0):4d} | "
                  f"Revenue: R${stats.get('total_revenue',0):>10,.2f} | "
                  f"Spikes: {sum(1 for e in received[-25:] if e.get('is_spike',False))}")

    sim.subscribe(on_event)
    sim.start()

    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()

    stats = agg.aggregates()
    logger.info(f"\nStream summary:")
    logger.info(f"  Total events:  {len(received)}")
    logger.info(f"  Total revenue: R${stats.get('total_revenue',0):,.2f}")
    logger.info(f"  Anomalies:     {sum(1 for e in received if e.get('is_anomaly',False))}")


def run_all():
    """Run the complete pipeline."""
    start = time.time()
    logger.info("\n" + "=" * 60)
    logger.info("E-Commerce Revenue Intelligence Pipeline - Full Run")
    logger.info("=" * 60 + "\n")

    master = run_ingestion()
    run_rfm(master)
    run_anomaly_detection()
    run_forecasting()
    run_cohort(master)

    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info("=" * 60)
    logger.info("\nOutputs:")
    proc = BASE_DIR / "data" / "processed"
    for f in sorted(proc.glob("*.csv")):
        import os
        size = os.path.getsize(f)
        logger.info(f"  {f.name}: {size/1024:.1f} KB")

    logger.info("\nNext steps:")
    logger.info("  pip install dash plotly && python src/dashboard/app.py")
    logger.info("  # OR")
    logger.info("  pip install streamlit plotly && streamlit run src/dashboard/streamlit_app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Revenue Intelligence Pipeline")
    parser.add_argument("--layer", choices=["ingest","rfm","anomaly","forecast","cohort","stream"],
                        help="Run a specific layer only")
    parser.add_argument("--stream", action="store_true", help="Run stream demo")
    args = parser.parse_args()

    if args.stream:
        run_stream_demo()
    elif args.layer == "ingest":
        run_ingestion()
    elif args.layer == "rfm":
        master = run_ingestion()
        run_rfm(master)
    elif args.layer == "anomaly":
        run_anomaly_detection()
    elif args.layer == "forecast":
        run_forecasting()
    elif args.layer == "cohort":
        master = run_ingestion()
        run_cohort(master)
    else:
        run_all()
