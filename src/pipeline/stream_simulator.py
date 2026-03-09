"""
Real-Time Stream Simulator
Simulates Apache Kafka / Pub-Sub by replaying historical orders as a live stream.
Replace the LocalStreamSimulator with KafkaProducer for production use.
"""

import json
import time
import random
import threading
import queue
from datetime import datetime, timedelta
from typing import Callable, Optional
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# ─────────────────────────────────────────────────────────────────────────────
# Transaction Event Schema
# ─────────────────────────────────────────────────────────────────────────────
def make_transaction_event(order_id: str, customer_id: str, revenue: float,
                            category: str, state: str, payment_type: str) -> dict:
    return {
        "event_id":     f"EVT_{random.randint(1_000_000, 9_999_999)}",
        "order_id":     order_id,
        "customer_id":  customer_id,
        "timestamp":    datetime.utcnow().isoformat(),
        "revenue":      round(revenue, 2),
        "category":     category,
        "state":        state,
        "payment_type": payment_type,
        "channel":      random.choice(["mobile", "web", "app"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Local In-Process Stream Simulator
# ─────────────────────────────────────────────────────────────────────────────
class LocalStreamSimulator:
    """
    Simulates a real-time event stream without Kafka.
    Replays historical Olist transactions with configurable speed.
    
    Usage:
        sim = LocalStreamSimulator(speed_multiplier=10)
        sim.subscribe(my_handler)
        sim.start()
        ...
        sim.stop()
    """

    def __init__(self, speed_multiplier: int = 5, events_per_second: int = 3):
        self.speed          = speed_multiplier
        self.eps            = events_per_second
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._subscribers:  list[Callable] = []
        self._running       = False
        self._producer_thr: Optional[threading.Thread] = None
        self._consumer_thr: Optional[threading.Thread] = None
        self._event_buffer: list[dict] = []
        self._load_historical()

    def _load_historical(self):
        """Load historical orders to use as event templates."""
        try:
            orders   = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
            items    = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
            payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
            products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
            customers= pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")

            revenue_by_order = items.groupby("order_id")["price"].sum().reset_index()
            revenue_by_order.columns = ["order_id", "revenue"]
            merged = (
                orders[["order_id","customer_id","order_status"]]
                .merge(revenue_by_order, on="order_id", how="left")
                .merge(payments[["order_id","payment_type"]], on="order_id", how="left")
                .merge(customers[["customer_id","customer_state"]], on="customer_id", how="left")
            )

            # Map products to categories
            cat_map = dict(zip(products.product_id, products.product_category_name))
            item_cats = items.copy()
            item_cats["category"] = item_cats["product_id"].map(cat_map)
            top_cat = item_cats.groupby("order_id")["category"].first().reset_index()
            merged = merged.merge(top_cat, on="order_id", how="left")
            merged = merged.dropna(subset=["revenue"])

            self._event_buffer = merged.to_dict("records")
            logger.info(f"Loaded {len(self._event_buffer):,} historical events for streaming")
        except Exception as e:
            logger.warning(f"Could not load historical data, using synthetic: {e}")
            self._event_buffer = []

    def _generate_event(self) -> dict:
        """Generate one transaction event."""
        if self._event_buffer:
            rec = random.choice(self._event_buffer)
            # Add small noise to revenue to make it feel live
            revenue = float(rec.get("revenue", 50)) * random.uniform(0.9, 1.1)
            return make_transaction_event(
                order_id=rec.get("order_id", f"ORD_{random.randint(1,999999):06d}"),
                customer_id=rec.get("customer_id", f"CUST_{random.randint(1,3000):05d}"),
                revenue=max(revenue, 5.0),
                category=rec.get("category", "electronics"),
                state=rec.get("customer_state", "SP"),
                payment_type=rec.get("payment_type", "credit_card"),
            )
        else:
            # Fully synthetic fallback
            return make_transaction_event(
                order_id=f"ORD_{random.randint(1,999999):06d}",
                customer_id=f"CUST_{random.randint(1,3000):05d}",
                revenue=round(random.lognormvariate(4, 0.8), 2),
                category=random.choice(["electronics","furniture","health_beauty","toys"]),
                state=random.choice(["SP","RJ","MG","RS","PR"]),
                payment_type=random.choice(["credit_card","boleto","voucher"]),
            )

    def _producer_loop(self):
        """Produce events into the queue at the configured rate."""
        interval = 1.0 / self.eps
        while self._running:
            evt = self._generate_event()
            try:
                self._queue.put(evt, timeout=0.5)
            except queue.Full:
                pass
            time.sleep(interval)

    def _consumer_loop(self):
        """Consume events from queue and dispatch to subscribers."""
        while self._running or not self._queue.empty():
            try:
                evt = self._queue.get(timeout=0.5)
                for handler in self._subscribers:
                    try:
                        handler(evt)
                    except Exception as e:
                        logger.error(f"Subscriber error: {e}")
            except queue.Empty:
                continue

    def subscribe(self, handler: Callable[[dict], None]):
        """Register an event handler."""
        self._subscribers.append(handler)

    def start(self):
        """Start streaming."""
        self._running = True
        self._producer_thr = threading.Thread(target=self._producer_loop, daemon=True)
        self._consumer_thr = threading.Thread(target=self._consumer_loop, daemon=True)
        self._producer_thr.start()
        self._consumer_thr.start()
        logger.info(f"Stream started at {self.eps} events/sec")

    def stop(self):
        """Stop streaming."""
        self._running = False
        if self._producer_thr:
            self._producer_thr.join(timeout=2)
        if self._consumer_thr:
            self._consumer_thr.join(timeout=2)
        logger.info("Stream stopped")


# ─────────────────────────────────────────────────────────────────────────────
# Kafka Integration (requires kafka-python)
# ─────────────────────────────────────────────────────────────────────────────
class KafkaStreamProducer:
    """
    Production Kafka producer. Requires:
        pip install kafka-python
        Kafka broker running at KAFKA_BROKER
    
    Usage:
        producer = KafkaStreamProducer(broker="localhost:9092", topic="orders")
        producer.send_event(event_dict)
    """

    def __init__(self, broker: str = "localhost:9092", topic: str = "ecommerce-orders"):
        self.broker = broker
        self.topic  = topic
        self._producer = None
        self._connect()

    def _connect(self):
        try:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(
                bootstrap_servers=self.broker,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            logger.info(f"Connected to Kafka at {self.broker}")
        except ImportError:
            logger.warning("kafka-python not installed. Use LocalStreamSimulator instead.")
        except Exception as e:
            logger.error(f"Kafka connection failed: {e}")

    def send_event(self, event: dict):
        if self._producer:
            self._producer.send(self.topic, event)

    def close(self):
        if self._producer:
            self._producer.close()


class KafkaStreamConsumer:
    """
    Production Kafka consumer.
    
    Usage:
        consumer = KafkaStreamConsumer(broker="localhost:9092", topic="orders")
        for event in consumer.events():
            process(event)
    """

    def __init__(self, broker: str = "localhost:9092", topic: str = "ecommerce-orders",
                 group_id: str = "revenue-intelligence"):
        self.broker   = broker
        self.topic    = topic
        self.group_id = group_id

    def events(self):
        try:
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.broker,
                group_id=self.group_id,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",
            )
            for msg in consumer:
                yield msg.value
        except ImportError:
            logger.error("kafka-python not installed.")


# ─────────────────────────────────────────────────────────────────────────────
# Sliding Window Aggregator
# ─────────────────────────────────────────────────────────────────────────────
class SlidingWindowAggregator:
    """
    Maintains a rolling window of events and computes live aggregates.
    Useful for real-time anomaly detection triggers.
    """

    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self._events: list[dict] = []
        self._lock = threading.Lock()

    def add(self, event: dict):
        ts = datetime.utcnow()
        event["_ts"] = ts
        with self._lock:
            self._events.append(event)
            cutoff = ts - timedelta(seconds=self.window_seconds)
            self._events = [e for e in self._events if e["_ts"] > cutoff]

    def aggregates(self) -> dict:
        with self._lock:
            if not self._events:
                return {}
            revenues = [e["revenue"] for e in self._events]
            return {
                "window_seconds":  self.window_seconds,
                "event_count":     len(self._events),
                "total_revenue":   round(sum(revenues), 2),
                "avg_revenue":     round(np.mean(revenues), 2),
                "max_revenue":     round(max(revenues), 2),
                "min_revenue":     round(min(revenues), 2),
                "std_revenue":     round(float(np.std(revenues)), 2),
                "events_per_min":  round(len(self._events) / (self.window_seconds / 60), 2),
                "top_categories":  self._top_n("category", 3),
                "top_states":      self._top_n("state", 3),
            }

    def _top_n(self, field: str, n: int) -> dict:
        from collections import Counter
        counts = Counter(e.get(field,"unknown") for e in self._events)
        return dict(counts.most_common(n))


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting stream simulator demo (10 seconds)...")
    agg = SlidingWindowAggregator(window_seconds=60)
    sim = LocalStreamSimulator(events_per_second=5)

    received = []
    def on_event(evt):
        received.append(evt)
        agg.add(evt)
        if len(received) % 10 == 0:
            stats = agg.aggregates()
            print(f"  Events: {stats['event_count']} | "
                  f"Revenue: R${stats['total_revenue']:,.2f} | "
                  f"Avg: R${stats['avg_revenue']:.2f}")

    sim.subscribe(on_event)
    sim.start()
    time.sleep(10)
    sim.stop()

    print(f"\nFinal: Processed {len(received)} events")
    print(json.dumps(agg.aggregates(), indent=2))
