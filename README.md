# 🛒 E-Commerce Revenue Intelligence Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-green.svg)](https://www.docker.com/)

A comprehensive revenue intelligence platform for e-commerce businesses featuring **anomaly detection**, **revenue forecasting**, **customer segmentation (RFM)**, **cohort analysis**, **churn prediction**, and **sentiment analysis**.

![Dashboard Preview](docs/dashboard_preview.png)

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Docker Deployment](#-docker-deployment)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 📊 Analytics Layers

| Layer | Description | Output |
|-------|-------------|--------|
| **Data Ingestion** | Load and merge all CSV datasets | `master_dataset.csv` |
| **RFM Segmentation** | Customer segmentation (Recency, Frequency, Monetary) | `rfm_segments.csv` |
| **Anomaly Detection** | Detect revenue spikes/drops using Z-Score, IQR, Isolation Forest | `anomaly_detection.csv` |
| **Revenue Forecasting** | 30-day forecast using SARIMA + GBM ensemble | `revenue_forecast.csv` |
| **Cohort Retention** | Monthly retention rates and LTV analysis | `cohort_retention.csv` |
| **Churn Prediction** | Customer churn risk scoring | `churn_predictions.csv` |
| **Sentiment Analysis** | NLP-based review sentiment scoring | `sentiment_analysis.csv` |

### 🎯 Interactive Dashboard

- **Real-time Revenue Tracking** with anomaly detection
- **30-Day Revenue Forecast** with confidence intervals
- **3D RFM Visualization** for customer segments
- **Brazil Geographic Map** with state-level revenue
- **Cohort Retention Heatmap**
- **Product Category Treemap**
- **Customer Detail Panel** with SHAP explainability
- **What-If Simulator** for scenario planning

### 🤖 AI/ML Features

- **Automated Insights**: NLP-generated summaries
- **SHAP Values**: Model explainability for churn predictions
- **Sentiment Gauge**: Customer review sentiment analysis
- **Next Best Action**: Recommendations per customer segment
- **Anomaly Confidence Scores**: Z-score based probability

### 📧 Alerts & Automation

- **Email Notifications**: Pipeline completion/failure alerts
- **Critical Anomaly Alerts**: Real-time notifications
- **Daily Scheduling**: Automated pipeline runs
- **Weekly Forecast Summary**: Email reports

---

## 🚀 Quick Start

### Option 1: Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ecommerce_intelligence.git
cd ecommerce_intelligence

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main_pipeline.py

# Start dashboard
python src/dashboard/app.py
```

Open http://localhost:8050 in your browser.

### Option 2: Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run just the app
docker build -t ecommerce-intelligence .
docker run -p 8050:8050 -v $(pwd)/data:/app/data ecommerce-intelligence
```

---

## 📦 Installation

### Prerequisites

- Python 3.12+
- pip 23.0+
- (Optional) Docker 20.10+

### Step-by-Step

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ecommerce_intelligence.git
   cd ecommerce_intelligence
   ```

2. **Create virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset** (Olist E-Commerce Dataset from Kaggle)
   ```bash
   # Place CSV files in data/raw/
   # Required files:
   # - olist_orders_dataset.csv
   # - olist_customers_dataset.csv
   # - olist_order_items_dataset.csv
   # - olist_order_payments_dataset.csv
   # - olist_order_reviews_dataset.csv
   # - olist_products_dataset.csv
   # - olist_sellers_dataset.csv
   # - olist_geolocation_dataset.csv
   # - product_category_name_translation.csv
   ```

5. **Configure environment** (optional)
   ```bash
   # Copy .env.example to .env
   cp .env.example .env

   # Edit with your settings
   # SMTP_SERVER=smtp.gmail.com
   # SMTP_USER=your_email@gmail.com
   # SMTP_PASSWORD=your_app_password
   ```

---

## 💻 Usage

### Run Full Pipeline

```bash
python main_pipeline.py
```

### Run Specific Layer

```bash
# Run only RFM segmentation
python main_pipeline.py --layer rfm

# Run only forecasting
python main_pipeline.py --layer forecast

# Run only anomaly detection
python main_pipeline.py --layer anomaly
```

### Run Stream Simulator

```bash
python main_pipeline.py --stream
```

### Run Dashboard

```bash
python src/dashboard/app.py
```

### Run Scheduler (Daily Automation)

```bash
# Run once
python scheduler.py --once

# Run with daily schedule
python scheduler.py --schedule

# Create system service
python scheduler.py --create-service
```

### Run Tests

```bash
pytest tests/ -v --cov=src
```

---

## 🏗️ Architecture

```
ecommerce_intelligence/
├── main_pipeline.py          # Main entry point
├── scheduler.py              # Pipeline scheduler
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose
├── data/
│   ├── raw/                 # Raw CSV files
│   └── processed/           # Processed outputs
├── src/
│   ├── dashboard/
│   │   └── app.py          # Dash dashboard
│   ├── models/
│   │   ├── anomaly_detection.py
│   │   ├── revenue_forecasting.py
│   │   ├── rfm_segmentation.py
│   │   ├── cohort_retention.py
│   │   ├── shap_explainability.py
│   │   └── sentiment_analysis.py
│   ├── pipeline/
│   │   ├── data_ingestion.py
│   │   ├── stream_simulator.py
│   │   └── email_alerts.py
│   ├── database/
│   │   └── db_manager.py
│   └── auth/
│       └── auth.py
└── tests/
    └── test_pipeline.py
```

### Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Raw CSVs   │ ──► │  Ingestion   │ ──► │   Master    │
│  (9 files)  │     │   Pipeline   │     │  Dataset    │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
         ┌───────────────────────────────────────┼───────────────────────────────────────┐
         │                                       │                                       │
         ▼                                       ▼                                       ▼
┌─────────────────┐                   ┌─────────────────┐                   ┌─────────────────┐
│  RFM Analysis   │                   │   Anomaly       │                   │   Forecast      │
│  (Segmentation) │                   │   Detection     │                   │   (SARIMA+GBM)  │
└────────┬────────┘                   └────────┬────────┘                   └────────┬────────┘
         │                                     │                                      │
         ▼                                     ▼                                      ▼
┌─────────────────┐                   ┌─────────────────┐                   ┌─────────────────┐
│  Churn          │                   │   Root Cause    │                   │   Dashboard     │
│  Prediction     │                   │   Attribution   │                   │   Visualization │
└─────────────────┘                   └─────────────────┘                   └─────────────────┘
```

---

## 📖 API Reference

### Database Manager

```python
from src.database.db_manager import DatabaseManager

db = DatabaseManager("sqlite:///data/ecommerce.db")
db.load_from_csvs(Path("data/raw"))

# Query data
df = db.get_daily_revenue(start_date="2024-01-01")
df = db.get_rfm_segments()
df = db.get_customer_detail("customer_123")
```

### Authentication

```python
from src.auth import AuthManager

auth = AuthManager()
auth.add_user("analyst", "password123", "analyst")

# In dashboard callback
token = auth.authenticate("analyst", "password123")
session = auth.validate_session(token)
```

### Sentiment Analysis

```python
from src.models.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
results = analyzer.analyze_reviews(reviews_df)

# Get summary
summary = analyzer.get_sentiment_summary()

# Get trend
trend = analyzer.get_sentiment_trend(freq="W")
```

### SHAP Explainability

```python
from src.models.shap_explainability import ChurnExplainer

explainer = ChurnExplainer()
explainer.fit(churn_features_df)

# Explain specific customer
explanation = explainer.explain_customer("customer_123")
print(explanation["churn_probability"])
print(explanation["next_best_action"])
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///data/ecommerce.db` |
| `SMTP_SERVER` | SMTP server for email alerts | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP port | `587` |
| `SMTP_USER` | Email address for alerts | - |
| `SMTP_PASSWORD` | Email password/app password | - |
| `EMAIL_RECIPIENTS` | Comma-separated email list | - |
| `SECRET_KEY` | Dashboard secret key | Auto-generated |

### Scheduler Configuration

Edit `scheduler_config.json`:

```json
{
  "schedule_time": "02:00",
  "email_notifications": true,
  "email_on_success": true,
  "email_on_failure": true,
  "max_retries": 3,
  "retry_delay_minutes": 5
}
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t ecommerce-intelligence:latest .
```

### Run with Docker Compose

```bash
# Start all services (app, scheduler, streamlit)
docker-compose --profile production up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Production Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    image: ecommerce-intelligence:latest
    ports:
      - "8050:8050"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/ecommerce
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - postgres
      - redis
  
  postgres:
    image: postgres:15-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
  
  redis:
    image: redis:7-alpine
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Run Specific Test Class

```bash
pytest tests/test_pipeline.py::TestRFMSegmentation -v
```

---

## 📊 Dashboard Screenshots

### Revenue & Anomalies Tab
Shows daily revenue with anomaly detection, Z-score analysis, and Brazilian state heatmap.

### Forecast Tab
30-day revenue forecast using SARIMA + GBM ensemble with 80% confidence intervals.

### RFM 3D Tab
Interactive 3D visualization of customer segments based on Recency, Frequency, and Monetary value.

### AI Insights Tab
Automated NLP insights, sentiment gauge, what-if simulator, and customer detail panel with SHAP values.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: [Olist E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/olist-ecommerce-dataset)
- Framework: [Plotly Dash](https://plotly.com/dash/)
- Time Series: [statsmodels](https://www.statsmodels.org/)
- ML: [scikit-learn](https://scikit-learn.org/)

---

## 📞 Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/yourusername/ecommerce_intelligence/issues)
- Email: support@example.com

---

<div align="center">

**Built with ❤️ for E-Commerce Analytics**

[⬆ Back to Top](#-ecommerce-revenue-intelligence-platform)

</div>
