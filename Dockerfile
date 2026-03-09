# E-Commerce Revenue Intelligence Platform
# Multi-stage Dockerfile for production deployment

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Base Image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim as base

LABEL maintainer="E-Commerce Intelligence Team"
LABEL description="Revenue Intelligence Platform with Anomaly Detection, Forecasting, and RFM Analysis"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    gunicorn \
    sqlalchemy \
    psycopg2-binary \
    dash-auth

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Build
# ─────────────────────────────────────────────────────────────────────────────
FROM dependencies as build

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed

# Pre-compile Python files
RUN python -m compileall -q src/

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Production
# ─────────────────────────────────────────────────────────────────────────────
FROM base as production

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=build /app .

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8050')" || exit 1

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--threads", "4", \
     "--access-logfile", "-", "--error-logfile", "-", \
     "src.dashboard.app:app.server"]

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Development
# ─────────────────────────────────────────────────────────────────────────────
FROM dependencies as development

# Copy source code
COPY . .

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Create data directories
RUN mkdir -p data/raw data/processed

# Expose port
EXPOSE 8050

# Default command for development
CMD ["python", "src/dashboard/app.py"]
