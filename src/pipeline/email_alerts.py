"""
Email Alerts Module
Sends email notifications for pipeline events and anomalies.

Configuration (set in .env or environment variables):
    SMTP_SERVER=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your_email@gmail.com
    SMTP_PASSWORD=your_app_password
    EMAIL_RECIPIENTS=admin@company.com,analyst@company.com

Usage:
    from src.pipeline.email_alerts import EmailNotifier

    notifier = EmailNotifier()
    notifier.send_pipeline_complete(elapsed_seconds=120)
    notifier.send_anomaly_alert(date="2024-01-15", anomaly_type="spike")
"""

import logging
import os
import smtplib
from datetime import datetime
from email.mime.html import MIMEHtml
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class EmailNotifier:
    """
    Email notification handler for pipeline events and anomalies.
    """

    def __init__(self, config_file: str = ".env"):
        """
        Initialize email notifier.

        Args:
            config_file: Path to environment file
        """
        # Load configuration
        self._load_config(config_file)

        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.recipients = self._parse_recipients(os.getenv("EMAIL_RECIPIENTS", ""))
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_user)
        self.enabled = bool(self.smtp_user and self.smtp_password)

    def _load_config(self, config_file: str):
        """Load environment variables from .env file."""
        env_path = Path(config_file)
        if env_path.exists():
            try:
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            os.environ.setdefault(key.strip(), value.strip())
            except Exception as e:
                logger.warning(f"Error loading .env: {e}")

    def _parse_recipients(self, recipients_str: str) -> List[str]:
        """Parse comma-separated recipient list."""
        if not recipients_str:
            return []
        return [r.strip() for r in recipients_str.split(",") if r.strip()]

    def _send_email(self, subject: str, body: str, html: bool = False) -> bool:
        """
        Send email to all recipients.

        Args:
            subject: Email subject
            body: Email body
            html: Whether body is HTML

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.warning("Email not configured. Set SMTP_USER and SMTP_PASSWORD")
            return False

        if not self.recipients:
            logger.warning("No recipients configured")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = f"[Revenue Intelligence] {subject}"

            # Add timestamp
            body_with_time = f"{body}\n\n---\nSent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            if html:
                msg.attach(MIMEHtml(body_with_time, "html"))
            else:
                msg.attach(MIMEText(body_with_time, "plain"))

            # Connect and send
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
            server.quit()

            logger.info(f"Email sent to {len(self.recipients)} recipients: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_pipeline_complete(self, elapsed_seconds: float, run_number: int = 1) -> bool:
        """
        Send notification when pipeline completes successfully.

        Args:
            elapsed_seconds: Pipeline execution time
            run_number: Run number
        """
        minutes = elapsed_seconds / 60
        subject = f"Pipeline Complete (Run #{run_number})"

        body = f"""
✅ Pipeline Completed Successfully

Execution Time: {elapsed_seconds:.1f}s ({minutes:.1f} minutes)
Run Number: #{run_number}
Completed At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The following layers were executed:
  ✓ Data Ingestion
  ✓ RFM Segmentation
  ✓ Anomaly Detection
  ✓ Revenue Forecasting
  ✓ Cohort Analysis

View dashboard: http://localhost:8050
        """

        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <div style="background-color: #2ECC71; color: white; padding: 20px; border-radius: 8px;">
        <h2>✅ Pipeline Completed Successfully</h2>
    </div>
    <div style="padding: 20px;">
        <p><strong>Execution Time:</strong> {elapsed_seconds:.1f}s ({minutes:.1f} minutes)</p>
        <p><strong>Run Number:</strong> #{run_number}</p>
        <p><strong>Completed At:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h3>Executed Layers:</h3>
        <ul style="color: #2ECC71;">
            <li>✓ Data Ingestion</li>
            <li>✓ RFM Segmentation</li>
            <li>✓ Anomaly Detection</li>
            <li>✓ Revenue Forecasting</li>
            <li>✓ Cohort Analysis</li>
        </ul>

        <p><a href="http://localhost:8050" style="background-color: #6C63FF; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">View Dashboard</a></p>
    </div>
</body>
</html>
        """

        return self._send_email(subject, html_body, html=True)

    def send_pipeline_failure(self, error_message: str, run_number: int = 1) -> bool:
        """
        Send notification when pipeline fails.

        Args:
            error_message: Error message
            run_number: Run number
        """
        subject = f"⚠️ Pipeline FAILED (Run #{run_number})"

        body = f"""
❌ Pipeline Failed

Run Number: #{run_number}
Failed At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Error Message:
{error_message}

Please check the logs and investigate.
        """

        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <div style="background-color: #E74C3C; color: white; padding: 20px; border-radius: 8px;">
        <h2>❌ Pipeline Failed</h2>
    </div>
    <div style="padding: 20px;">
        <p><strong>Run Number:</strong> #{run_number}</p>
        <p><strong>Failed At:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; border-left: 4px solid #E74C3C;">
            <strong>Error Message:</strong><br>
            <code style="color: #E74C3C;">{error_message}</code>
        </div>

        <p style="color: #666;">Please check the logs and investigate.</p>
    </div>
</body>
</html>
        """

        return self._send_email(subject, html_body, html=True)

    def send_anomaly_alert(self, date: str, anomaly_type: str, severity: str,
                           revenue_impact: float, top_driver: str) -> bool:
        """
        Send alert when critical anomaly is detected.

        Args:
            date: Anomaly date
            anomaly_type: Type (spike/drop)
            severity: Severity level
            revenue_impact: Revenue impact amount
            top_driver: Top contributing factor
        """
        icon = "📈" if anomaly_type == "spike" else "📉"
        color = "#2ECC71" if anomaly_type == "spike" else "#E74C3C"

        subject = f"{icon} Critical Anomaly Detected: {anomaly_type.upper()} on {date}"

        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <div style="background-color: {color}; color: white; padding: 20px; border-radius: 8px;">
        <h2>{icon} Critical Anomaly Detected</h2>
    </div>
    <div style="padding: 20px;">
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Date:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{date}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Type:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{anomaly_type.upper()}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Severity:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">
                    <span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px;">{severity}</span>
                </td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Revenue Impact:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">R$ {revenue_impact:,.0f}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Top Driver:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{top_driver}</td>
            </tr>
        </table>

        <p style="margin-top: 20px;">
            <a href="http://localhost:8050" style="background-color: #6C63FF; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
                View in Dashboard
            </a>
        </p>
    </div>
</body>
</html>
        """

        return self._send_email(subject, html_body, html=True)

    def send_forecast_summary(self, forecast_7d: float, forecast_30d: float,
                              model_accuracy: float) -> bool:
        """
        Send weekly forecast summary.

        Args:
            forecast_7d: 7-day forecast total
            forecast_30d: 30-day forecast total
            model_accuracy: Model MAPE accuracy
        """
        subject = f"📊 Weekly Forecast Summary"

        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <div style="background-color: #3498DB; color: white; padding: 20px; border-radius: 8px;">
        <h2>📊 Weekly Forecast Summary</h2>
    </div>
    <div style="padding: 20px;">
        <div style="display: flex; gap: 20px;">
            <div style="flex: 1; background-color: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0; color: #666;">Next 7 Days</h3>
                <p style="font-size: 24px; color: #3498DB; margin: 10px 0;">R$ {forecast_7d:,.0f}</p>
            </div>
            <div style="flex: 1; background-color: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0; color: #666;">Next 30 Days</h3>
                <p style="font-size: 24px; color: #2ECC71; margin: 10px 0;">R$ {forecast_30d:,.0f}</p>
            </div>
        </div>

        <p style="margin-top: 20px;"><strong>Model Accuracy (MAPE):</strong> {100 - model_accuracy:.1f}%</p>

        <p style="margin-top: 20px;">
            <a href="http://localhost:8050" style="background-color: #6C63FF; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
                View Full Forecast
            </a>
        </p>
    </div>
</body>
</html>
        """

        return self._send_email(subject, html_body, html=True)

    def test_connection(self) -> bool:
        """Test email configuration."""
        return self._send_email(
            "Test Email",
            "This is a test email from Revenue Intelligence Platform.\n\nConfiguration is working correctly!"
        )


# Global notifier instance
notifier = EmailNotifier()


def get_notifier() -> EmailNotifier:
    """Get global notifier instance."""
    return notifier
