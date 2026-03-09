"""
Pipeline Automation Scheduler
Schedules and runs the main pipeline automatically.

Features:
- Daily scheduled runs
- Email notifications on completion/failure
- Log rotation
- Health checks

Usage:
    python scheduler.py              # Run scheduler
    python scheduler.py --once       # Run pipeline once
    python scheduler.py --schedule   # Run with daily schedule
"""

import argparse
import logging
import sys
import time
import schedule
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

# Add src to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.pipeline.email_alerts import EmailNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/scheduler.log"),
    ],
)
logger = logging.getLogger("scheduler")

# Configuration
CONFIG_FILE = Path("scheduler_config.json")

DEFAULT_CONFIG = {
    "schedule_time": "02:00",  # 2 AM daily
    "email_notifications": True,
    "email_on_success": True,
    "email_on_failure": True,
    "max_retries": 3,
    "retry_delay_minutes": 5,
    "log_retention_days": 30,
}


class PipelineScheduler:
    """
    Scheduler for running the revenue intelligence pipeline.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize scheduler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._load_config()
        self.notifier = EmailNotifier()
        self.run_count = 0
        self.success_count = 0
        self.failure_count = 0
    
    def _load_config(self) -> dict:
        """Load configuration from file."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading config: {e}, using defaults")
        return DEFAULT_CONFIG.copy()
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def run_pipeline(self) -> bool:
        """
        Run the main pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info(f"Starting pipeline run #{self.run_count + 1}")
        logger.info("=" * 60)
        
        self.run_count += 1
        
        try:
            # Import and run pipeline
            from main_pipeline import run_all
            
            run_all()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.success_count += 1
            
            logger.info(f"Pipeline completed successfully in {elapsed:.1f}s")
            
            # Send success notification
            if self.config.get("email_on_success") and self.config.get("email_notifications"):
                self.notifier.send_pipeline_complete(
                    elapsed_seconds=elapsed,
                    run_number=self.run_count,
                )
            
            return True
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            
            # Send failure notification
            if self.config.get("email_on_failure") and self.config.get("email_notifications"):
                self.notifier.send_pipeline_failure(
                    error_message=str(e),
                    run_number=self.run_count,
                )
            
            # Retry logic
            retries = 0
            while retries < self.config.get("max_retries", 3):
                retries += 1
                delay = self.config.get("retry_delay_minutes", 5)
                logger.warning(f"Retrying in {delay} minutes (attempt {retries})...")
                time.sleep(delay * 60)
                
                try:
                    from main_pipeline import run_all
                    run_all()
                    logger.info("Retry successful!")
                    return True
                except Exception as retry_error:
                    logger.error(f"Retry {retries} failed: {retry_error}")
            
            return False
    
    def schedule_daily(self, time_str: str = None):
        """
        Schedule daily pipeline runs.
        
        Args:
            time_str: Time string in HH:MM format (default: from config)
        """
        time_str = time_str or self.config.get("schedule_time", "02:00")
        
        schedule.every().day.at(time_str).do(self.run_pipeline)
        
        logger.info(f"Scheduled daily pipeline run at {time_str}")
        logger.info(f"Current time: {datetime.now().strftime('%H:%M:%S')}")
        logger.info("Press Ctrl+C to stop...")
        
        # Run initial check
        self.run_status_check()
        
        # Main loop
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def run_status_check(self):
        """Log current scheduler status."""
        logger.info(f"Scheduler Status:")
        logger.info(f"  Total runs: {self.run_count}")
        logger.info(f"  Successful: {self.success_count}")
        logger.info(f"  Failed: {self.failure_count}")
        if self.run_count > 0:
            success_rate = self.success_count / self.run_count * 100
            logger.info(f"  Success rate: {success_rate:.1f}%")


def create_systemd_service():
    """Create systemd service file for Linux."""
    service_content = """[Unit]
Description=E-Commerce Revenue Intelligence Pipeline Scheduler
After=network.target

[Service]
Type=simple
User=ecommerce
WorkingDirectory=/opt/ecommerce_intelligence
Environment="PATH=/opt/ecommerce_intelligence/venv/bin"
ExecStart=/opt/ecommerce_intelligence/venv/bin/python scheduler.py --schedule
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    service_path = Path("/etc/systemd/system/ecommerce-scheduler.service")
    try:
        service_path.write_text(service_content)
        logger.info(f"Systemd service created: {service_path}")
        logger.info("Run: sudo systemctl daemon-reload && sudo systemctl enable ecommerce-scheduler")
    except PermissionError:
        logger.error("Permission denied. Run with sudo to create systemd service.")
    except Exception as e:
        logger.error(f"Error creating systemd service: {e}")


def create_windows_task():
    """Create Windows Task Scheduler XML."""
    xml_content = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>E-Commerce Revenue Intelligence Pipeline - Daily Run</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>{datetime.now().strftime('%Y-%m-%dT02:00:00')}</StartBoundary>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>python</Command>
      <Arguments>{BASE_DIR}\\scheduler.py --once</Arguments>
      <WorkingDirectory>{BASE_DIR}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""
    
    task_path = Path("ecommerce_scheduler.xml")
    task_path.write_text(xml_content)
    logger.info(f"Windows task XML created: {task_path}")
    logger.info("Import with: schtasks /Create /XML ecommerce_scheduler.xml /TN \"Ecommerce Pipeline\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Scheduler")
    parser.add_argument("--once", action="store_true", help="Run pipeline once")
    parser.add_argument("--schedule", action="store_true", help="Run with daily schedule")
    parser.add_argument("--time", type=str, help="Schedule time (HH:MM)")
    parser.add_argument("--config", action="store_true", help="Show/edit configuration")
    parser.add_argument("--create-service", action="store_true", help="Create systemd/Windows service")
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    scheduler = PipelineScheduler()
    
    if args.create_service:
        if sys.platform == "win32":
            create_windows_task()
        else:
            create_systemd_service()
    elif args.config:
        print(f"Current configuration: {json.dumps(scheduler.config, indent=2)}")
        print(f"\nConfig file: {CONFIG_FILE.absolute()}")
    elif args.once:
        logger.info("Running pipeline once...")
        success = scheduler.run_pipeline()
        sys.exit(0 if success else 1)
    elif args.schedule:
        scheduler.schedule_daily(args.time)
    else:
        parser.print_help()
