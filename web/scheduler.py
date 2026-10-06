"""Background scheduler management for the Alpaca Swing Trading Bot.

Schedules automated market scans using APScheduler:
- Daily scan: Monday through Friday at 3:50 PM EST (10 minutes before NYSE close).
- Optional periodic interval scans during market hours.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger("Scheduler")

# US Eastern Timezone (NYSE / NASDAQ)
EASTERN_TZ = pytz.timezone("America/New_York")


class BotScheduler:
    """Manages APScheduler background execution for scheduled market scanning."""

    def __init__(
        self,
        scan_job_func: Callable[[], Any],
        cron_hour: int = 15,
        cron_minute: int = 50,
        interval_minutes: Optional[int] = None,
    ) -> None:
        """Initialize scheduler with scan callback and timing rules.

        Args:
            scan_job_func: Callable to execute for a market scan.
            cron_hour: Hour for daily market-close scan (default: 15 / 3 PM).
            cron_minute: Minute for daily market-close scan (default: 50 / 3:50 PM).
            interval_minutes: Optional interval in minutes to run periodic scans.
        """
        self.scan_job_func = scan_job_func
        self.cron_hour = cron_hour
        self.cron_minute = cron_minute
        self.interval_minutes = interval_minutes

        self._scheduler = BackgroundScheduler(timezone=EASTERN_TZ)
        self._is_running = False

    def start(self) -> None:
        """Configure jobs and start the background scheduler thread."""
        if self._is_running:
            logger.warning("Scheduler is already running.")
            return

        # 1. Main NYSE Pre-Close Scan: Mon-Fri at 3:50 PM EST
        cron_trigger = CronTrigger(
            day_of_week="mon-fri",
            hour=self.cron_hour,
            minute=self.cron_minute,
            timezone=EASTERN_TZ,
        )
        self._scheduler.add_job(
            func=self.scan_job_func,
            trigger=cron_trigger,
            id="market_preclose_scan",
            name=f"NYSE Pre-Close Scan ({self.cron_hour}:{self.cron_minute:02d} EST)",
            replace_existing=True,
            misfire_grace_time=300,
        )
        logger.info(
            "Registered daily scan job: Mon-Fri at %02d:%02d EST (America/New_York)",
            self.cron_hour,
            self.cron_minute,
        )

        # 2. Optional interval scan if configured
        if self.interval_minutes and self.interval_minutes > 0:
            interval_trigger = IntervalTrigger(
                minutes=self.interval_minutes,
                timezone=EASTERN_TZ,
            )
            self._scheduler.add_job(
                func=self.scan_job_func,
                trigger=interval_trigger,
                id="periodic_interval_scan",
                name=f"Periodic Scan (Every {self.interval_minutes}m)",
                replace_existing=True,
                misfire_grace_time=60,
            )
            logger.info("Registered periodic scan job: every %d minutes", self.interval_minutes)

        self._scheduler.start()
        self._is_running = True
        logger.info("Background scheduler started successfully.")

    def shutdown(self, wait: bool = False) -> None:
        """Gracefully stop the background scheduler."""
        if self._is_running:
            self._scheduler.shutdown(wait=wait)
            self._is_running = False
            logger.info("Background scheduler shut down.")

    def get_next_run_time(self) -> Optional[datetime]:
        """Retrieve the earliest next scheduled execution time across all jobs."""
        if not self._is_running:
            return None

        jobs = self._scheduler.get_jobs()
        next_times = [j.next_run_time for j in jobs if j.next_run_time is not None]
        if not next_times:
            return None
        return min(next_times)

    def get_jobs_info(self) -> List[Dict[str, Any]]:
        """Return structured summary of registered scheduler jobs."""
        if not self._is_running:
            return []

        job_list = []
        for job in self._scheduler.get_jobs():
            job_list.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
            })
        return job_list
