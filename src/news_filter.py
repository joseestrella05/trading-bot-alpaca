"""News and Earnings Calendar Filter.

Enforces fundamental and macroeconomic risk controls:
1. Macroeconomic Filter: Checks Forex Factory public feed for high-impact USD events (FOMC, CPI, etc.).
   Blocks purchase if a critical event is scheduled within the next 2 hours.
2. Earnings Calendar Filter: Checks quarterly earnings announcement dates for evaluated tickers via yfinance.
   Blocks purchase if earnings are scheduled within the next 2 business days.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
import json
import logging
import os
from typing import Any, Dict, List, Optional
import httpx
import yfinance as yf

logger = logging.getLogger("NewsFilter")

FOREX_FACTORY_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
DISK_MACRO_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "macro_calendar_cache.json")


@dataclass(frozen=True)
class NewsFilterResult:
    """Evaluation result for news and earnings events.

    Attributes:
        symbol: Market ticker evaluated.
        safe: True if both macro and earnings conditions are safe to trade.
        earnings_safe: True if no imminent earnings within 2 business days.
        macro_safe: True if no high-impact US macro event within 2 hours.
        earnings_date: Next earnings announcement date if found.
        imminent_macro_event: Title and time of imminent macro event if any.
        reason: Diagnostic explanation.
    """
    symbol: str
    safe: bool
    earnings_safe: bool
    macro_safe: bool
    earnings_date: Optional[str] = None
    imminent_macro_event: Optional[str] = None
    reason: str = "Safe"


class NewsFilter:
    """Manages macroeconomic calendar and quarterly earnings risk checks with in-memory caching."""

    def __init__(
        self,
        macro_window_hours: float = 2.0,
        earnings_window_days: int = 2,
        cache_ttl_seconds: int = 1800,  # 30 minutes
    ) -> None:
        """Initialize filter parameters and caching structures.

        Args:
            macro_window_hours: Window in hours to avoid high-impact macro announcements.
            earnings_window_days: Window in business days to avoid earnings announcements.
            cache_ttl_seconds: In-memory cache duration for macro events.
        """
        self.macro_window_hours = macro_window_hours
        self.earnings_window_days = earnings_window_days
        self.cache_ttl_seconds = cache_ttl_seconds

        # In-memory caches
        self._macro_cache_time: Optional[datetime] = None
        self._macro_events_cache: List[Dict[str, Any]] = []
        self._earnings_cache: Dict[str, tuple[datetime, Optional[date]]] = {}

        # Known broad-market ETFs that do not file corporate earnings
        self._etf_symbols = {
            "SPY", "QQQ", "IWM", "DIA", "SMH", "XLE", "XLF", "XLV", "XLI", "XLY",
            "XLP", "XLU", "XLB", "XLC", "VNQ", "VTI", "VOO", "TLT", "EEM", "EFA"
        }

    def _load_disk_cache(self) -> List[Dict[str, Any]]:
        """Load fallback macro events from disk if available."""
        if os.path.exists(DISK_MACRO_CACHE_PATH):
            try:
                with open(DISK_MACRO_CACHE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    events = []
                    for item in data:
                        country = str(item.get("country", "")).strip().upper()
                        impact = str(item.get("impact", "")).strip().lower()
                        if country == "USD" and impact in ("high", "critical"):
                            events.append(item)
                    return events
            except Exception as e:
                logger.debug("Could not read disk macro cache: %s", e)
        return []

    def _save_disk_cache(self, raw_data: Any) -> None:
        """Save raw macro events to disk cache."""
        try:
            os.makedirs(os.path.dirname(DISK_MACRO_CACHE_PATH), exist_ok=True)
            with open(DISK_MACRO_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=2)
        except Exception as e:
            logger.debug("Could not save disk macro cache: %s", e)

    def fetch_macro_calendar(self) -> List[Dict[str, Any]]:
        """Fetch weekly calendar from Forex Factory public feed with caching.

        Returns:
            List of high-impact USD events.
        """
        now_utc = datetime.now(timezone.utc)
        if (
            self._macro_cache_time is not None
            and (now_utc - self._macro_cache_time).total_seconds() < self.cache_ttl_seconds
        ):
            return self._macro_events_cache

        self._macro_cache_time = now_utc
        events: List[Dict[str, Any]] = []
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            with httpx.Client(timeout=10.0, headers=headers) as client:
                resp = client.get(FOREX_FACTORY_CALENDAR_URL)
                if resp.status_code == 200:
                    raw_events = resp.json()
                    self._save_disk_cache(raw_events)
                    for item in raw_events:
                        country = str(item.get("country", "")).strip().upper()
                        impact = str(item.get("impact", "")).strip().lower()
                        # Only check high-impact USD events
                        if country == "USD" and impact in ("high", "critical"):
                            events.append(item)
                    self._macro_events_cache = events
                    logger.debug("Fetched %d high-impact USD macro events from Forex Factory.", len(events))
                    return self._macro_events_cache
                else:
                    logger.warning("Forex Factory feed returned HTTP %s. Using cached events.", resp.status_code)
        except Exception as exc:
            logger.warning("Failed to fetch Forex Factory calendar (%s). Proceeding with cache.", exc)

        # Fallback to disk cache if in-memory cache is empty
        if not self._macro_events_cache:
            disk_events = self._load_disk_cache()
            if disk_events:
                self._macro_events_cache = disk_events
                logger.info("Loaded %d high-impact USD events from disk cache.", len(disk_events))

        return self._macro_events_cache

    async def fetch_macro_calendar_async(self) -> List[Dict[str, Any]]:
        """Asynchronously fetch weekly calendar from Forex Factory public feed with caching.

        Returns:
            List of high-impact USD events.
        """
        now_utc = datetime.now(timezone.utc)
        if (
            self._macro_cache_time is not None
            and (now_utc - self._macro_cache_time).total_seconds() < self.cache_ttl_seconds
        ):
            return self._macro_events_cache

        self._macro_cache_time = now_utc
        events: List[Dict[str, Any]] = []
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
                resp = await client.get(FOREX_FACTORY_CALENDAR_URL)
                if resp.status_code == 200:
                    raw_events = resp.json()
                    self._save_disk_cache(raw_events)
                    for item in raw_events:
                        country = str(item.get("country", "")).strip().upper()
                        impact = str(item.get("impact", "")).strip().lower()
                        if country == "USD" and impact in ("high", "critical"):
                            events.append(item)
                    self._macro_events_cache = events
                    logger.debug("Fetched %d high-impact USD macro events from Forex Factory async.", len(events))
                    return self._macro_events_cache
                else:
                    logger.warning("Forex Factory feed returned HTTP %s async. Using cached events.", resp.status_code)
        except Exception as exc:
            logger.warning("Failed to fetch Forex Factory calendar async (%s). Proceeding with cache.", exc)

        if not self._macro_events_cache:
            disk_events = self._load_disk_cache()
            if disk_events:
                self._macro_events_cache = disk_events

        return self._macro_events_cache

    def check_macro_risk(self) -> tuple[bool, Optional[str]]:
        """Check if any critical US macro event (FOMC, CPI, etc.) is scheduled in the next 2 hours.

        Returns:
            Tuple of (macro_safe: bool, imminent_event_description: Optional[str])
        """
        events = self.fetch_macro_calendar()
        if not events:
            return True, None

        now = datetime.now(timezone.utc)

        for event in events:
            date_str = event.get("date")
            title = event.get("title", "High-Impact Event")
            if not date_str:
                continue

            try:
                # Forex Factory dates are ISO format (e.g. "2026-10-07T14:00:00-04:00")
                event_dt = datetime.fromisoformat(date_str)
                if event_dt.tzinfo is None:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
                else:
                    event_dt = event_dt.astimezone(timezone.utc)

                diff_seconds = (event_dt - now).total_seconds()
                diff_hours = diff_seconds / 3600.0

                # If event is within next 2 hours (or occurred in last 30 minutes of high volatility)
                if -0.5 <= diff_hours <= self.macro_window_hours:
                    desc = f"{title} at {event_dt.strftime('%H:%M UTC')} (in {int(diff_hours*60)}m)"
                    logger.info("Macro Risk Alert: %s within %.1fh window", desc, self.macro_window_hours)
                    return False, desc
            except Exception as exc:
                logger.debug("Could not parse macro event date '%s': %s", date_str, exc)

        return True, None

    def get_next_earnings_date(self, symbol: str) -> Optional[date]:
        """Fetch next quarterly earnings report date for symbol using yfinance.

        Args:
            symbol: Equity ticker symbol.

        Returns:
            date object if upcoming earnings found, None otherwise.
        """
        sym = symbol.upper()
        if sym in self._etf_symbols:
            return None  # ETFs do not have quarterly earnings reports

        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()

        # Check in-memory cache
        if sym in self._earnings_cache:
            cache_ts, cached_date = self._earnings_cache[sym]
            if (now_utc - cache_ts).total_seconds() < 3600:  # 1 hour cache
                return cached_date

        upcoming_date: Optional[date] = None
        try:
            ticker = yf.Ticker(sym)
            cal = ticker.calendar
            if isinstance(cal, dict):
                dates_list = cal.get("Earnings Date")
                if dates_list and isinstance(dates_list, (list, tuple)):
                    for d in dates_list:
                        candidate_date: Optional[date] = None
                        if isinstance(d, datetime):
                            candidate_date = d.date()
                        elif isinstance(d, date):
                            candidate_date = d

                        if candidate_date and candidate_date >= today:
                            if upcoming_date is None or candidate_date < upcoming_date:
                                upcoming_date = candidate_date

            self._earnings_cache[sym] = (now_utc, upcoming_date)
        except Exception as exc:
            logger.debug("Failed to retrieve earnings calendar for %s: %s", sym, exc)
            self._earnings_cache[sym] = (now_utc, None)

        return upcoming_date

    def check_earnings_risk(self, symbol: str) -> tuple[bool, Optional[str]]:
        """Check if symbol reports earnings within the next 2 business days.

        Args:
            symbol: Ticker symbol.

        Returns:
            Tuple of (earnings_safe: bool, reason: Optional[str])
        """
        sym = symbol.upper()
        if sym in self._etf_symbols:
            return True, None

        next_earnings = self.get_next_earnings_date(sym)
        if not next_earnings:
            return True, None

        today = datetime.now(timezone.utc).date()
        days_until = (next_earnings - today).days

        # Consider next 2 business days (up to 3 calendar days to bridge weekends)
        max_calendar_days = 4 if today.weekday() in (3, 4) else (self.earnings_window_days + 1)

        if 0 <= days_until <= max_calendar_days:
            reason = f"Upcoming earnings on {next_earnings.isoformat()} (in {days_until}d)"
            logger.info("[%s] Earnings risk detected: %s", sym, reason)
            return False, reason

        return True, f"Next earnings on {next_earnings.isoformat()}"

    def evaluate(self, symbol: str) -> NewsFilterResult:
        """Run macro calendar and earnings checks on target symbol.

        Args:
            symbol: Market ticker symbol.

        Returns:
            NewsFilterResult with overall safety and detailed diagnostics.
        """
        # 1. Macro risk check
        macro_safe, macro_desc = self.check_macro_risk()

        # 2. Earnings risk check
        earnings_safe, earnings_desc = self.check_earnings_risk(symbol)

        is_safe = macro_safe and earnings_safe

        reasons = []
        if not macro_safe:
            reasons.append(f"Blocked by Macro Risk ({macro_desc})")
        if not earnings_safe:
            reasons.append(f"Blocked by Earnings ({earnings_desc})")

        reason_str = "; ".join(reasons) if reasons else "News & Earnings Safe"

        next_earnings = self.get_next_earnings_date(symbol)
        earnings_date_str = next_earnings.isoformat() if next_earnings else None

        return NewsFilterResult(
            symbol=symbol.upper(),
            safe=is_safe,
            earnings_safe=earnings_safe,
            macro_safe=macro_safe,
            earnings_date=earnings_date_str,
            imminent_macro_event=macro_desc,
            reason=reason_str,
        )

    async def evaluate_async(self, symbol: str) -> NewsFilterResult:
        """Asynchronously run macro calendar and earnings checks on target symbol."""
        await self.fetch_macro_calendar_async()
        return self.evaluate(symbol)
