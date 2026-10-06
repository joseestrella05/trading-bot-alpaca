"""FastAPI Application providing REST API and Dashboard for the Trading Bot.

Manages background scheduling, real-time portfolio monitoring, manual scanning triggers,
and log streaming for the Web Dashboard.
"""

from __future__ import annotations

import asyncio
import collections
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os
from typing import Any, Deque, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from config import AppConfig, load_config
from main import SwingTradingBot
from web.scheduler import BotScheduler

# ---------------------------------------------------------------------------
# In-Memory Circular Log Buffer for Live Dashboard Terminal
# ---------------------------------------------------------------------------
MAX_LOG_RECORDS = 250
LOG_BUFFER: Deque[Dict[str, str]] = collections.deque(maxlen=MAX_LOG_RECORDS)


class DashboardLogHandler(logging.Handler):
    """Custom logging handler capturing formatted log events into a deque."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            entry = {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S"),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
                "formatted": msg,
            }
            LOG_BUFFER.append(entry)
        except Exception:
            self.handleError(record)


# Configure logging capture
dash_handler = DashboardLogHandler()
dash_handler.setLevel(logging.INFO)
dash_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(dash_handler)

# Also explicitly attach to specific app loggers to ensure capture
for logger_name in ("TradingBot", "Scheduler", "API", "WebLifespan", "data_provider", "execution", "strategy", "risk_manager"):
    l = logging.getLogger(logger_name)
    l.setLevel(logging.INFO)
    l.addHandler(dash_handler)

# ---------------------------------------------------------------------------
# Global State Singletons
# ---------------------------------------------------------------------------
bot_instance: Optional[SwingTradingBot] = None
scheduler_instance: Optional[BotScheduler] = None
app_config: Optional[AppConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI starting the Bot and Background Scheduler."""
    global bot_instance, scheduler_instance, app_config

    logger = logging.getLogger("WebLifespan")
    logger.info("Initializing Trading Bot and Web Dashboard...")

    # Load configuration
    app_config = load_config()

    # Initialize Bot
    bot_instance = SwingTradingBot(app_config)

    # Initialize & Start Scheduler
    # Monday to Friday at 3:50 PM EST (10 min before close)
    interval_env = os.getenv("SCHEDULER_INTERVAL_MINUTES")
    interval_mins = int(interval_env) if interval_env and interval_env.isdigit() else None

    scheduler_instance = BotScheduler(
        scan_job_func=lambda: bot_instance.run_scan(),
        cron_hour=15,
        cron_minute=50,
        interval_minutes=interval_mins,
    )
    scheduler_instance.start()

    logger.info("Bot and Scheduler online. Next scan: %s", scheduler_instance.get_next_run_time())

    yield

    # Teardown
    logger.info("Shutting down Scheduler...")
    if scheduler_instance:
        scheduler_instance.shutdown()
    logger.info("Dashboard shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI Application & Templates
# ---------------------------------------------------------------------------
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

app = FastAPI(
    title="Alpaca Swing Trading Dashboard",
    description="Quantitative Swing Trading Bot Control Panel & Live Monitor",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dashboard Route
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Render the primary Bloomberg/TradingView style dark dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "Alpaca Swing Trading Bot | Terminal",
            "paper_mode": app_config.alpaca.paper if app_config else True,
            "symbols": app_config.symbols if app_config else [],
        },
    )


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_status():
    """Retrieve bot operational status, Alpaca balances, and open positions."""
    if not bot_instance or not app_config:
        return JSONResponse(status_code=503, content={"error": "Bot subsystem not initialized"})

    # Fetch live account state from Alpaca
    try:
        account_state = await asyncio.to_thread(bot_instance.execution_handler.get_account_state)
    except Exception as exc:
        logging.getLogger("API").error("Failed to query account state: %s", exc)
        return JSONResponse(status_code=500, content={"error": f"Brokerage API error: {exc}"})

    # Fetch open positions with attached Stop Loss and Take Profit levels
    try:
        positions = await asyncio.to_thread(bot_instance.execution_handler.get_positions_with_bracket_legs)
    except Exception as exc:
        logging.getLogger("API").error("Failed to query positions: %s", exc)
        positions = []

    # Calculate intraday PnL
    intraday_pnl = account_state.equity - account_state.last_equity

    # Scheduler info
    next_run = scheduler_instance.get_next_run_time() if scheduler_instance else None

    return {
        "is_active": bot_instance.is_active,
        "is_scanning": bot_instance.is_scanning,
        "paper_mode": app_config.alpaca.paper,
        "data_feed": app_config.alpaca.data_feed,
        "last_scan_time": bot_instance.last_scan_time.isoformat() if bot_instance.last_scan_time else None,
        "next_scan_time": next_run.isoformat() if next_run else None,
        "monitored_symbols": app_config.symbols,
        "account": {
            "equity": account_state.equity,
            "last_equity": account_state.last_equity,
            "cash": account_state.cash,
            "buying_power": account_state.buying_power,
            "intraday_pnl": intraday_pnl,
            "intraday_drawdown_pct": round(account_state.intraday_drawdown_pct * 100, 2),
            "kill_switch_active": account_state.kill_switch_active,
            "currency": account_state.currency,
        },
        "risk_limits": {
            "max_risk_per_trade_pct": app_config.risk.max_risk_per_trade_pct * 100,
            "max_intraday_drawdown_pct": app_config.risk.max_intraday_drawdown_pct * 100,
            "risk_reward_ratio": app_config.risk.risk_reward_ratio,
        },
        "positions": positions,
        "positions_count": len(positions),
    }


@app.get("/api/signals")
async def get_signals():
    """Retrieve results and technical indicators from the most recent scan."""
    if not bot_instance:
        return JSONResponse(status_code=503, content={"error": "Bot subsystem not initialized"})

    return {
        "last_scan_time": bot_instance.last_scan_time.isoformat() if bot_instance.last_scan_time else None,
        "is_scanning": bot_instance.is_scanning,
        "signals_count": len(bot_instance.last_results),
        "results": bot_instance.last_results,
    }


@app.post("/api/scan-now")
async def trigger_scan_now():
    """Immediately trigger a market scan cycle across monitored tickers."""
    if not bot_instance:
        return JSONResponse(status_code=503, content={"error": "Bot subsystem not initialized"})

    if bot_instance.is_scanning:
        return JSONResponse(
            status_code=409,
            content={"status": "busy", "message": "A scan is already in progress. Please wait."},
        )

    logging.getLogger("API").info("Manual scan triggered via Web API.")

    # Execute scan in background thread to avoid blocking FastAPI event loop
    result = await asyncio.to_thread(bot_instance.run_scan)

    return {
        "status": "completed",
        "aborted": result.get("aborted", False),
        "reason": result.get("reason"),
        "scan_time": bot_instance.last_scan_time.isoformat() if bot_instance.last_scan_time else None,
        "results": bot_instance.last_results,
    }


@app.post("/api/toggle-bot")
async def toggle_bot_state():
    """Toggle the bot between Active and Paused operational states."""
    if not bot_instance:
        return JSONResponse(status_code=503, content={"error": "Bot subsystem not initialized"})

    bot_instance.is_active = not bot_instance.is_active
    new_state = bot_instance.is_active

    state_str = "ACTIVE (ONLINE)" if new_state else "PAUSED (OFFLINE)"
    logging.getLogger("API").info("Bot operational state changed to: %s", state_str)

    return {
        "is_active": new_state,
        "status_text": state_str,
        "message": f"Bot is now {state_str}.",
    }


@app.get("/api/logs")
async def get_logs():
    """Retrieve recent in-memory log entries for live terminal viewer."""
    return {"logs": list(LOG_BUFFER)}


@app.get("/api/jobs")
async def get_jobs():
    """Retrieve scheduled background jobs information."""
    if not scheduler_instance:
        return {"jobs": []}
    return {"jobs": scheduler_instance.get_jobs_info()}
