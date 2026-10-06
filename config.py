"""Configuration management for the quantitative trading bot.

Loads environment variables from .env and provides strongly typed dataclasses
for Alpaca API credentials, risk parameters, strategy hyper-parameters, and
execution policies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Automatically load .env file from the current working directory
load_dotenv()


@dataclass(frozen=True)
class AlpacaConfig:
    """Credentials and connection settings for Alpaca Markets."""
    api_key: str
    secret_key: str
    paper: bool = True
    data_feed: str = "iex"  # "iex" for free tier / paper; "sip" for paid market data subscription


@dataclass(frozen=True)
class RiskConfig:
    """Quantitative risk management parameters.

    Attributes:
        max_risk_per_trade_pct: Maximum portfolio fraction risked per trade (0.01 = 1.0%).
        max_intraday_drawdown_pct: Kill-switch threshold (0.025 = 2.5% intraday drawdown).
        risk_reward_ratio: Target reward-to-risk ratio (2.0 = 2:1).
        atr_sl_multiplier: ATR multiplier for stop loss placement (1.5 * ATR(14)).
        max_position_equity_pct: Ceiling for position gross notional as % of portfolio.
        allow_fractional: Bracket orders in Alpaca require whole shares (False).
    """
    max_risk_per_trade_pct: float = 0.01
    max_intraday_drawdown_pct: float = 0.025
    risk_reward_ratio: float = 2.0
    atr_sl_multiplier: float = 1.5
    max_position_equity_pct: float = 0.30
    allow_fractional: bool = False


@dataclass(frozen=True)
class StrategyConfig:
    """Technical parameters for the Swing Trend Following Strategy (1D).

    Attributes:
        ema_fast: Fast EMA period (Pullback support level).
        ema_medium: Medium EMA period (Trend alignment filter).
        ema_slow: Slow EMA period (Macro trend filter).
        atr_period: Volatility period for ATR.
        rsi_period: Momentum period for RSI.
        rsi_threshold: Level above which RSI must bounce (40.0).
        rsi_zone_upper: Upper threshold for the recent pullback zone (e.g., 45.0).
        rsi_lookback: Number of bars evaluated for RSI bounce detection.
        pullback_lookback: Lookback window in bars to confirm EMA(20) test.
        pullback_tolerance_pct: Price tolerance percentage when testing EMA(20).
    """
    ema_fast: int = 20
    ema_medium: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    rsi_period: int = 14
    rsi_threshold: float = 40.0
    rsi_zone_upper: float = 45.0
    rsi_lookback: int = 3
    pullback_lookback: int = 3
    pullback_tolerance_pct: float = 0.01


@dataclass(frozen=True)
class ExecutionConfig:
    """Settings controlling order routing and bracket order dispatch."""
    entry_order_type: str = "market"  # "market" or "limit"
    time_in_force: str = "gtc"       # "gtc" or "day"
    dry_run: bool = False            # When True, signals are computed but no orders sent


@dataclass(frozen=True)
class AppConfig:
    """Root configuration container for the trading system."""
    alpaca: AlpacaConfig
    risk: RiskConfig
    strategy: StrategyConfig
    execution: ExecutionConfig
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"])
    log_level: str = "INFO"


def load_config() -> AppConfig:
    """Load configuration from environment variables with production defaults.

    Returns:
        AppConfig: Fully initialized and validated configuration object.

    Raises:
        ValueError: If essential Alpaca credentials are not configured.
    """
    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
    paper_str = os.getenv("ALPACA_PAPER", "True").strip().lower()
    paper = paper_str in ("true", "1", "yes")
    data_feed = os.getenv("ALPACA_DATA_FEED", "iex").strip().lower()

    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca credentials missing! Please configure ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY in your .env file or environment variables."
        )

    # Configurable symbols via environment variable SYMBOLS="SPY,AAPL,QQQ"
    symbols_env = os.getenv("SYMBOLS", "")
    if symbols_env:
        symbols = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
    else:
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

    # Parse dry-run mode
    dry_run_env = os.getenv("DRY_RUN", "False").strip().lower()
    dry_run = dry_run_env in ("true", "1", "yes")

    alpaca_cfg = AlpacaConfig(
        api_key=api_key,
        secret_key=secret_key,
        paper=paper,
        data_feed=data_feed,
    )

    risk_cfg = RiskConfig(
        max_risk_per_trade_pct=float(os.getenv("MAX_RISK_PER_TRADE_PCT", "0.01")),
        max_intraday_drawdown_pct=float(os.getenv("MAX_INTRADAY_DRAWDOWN_PCT", "0.025")),
        risk_reward_ratio=float(os.getenv("RISK_REWARD_RATIO", "2.0")),
        atr_sl_multiplier=float(os.getenv("ATR_SL_MULTIPLIER", "1.5")),
        max_position_equity_pct=float(os.getenv("MAX_POSITION_EQUITY_PCT", "0.30")),
        allow_fractional=os.getenv("ALLOW_FRACTIONAL", "False").strip().lower() in ("true", "1"),
    )

    strategy_cfg = StrategyConfig(
        ema_fast=int(os.getenv("EMA_FAST", "20")),
        ema_medium=int(os.getenv("EMA_MEDIUM", "50")),
        ema_slow=int(os.getenv("EMA_SLOW", "200")),
        atr_period=int(os.getenv("ATR_PERIOD", "14")),
        rsi_period=int(os.getenv("RSI_PERIOD", "14")),
        rsi_threshold=float(os.getenv("RSI_THRESHOLD", "40.0")),
    )

    execution_cfg = ExecutionConfig(
        entry_order_type=os.getenv("ENTRY_ORDER_TYPE", "market").strip().lower(),
        time_in_force=os.getenv("TIME_IN_FORCE", "gtc").strip().lower(),
        dry_run=dry_run,
    )

    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()

    return AppConfig(
        alpaca=alpaca_cfg,
        risk=risk_cfg,
        strategy=strategy_cfg,
        execution=execution_cfg,
        symbols=symbols,
        log_level=log_level,
    )
