"""Domain models and data structures for the quantitative swing trading system.

This module defines typed dataclasses for signals, risk assessments,
account states, and order execution results to ensure strict contract compliance
between system layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class SignalType(str, Enum):
    """Enumeration of possible strategy trading signals."""
    BUY = "BUY"
    HOLD = "HOLD"


@dataclass(frozen=True)
class TradingSignal:
    """Represents a trading signal produced by a quantitative strategy.

    Attributes:
        symbol: The market ticker symbol (e.g., SPY, AAPL).
        signal_type: Strategy directive (BUY or HOLD).
        timestamp: Time of the evaluated bar.
        close_price: Latest bar close price.
        entry_price: Target execution price for entry.
        ema_20: Current Exponential Moving Average (20 periods).
        ema_50: Current Exponential Moving Average (50 periods).
        ema_200: Current Exponential Moving Average (200 periods).
        rsi_14: Current Relative Strength Index (14 periods).
        atr_14: Current Average True Range (14 periods).
        prev_high: High of the previous bar (breakout trigger level).
        reason: Diagnostic explanation of the signal.
        metrics: Dictionary containing additional debug or indicator values.
    """
    symbol: str
    signal_type: SignalType
    timestamp: datetime
    close_price: float
    entry_price: float
    ema_20: float
    ema_50: float
    ema_200: float
    rsi_14: float
    atr_14: float
    prev_high: float
    reason: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionSize:
    """Computed sizing parameters and bracket order levels for an execution.

    Attributes:
        symbol: Market ticker symbol.
        shares: Allocated number of shares (integer or fractional).
        is_fractional: Flag indicating if shares contains decimal precision.
        entry_price: Desired entry price.
        stop_loss_price: Absolute price for Stop Loss leg.
        take_profit_price: Absolute price for Take Profit leg.
        risk_amount_usd: Maximum dollar capital placed at risk (1.0% of equity).
        risk_per_share: Absolute risk per share (entry_price - stop_loss_price).
        reward_per_share: Absolute reward per share (take_profit_price - entry_price).
        risk_reward_ratio: Calculated reward-to-risk ratio (target >= 2.0).
        total_exposure_usd: Total capital required for entry (shares * entry_price).
        is_valid: True if trade sizing satisfies all risk constraints.
        rejection_reason: Descriptive message if sizing is invalid.
    """
    symbol: str
    shares: float
    is_fractional: bool
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_amount_usd: float
    risk_per_share: float
    reward_per_share: float
    risk_reward_ratio: float
    total_exposure_usd: float
    is_valid: bool
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class AccountState:
    """Current financial state of the Alpaca brokerage account.

    Attributes:
        equity: Real-time total portfolio equity.
        last_equity: Portfolio equity at previous market close.
        cash: Current uninvested cash balance.
        buying_power: Available buying power for trading.
        intraday_drawdown_pct: Fractional intraday drawdown ((last_equity - equity) / last_equity).
        kill_switch_active: True if intraday drawdown exceeds threshold (e.g., 2.5%).
        currency: Account currency code.
    """
    equity: float
    last_equity: float
    cash: float
    buying_power: float
    intraday_drawdown_pct: float
    kill_switch_active: bool
    currency: str = "USD"


@dataclass(frozen=True)
class ExecutionResult:
    """Structured report of an order dispatch to Alpaca.

    Attributes:
        symbol: Market ticker symbol.
        success: True if the order was accepted by Alpaca.
        order_id: Brokerage assigned order UUID if accepted.
        client_order_id: Idempotency client identifier.
        status: Brokerage order status string.
        submitted_qty: Number of shares submitted in the order.
        error_code: Error code returned by API if rejected.
        error_message: Detailed rejection or exception explanation.
        raw_response: Full payload returned by brokerage client.
    """
    symbol: str
    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: str = "PENDING"
    submitted_qty: float = 0.0
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
