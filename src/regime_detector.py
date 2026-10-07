"""Market Regime Detection Engine (Regime-Switching Architecture).

Evaluates macro market structure (SPY benchmark) and individual equities/ETFs
in a daily (1D) timeframe to classify market condition into three distinct states:
1. BULL_TREND: Clear bullish trend (Price > EMA20 > EMA50, ATR Percentile <= 70%).
2. RANGE_CHOP: Range-bound / consolidation (lateral slopes or oscillating price, medium volatility).
3. HIGH_VOLATILITY_DEFENSIVE: Panic / high volatility / trend breakdown (Price < EMA200 or ATR Percentile >= 85%).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from models import MarketRegime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeAnalysis:
    """Detailed quantitative diagnostic output of market regime detection."""
    regime: MarketRegime
    symbol: str
    price: float
    ema_20: float
    ema_50: float
    ema_200: float
    ema20_slope: float
    ema50_slope: float
    atr_14: float
    atr_percentile: float
    adx_14: float
    mode_text: str
    display_badge: str
    risk_per_trade_pct: float
    risk_reward_ratio: float
    allow_entries: bool
    description: str


class MarketRegimeDetector:
    """Classifies market environment dynamically based on trend structure and volatility percentiles."""

    def __init__(
        self,
        volatility_lookback: int = 100,
        defensive_atr_percentile: float = 85.0,
        bull_atr_percentile: float = 70.0,
        slope_bars: int = 5,
    ) -> None:
        """Initialize regime detector parameters.

        Args:
            volatility_lookback: Historical window size (bars) for ATR percentile ranking (default: 100).
            defensive_atr_percentile: Threshold for defensive mode triggering (default: 85.0%).
            bull_atr_percentile: Maximum volatility percentile allowed for bull trend (default: 70.0%).
            slope_bars: Number of bars over which EMA slope is evaluated (default: 5).
        """
        self.volatility_lookback = volatility_lookback
        self.defensive_atr_percentile = defensive_atr_percentile
        self.bull_atr_percentile = bull_atr_percentile
        self.slope_bars = slope_bars

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute EMA(20, 50, 200), ATR(14), and Wilder's ADX(14).

        Args:
            df: Historical daily OHLCV DataFrame.

        Returns:
            pd.DataFrame: Enriched DataFrame with technical and directional columns.
        """
        if df.empty or len(df) < 5:
            return df.copy()

        out = df.copy()

        # 1. EMAs
        out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
        out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
        out["ema_200"] = out["close"].ewm(span=200, adjust=False).mean()

        # 2. Average True Range (Wilder's smoothing)
        prev_close = out["close"].shift(1)
        tr1 = out["high"] - out["low"]
        tr2 = (out["high"] - prev_close).abs()
        tr3 = (out["low"] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        out["atr_14"] = true_range.ewm(alpha=1.0 / 14.0, adjust=False).mean()

        # 3. Wilder's ADX(14) (Average Directional Movement Index)
        up_move = out["high"].diff()
        down_move = -out["low"].diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smooth +DM, -DM with Wilder's alpha = 1/14
        smooth_plus_dm = pd.Series(plus_dm, index=out.index).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        smooth_minus_dm = pd.Series(minus_dm, index=out.index).ewm(alpha=1.0 / 14.0, adjust=False).mean()

        safe_atr = out["atr_14"].replace(0.0, 1e-9)
        plus_di = 100.0 * (smooth_plus_dm / safe_atr)
        minus_di = 100.0 * (smooth_minus_dm / safe_atr)

        di_sum = (plus_di + minus_di).replace(0.0, 1e-9)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        out["adx_14"] = dx.ewm(alpha=1.0 / 14.0, adjust=False).mean()

        return out

    def compute_atr_percentile(self, atr_series: pd.Series) -> float:
        """Calculate historical percentile of current ATR over rolling window.

        Args:
            atr_series: Historical ATR(14) time series.

        Returns:
            float: Percentile score between 0.0 and 100.0.
        """
        if atr_series.empty:
            return 50.0

        window_size = min(self.volatility_lookback, len(atr_series))
        if window_size <= 1:
            return 50.0

        historical_window = atr_series.iloc[-window_size:]
        current_atr = float(atr_series.iloc[-1])

        # Fractional rank of current ATR within historical window
        percentile = float((historical_window < current_atr).mean() * 100.0)
        return round(percentile, 1)

    def compute_slopes(self, ema20: pd.Series, ema50: pd.Series) -> Tuple[float, float]:
        """Compute percentage rate of change (slope) over the slope_bars window.

        Args:
            ema20: EMA(20) series.
            ema50: EMA(50) series.

        Returns:
            Tuple of (ema20_slope, ema50_slope) as fractional returns over the window.
        """
        bars = min(self.slope_bars, len(ema20) - 1)
        if bars < 1:
            return 0.0, 0.0

        prev_ema20 = float(ema20.iloc[-bars - 1])
        prev_ema50 = float(ema50.iloc[-bars - 1])
        curr_ema20 = float(ema20.iloc[-1])
        curr_ema50 = float(ema50.iloc[-1])

        slope_20 = (curr_ema20 - prev_ema20) / (prev_ema20 if prev_ema20 != 0 else 1.0)
        slope_50 = (curr_ema50 - prev_ema50) / (prev_ema50 if prev_ema50 != 0 else 1.0)

        return slope_20, slope_50

    def analyze(self, df: pd.DataFrame, symbol: str = "SPY") -> RegimeAnalysis:
        """Analyze market data for symbol/benchmark and classify into unique regime.

        Classification Logic:
        1. HIGH_VOLATILITY_DEFENSIVE (Priority 1):
           - Price < EMA200 OR ATR Percentile >= 85.0%
           -> Risk = 0.0%, R:R = 0:1, allow_entries = False
        2. BULL_TREND (Priority 2):
           - Price > EMA20 > EMA50 AND ATR Percentile <= 70.0%
           -> Risk = 1.25%, R:R = 3:1, allow_entries = True
        3. RANGE_CHOP (Priority 3):
           - Lateral slopes or Price oscillating between EMA20/EMA50 with medium volatility
           -> Risk = 0.50%, R:R = 1.5:1, allow_entries = True

        Args:
            df: Historical daily OHLCV DataFrame.
            symbol: Ticker identifier (e.g. 'SPY').

        Returns:
            RegimeAnalysis: Comprehensive diagnosis of regime state and adaptive parameters.
        """
        if df is None or len(df) < 15:
            return RegimeAnalysis(
                regime=MarketRegime.RANGE_CHOP,
                symbol=symbol,
                price=0.0,
                ema_20=0.0,
                ema_50=0.0,
                ema_200=0.0,
                ema20_slope=0.0,
                ema50_slope=0.0,
                atr_14=0.0,
                atr_percentile=50.0,
                adx_14=0.0,
                mode_text="CONSERVADOR (0.5%)",
                display_badge="RANGE [CONSERVADOR (0.5%)]",
                risk_per_trade_pct=0.0050,
                risk_reward_ratio=1.5,
                allow_entries=True,
                description="Insufficient history for full regime classification. Defaulting to RANGE_CHOP.",
            )

        ind_df = self.calculate_indicators(df)
        curr = ind_df.iloc[-1]

        price = float(curr["close"])
        ema_20 = float(curr["ema_20"])
        ema_50 = float(curr["ema_50"])
        ema_200 = float(curr["ema_200"])
        atr_14 = float(curr["atr_14"])
        adx_14 = float(curr["adx_14"])

        atr_percentile = self.compute_atr_percentile(ind_df["atr_14"])
        ema20_slope, ema50_slope = self.compute_slopes(ind_df["ema_20"], ind_df["ema_50"])

        # -------------------------------------------------------------
        # Condition 1: HIGH_VOLATILITY_DEFENSIVE (Panic / Trend Broken)
        # Price < EMA200 OR ATR Percentile >= 85.0%
        # -------------------------------------------------------------
        is_below_ema200 = price < ema_200
        is_extreme_volatility = atr_percentile >= self.defensive_atr_percentile

        if is_below_ema200 or is_extreme_volatility:
            reasons = []
            if is_below_ema200:
                reasons.append(f"Price (${price:.2f}) < EMA200 (${ema_200:.2f})")
            if is_extreme_volatility:
                reasons.append(f"ATR Volatility Percentile ({atr_percentile:.1f}%) >= {self.defensive_atr_percentile:.1f}%")

            desc = f"DEFENSIVE MODE: {'; '.join(reasons)}. New entries halted for capital protection."
            return RegimeAnalysis(
                regime=MarketRegime.HIGH_VOLATILITY_DEFENSIVE,
                symbol=symbol,
                price=price,
                ema_20=ema_20,
                ema_50=ema_50,
                ema_200=ema_200,
                ema20_slope=ema20_slope,
                ema50_slope=ema50_slope,
                atr_14=atr_14,
                atr_percentile=atr_percentile,
                adx_14=adx_14,
                mode_text="DEFENSIVE [OFF]",
                display_badge="DEFENSIVE [OFF]",
                risk_per_trade_pct=0.0,
                risk_reward_ratio=0.0,
                allow_entries=False,
                description=desc,
            )

        # -------------------------------------------------------------
        # Condition 2: BULL_TREND (Clear Bullish Trend)
        # Price > EMA20 > EMA50 AND ATR Percentile <= 70.0%
        # -------------------------------------------------------------
        is_bull_stacked = (price > ema_20) and (ema_20 > ema_50)
        is_volatility_controlled = atr_percentile <= self.bull_atr_percentile

        if is_bull_stacked and is_volatility_controlled:
            desc = (
                f"BULL TREND CONFIRMED: Price (${price:.2f}) > EMA20 (${ema_20:.2f}) > EMA50 (${ema_50:.2f}), "
                f"ATR Volatility Percentile at {atr_percentile:.1f}% (<= {self.bull_atr_percentile:.1f}%), "
                f"ADX={adx_14:.1f}. Aggressive position sizing unlocked."
            )
            return RegimeAnalysis(
                regime=MarketRegime.BULL_TREND,
                symbol=symbol,
                price=price,
                ema_20=ema_20,
                ema_50=ema_50,
                ema_200=ema_200,
                ema20_slope=ema20_slope,
                ema50_slope=ema50_slope,
                atr_14=atr_14,
                atr_percentile=atr_percentile,
                adx_14=adx_14,
                mode_text="AGRESIVO (1.25%)",
                display_badge="BULL TREND [AGRESIVO (1.25%)]",
                risk_per_trade_pct=0.0125,
                risk_reward_ratio=3.0,
                allow_entries=True,
                description=desc,
            )

        # -------------------------------------------------------------
        # Condition 3: RANGE_CHOP (Consolidation / Lateral Movement)
        # Price oscillating between EMAs or lateral slope with medium volatility
        # -------------------------------------------------------------
        reasons = []
        if not is_bull_stacked:
            reasons.append("EMAs not in bull sequence (Price oscillating or consolidating)")
        if not is_volatility_controlled:
            reasons.append(f"Elevated ATR Volatility ({atr_percentile:.1f}% > {self.bull_atr_percentile:.1f}%)")

        desc = (
            f"RANGE / CHOP MARKET: {'; '.join(reasons)}. "
            f"Conservative position sizing active."
        )
        return RegimeAnalysis(
            regime=MarketRegime.RANGE_CHOP,
            symbol=symbol,
            price=price,
            ema_20=ema_20,
            ema_50=ema_50,
            ema_200=ema_200,
            ema20_slope=ema20_slope,
            ema50_slope=ema50_slope,
            atr_14=atr_14,
            atr_percentile=atr_percentile,
            adx_14=adx_14,
            mode_text="CONSERVADOR (0.5%)",
            display_badge="RANGE [CONSERVADOR (0.5%)]",
            risk_per_trade_pct=0.0050,
            risk_reward_ratio=1.5,
            allow_entries=True,
            description=desc,
        )
