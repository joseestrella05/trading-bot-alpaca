"""Market Regime Detection Engine (Regime-Switching Architecture).

Evaluates macro market structure (SPY benchmark) and individual equities/ETFs
in a daily (1D) timeframe and assigns a continuous risk multiplier (0.0 - 1.25) applied
to the base risk per trade:
1. HALT (0.00): ATR(14) percentile over 100 bars > 88 or severe 5-bar drawdown -> no new entries.
2. DEFENSIVE ACTIVE (0.40): ATR percentile in (70, 88] or price < EMA200 -> small size only.
3. BULL_TREND (1.25): ADX >= 25, EMA20 > EMA50 > EMA200 with rising EMA50.
4. RANGE_CHOP (0.75): everything else (normal volatility, no clear trend).
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
    risk_multiplier: float = 1.0


class MarketRegimeDetector:
    """Classifies market environment dynamically based on trend structure and volatility percentiles."""

    def __init__(
        self,
        volatility_lookback: int = 100,
        defensive_atr_percentile: float = 88.0,
        bull_atr_percentile: float = 70.0,
        slope_bars: int = 5,
        bull_adx_threshold: float = 25.0,
        panic_drawdown_pct: float = 0.06,
        panic_lookback_bars: int = 5,
        base_risk_pct: float = 0.01,
    ) -> None:
        """Initialize regime detector parameters.

        Args:
            volatility_lookback: Historical window size (bars) for ATR percentile ranking (default: 100).
            defensive_atr_percentile: ATR percentile above which entries halt (default: 88.0%).
            bull_atr_percentile: ATR percentile above which sizing drops to defensive 0.40 (default: 70.0%).
            slope_bars: Number of bars over which EMA slope is evaluated (default: 5).
            bull_adx_threshold: Minimum ADX(14) for a clear bull trend (default: 25).
            panic_drawdown_pct: Drawdown from the highest close of the last N bars that halts entries (6%).
            panic_lookback_bars: N bars for the panic drawdown check (default: 5).
            base_risk_pct: Base risk per trade the multiplier scales; only used for display text.
        """
        self.volatility_lookback = volatility_lookback
        self.defensive_atr_percentile = defensive_atr_percentile
        self.bull_atr_percentile = bull_atr_percentile
        self.slope_bars = slope_bars
        self.bull_adx_threshold = bull_adx_threshold
        self.panic_drawdown_pct = panic_drawdown_pct
        self.panic_lookback_bars = panic_lookback_bars
        self.base_risk_pct = base_risk_pct

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
        """Classify the regime and assign a continuous risk multiplier (first match wins).

        1. HALT (0.00): ATR percentile > defensive_atr_percentile (88) OR drawdown from the
           highest close of the last panic_lookback_bars >= panic_drawdown_pct.
        2. DEFENSIVE ACTIVE (0.40): ATR percentile > bull_atr_percentile (70) OR price < EMA200.
        3. BULL_TREND (1.25): ADX >= 25 AND EMA20 > EMA50 > EMA200 AND EMA50 slope > 0.
        4. RANGE_CHOP (0.75): otherwise.

        Args:
            df: Historical daily OHLCV DataFrame.
            symbol: Ticker identifier (e.g. 'SPY').

        Returns:
            RegimeAnalysis: Regime state, risk multiplier and adaptive parameters.
        """
        if df is None or len(df) < 15:
            return self._build(
                MarketRegime.RANGE_CHOP, 0.75, symbol,
                "Insufficient history for full regime classification. Defaulting to RANGE_CHOP.",
                price=0.0, ema_20=0.0, ema_50=0.0, ema_200=0.0, ema20_slope=0.0, ema50_slope=0.0,
                atr_14=0.0, atr_percentile=50.0, adx_14=0.0,
            )

        ind_df = self.calculate_indicators(df)
        curr = ind_df.iloc[-1]
        price = float(curr["close"])
        ema_20 = float(curr["ema_20"])
        ema_50 = float(curr["ema_50"])
        ema_200 = float(curr["ema_200"])
        adx_14 = float(curr["adx_14"])
        atr_percentile = self.compute_atr_percentile(ind_df["atr_14"])
        ema20_slope, ema50_slope = self.compute_slopes(ind_df["ema_20"], ind_df["ema_50"])
        recent_high = float(ind_df["close"].iloc[-(self.panic_lookback_bars + 1):].max())
        drawdown = 1.0 - price / recent_high if recent_high > 0 else 0.0

        metrics = dict(
            price=price, ema_20=ema_20, ema_50=ema_50, ema_200=ema_200,
            ema20_slope=ema20_slope, ema50_slope=ema50_slope,
            atr_14=float(curr["atr_14"]), atr_percentile=atr_percentile, adx_14=adx_14,
        )

        halt_reasons = []
        if atr_percentile > self.defensive_atr_percentile:
            halt_reasons.append(f"ATR percentile {atr_percentile:.1f}% > {self.defensive_atr_percentile:.1f}%")
        if drawdown >= self.panic_drawdown_pct:
            halt_reasons.append(
                f"{self.panic_lookback_bars}-bar drawdown {drawdown*100:.1f}% >= {self.panic_drawdown_pct*100:.1f}%"
            )
        if halt_reasons:
            return self._build(
                MarketRegime.HIGH_VOLATILITY_DEFENSIVE, 0.0, symbol,
                f"REGIME HALT: {'; '.join(halt_reasons)}. New entries blocked.", **metrics,
            )

        defensive_reasons = []
        if atr_percentile > self.bull_atr_percentile:
            defensive_reasons.append(f"Elevated ATR percentile {atr_percentile:.1f}% > {self.bull_atr_percentile:.1f}%")
        if price < ema_200:
            defensive_reasons.append(f"Price (${price:.2f}) < EMA200 (${ema_200:.2f})")
        if defensive_reasons:
            return self._build(
                MarketRegime.HIGH_VOLATILITY_DEFENSIVE, 0.40, symbol,
                f"DEFENSIVE ACTIVE: {'; '.join(defensive_reasons)}. Reduced size for high-quality setups only.",
                **metrics,
            )

        if adx_14 >= self.bull_adx_threshold and ema_20 > ema_50 > ema_200 and ema50_slope > 0:
            return self._build(
                MarketRegime.BULL_TREND, 1.25, symbol,
                f"BULL TREND CONFIRMED: EMA20 (${ema_20:.2f}) > EMA50 (${ema_50:.2f}) > EMA200 (${ema_200:.2f}), "
                f"ADX={adx_14:.1f}, ATR percentile {atr_percentile:.1f}%. Aggressive sizing unlocked.",
                **metrics,
            )

        return self._build(
            MarketRegime.RANGE_CHOP, 0.75, symbol,
            f"RANGE / CHOP: no clear trend (ADX={adx_14:.1f}, EMA sequence not bullish or EMA50 flat), "
            f"ATR percentile {atr_percentile:.1f}%. Moderate sizing.",
            **metrics,
        )

    def _build(
        self, regime: MarketRegime, multiplier: float, symbol: str, description: str, **metrics: float
    ) -> RegimeAnalysis:
        """Assemble RegimeAnalysis with mode labels derived from the multiplier."""
        risk_pct = self.base_risk_pct * multiplier
        label, mode, rr = {
            MarketRegime.BULL_TREND: ("BULL TREND", "AGRESIVO", 3.0),
            MarketRegime.RANGE_CHOP: ("RANGE", "MODERADO", 1.5),
            MarketRegime.HIGH_VOLATILITY_DEFENSIVE: ("DEFENSIVE", "DEFENSIVO ACTIVO", 2.0),
        }[regime]
        if multiplier <= 0.0:
            label, mode, rr = "HALT", "HALT", 0.0
        return RegimeAnalysis(
            regime=regime,
            symbol=symbol,
            mode_text=f"{mode} ({risk_pct*100:.2f}%)",
            display_badge=f"{label} [{risk_pct*100:.2f}%]",
            risk_per_trade_pct=risk_pct,
            risk_reward_ratio=rr,
            allow_entries=multiplier > 0.0,
            description=description,
            risk_multiplier=multiplier,
            **metrics,
        )
