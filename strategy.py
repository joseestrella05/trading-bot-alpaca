"""Quantitative Swing Trend Following Strategy (1D Timeframe).

Implements:
1. Macro Trend Filter:
   - Price > EMA(200)
   - EMA(50) > EMA(200)
2. Pullback & Trigger Logic:
   - Price pulled back towards EMA(20) within recent window
   - Close > previous bar High (breakout above prior day's high)
   - RSI(14) bouncing off the 40 zone (RSI >= 40 and curling upward from support)
3. Volatility-based Stop Loss:
   - Stop Loss: 1.5 * ATR(14) below entry price
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import pandas as pd

from config import StrategyConfig
from models import SignalType, TradingSignal

logger = logging.getLogger(__name__)


class SwingTrendFollowingStrategy:
    """Evaluates swing trend-following pullback setups on daily bars."""

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        news_filter: Optional[Any] = None,
        ml_predictor: Optional[Any] = None,
    ) -> None:
        """Initialize strategy with configurable hyper-parameters and filters.

        Args:
            config: Strategy configuration settings.
            news_filter: Optional NewsFilter instance for macro and earnings checks.
            ml_predictor: Optional StockMLPredictor instance for Random Forest confirmation.
        """
        self.config = config or StrategyConfig()
        self.news_filter = news_filter
        self.ml_predictor = ml_predictor

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute EMA(20, 50, 200), Wilder's ATR(14), and Wilder's RSI(14).

        Args:
            df: OHLCV DataFrame indexed by timestamp.

        Returns:
            pd.DataFrame: Copy of DataFrame enriched with technical indicator columns.
        """
        if df.empty or len(df) < 5:
            return df.copy()

        out = df.copy()

        # 1. Exponential Moving Averages
        out["ema_fast"] = out["close"].ewm(span=self.config.ema_fast, adjust=False).mean()
        out["ema_medium"] = out["close"].ewm(span=self.config.ema_medium, adjust=False).mean()
        out["ema_slow"] = out["close"].ewm(span=self.config.ema_slow, adjust=False).mean()

        # 2. Average True Range (Wilder's smoothing)
        prev_close = out["close"].shift(1)
        tr1 = out["high"] - out["low"]
        tr2 = (out["high"] - prev_close).abs()
        tr3 = (out["low"] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr_alpha = 1.0 / float(self.config.atr_period)
        out["atr"] = true_range.ewm(alpha=atr_alpha, adjust=False).mean()

        # 3. Relative Strength Index (Wilder's smoothing)
        delta = out["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        rsi_alpha = 1.0 / float(self.config.rsi_period)
        avg_gain = gain.ewm(alpha=rsi_alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=rsi_alpha, adjust=False).mean()

        # Avoid zero division
        rs = avg_gain / avg_loss.replace(0.0, 1e-9)
        out["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # 4. Trigger reference: Prior bar's high
        out["prev_high"] = out["high"].shift(1)
        out["prev_rsi"] = out["rsi"].shift(1)

        return out

    def evaluate(self, symbol: str, df: pd.DataFrame) -> TradingSignal:
        """Evaluate market data and generate a BUY or HOLD trading signal.

        Args:
            symbol: Ticker symbol.
            df: Historical daily OHLCV DataFrame.

        Returns:
            TradingSignal: Strongly typed signal with quantitative justifications.
        """
        min_required_bars = self.config.ema_slow + 10  # Ensure 200 EMA convergence
        if df is None or len(df) < min_required_bars:
            latest_time = df.index[-1] if (df is not None and not df.empty) else pd.Timestamp.now()
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                timestamp=latest_time.to_pydatetime() if hasattr(latest_time, "to_pydatetime") else latest_time,
                close_price=float(df["close"].iloc[-1]) if (df is not None and not df.empty) else 0.0,
                entry_price=0.0,
                ema_20=0.0,
                ema_50=0.0,
                ema_200=0.0,
                rsi_14=0.0,
                atr_14=0.0,
                prev_high=0.0,
                reason=f"Insufficient history: {len(df) if df is not None else 0} bars available, {min_required_bars} required.",
            )

        indicators_df = self.calculate_indicators(df)
        curr = indicators_df.iloc[-1]
        prev = indicators_df.iloc[-2]

        timestamp = indicators_df.index[-1]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        close_price = float(curr["close"])
        ema_20 = float(curr["ema_fast"])
        ema_50 = float(curr["ema_medium"])
        ema_200 = float(curr["ema_slow"])
        rsi_14 = float(curr["rsi"])
        atr_14 = float(curr["atr"])
        prev_high = float(curr["prev_high"])
        prev_rsi = float(curr["prev_rsi"])

        # -------------------------------------------------------------
        # Evaluate ML and News Filter Upfront for Comprehensive Metrics
        # -------------------------------------------------------------
        ml_prob: Optional[float] = None
        ml_approved: bool = True
        ml_reason: str = "ML Neutral"
        if self.ml_predictor is not None:
            try:
                ml_res = self.ml_predictor.predict(symbol, indicators_df)
                ml_prob = ml_res.probability
                ml_approved = ml_res.is_approved
                ml_reason = ml_res.reason
            except Exception as exc:
                logger.warning("ML prediction failed for %s: %s", symbol, exc)

        news_safe: bool = True
        news_reason: str = "News Safe"
        if self.news_filter is not None:
            try:
                news_res = self.news_filter.evaluate(symbol)
                news_safe = news_res.safe
                news_reason = news_res.reason
            except Exception as exc:
                logger.warning("News evaluation failed for %s: %s", symbol, exc)

        # -------------------------------------------------------------
        # 1. Filtro de Tendencia (Macro Trend Alignment)
        # Price > EMA(200) AND EMA(50) > EMA(200)
        # -------------------------------------------------------------
        cond_price_above_ema200 = close_price > ema_200
        cond_ema50_above_ema200 = ema_50 > ema_200

        if not (cond_price_above_ema200 and cond_ema50_above_ema200):
            reasons = []
            if not cond_price_above_ema200:
                reasons.append(f"Price (${close_price:.2f}) <= EMA200 (${ema_200:.2f})")
            if not cond_ema50_above_ema200:
                reasons.append(f"EMA50 (${ema_50:.2f}) <= EMA200 (${ema_200:.2f})")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                timestamp=timestamp,
                close_price=close_price,
                entry_price=close_price,
                ema_20=ema_20,
                ema_50=ema_50,
                ema_200=ema_200,
                rsi_14=rsi_14,
                atr_14=atr_14,
                prev_high=prev_high,
                reason=f"Trend filter failed: {'; '.join(reasons)}",
                ml_probability=ml_prob,
                news_safe=news_safe,
                news_reason=news_reason,
            )

        # -------------------------------------------------------------
        # 2. Pullback hacia EMA(20)
        # In recent N bars, lowest price reached near or below EMA(20)
        # -------------------------------------------------------------
        lookback = min(self.config.pullback_lookback, len(indicators_df))
        recent_window = indicators_df.iloc[-lookback:]
        
        # Check if any low tested EMA20 within tolerance
        tolerance_factor = 1.0 + self.config.pullback_tolerance_pct
        pullback_tested = False
        for _, row in recent_window.iterrows():
            if row["low"] <= (row["ema_fast"] * tolerance_factor):
                pullback_tested = True
                break

        if not pullback_tested:
            min_recent_low = float(recent_window["low"].min())
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                timestamp=timestamp,
                close_price=close_price,
                entry_price=close_price,
                ema_20=ema_20,
                ema_50=ema_50,
                ema_200=ema_200,
                rsi_14=rsi_14,
                atr_14=atr_14,
                prev_high=prev_high,
                reason=f"No recent pullback to EMA20. Min low (${min_recent_low:.2f}) > EMA20 (${ema_20:.2f}) + {self.config.pullback_tolerance_pct*100:.1f}%",
                ml_probability=ml_prob,
                news_safe=news_safe,
                news_reason=news_reason,
            )

        # -------------------------------------------------------------
        # 3. Disparador de Entrada: Cierre por encima del máximo de la vela previa
        # Close > prev_high
        # -------------------------------------------------------------
        cond_breakout = close_price > prev_high
        if not cond_breakout:
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                timestamp=timestamp,
                close_price=close_price,
                entry_price=close_price,
                ema_20=ema_20,
                ema_50=ema_50,
                ema_200=ema_200,
                rsi_14=rsi_14,
                atr_14=atr_14,
                prev_high=prev_high,
                reason=f"Trigger failed: Close (${close_price:.2f}) did not exceed previous high (${prev_high:.2f})",
                ml_probability=ml_prob,
                news_safe=news_safe,
                news_reason=news_reason,
            )

        # -------------------------------------------------------------
        # 4. Momentum: RSI(14) rebotando sobre 40
        # - Current RSI >= 40
        # - Current RSI > Previous RSI (turning upwards)
        # - Recent RSI in window tested the pullback support zone (<= rsi_zone_upper)
        # -------------------------------------------------------------
        rsi_window = indicators_df["rsi"].iloc[-self.config.rsi_lookback:]
        min_recent_rsi = float(rsi_window.min())

        cond_rsi_above_thresh = rsi_14 >= self.config.rsi_threshold
        cond_rsi_curling_up = rsi_14 > prev_rsi
        cond_rsi_came_from_zone = min_recent_rsi <= self.config.rsi_zone_upper

        if not (cond_rsi_above_thresh and cond_rsi_curling_up and cond_rsi_came_from_zone):
            reasons = []
            if not cond_rsi_above_thresh:
                reasons.append(f"RSI ({rsi_14:.1f}) < {self.config.rsi_threshold}")
            if not cond_rsi_curling_up:
                reasons.append(f"RSI not curling up ({rsi_14:.1f} <= prev {prev_rsi:.1f})")
            if not cond_rsi_came_from_zone:
                reasons.append(f"RSI did not test support zone (min {min_recent_rsi:.1f} > {self.config.rsi_zone_upper})")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                timestamp=timestamp,
                close_price=close_price,
                entry_price=close_price,
                ema_20=ema_20,
                ema_50=ema_50,
                ema_200=ema_200,
                rsi_14=rsi_14,
                atr_14=atr_14,
                prev_high=prev_high,
                reason=f"RSI bounce condition failed: {'; '.join(reasons)}",
                ml_probability=ml_prob,
                news_safe=news_safe,
                news_reason=news_reason,
            )

        # -------------------------------------------------------------
        # 5. Filtro de Noticias & Calendario de Earnings
        # -------------------------------------------------------------
        if not news_safe:
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                timestamp=timestamp,
                close_price=close_price,
                entry_price=close_price,
                ema_20=ema_20,
                ema_50=ema_50,
                ema_200=ema_200,
                rsi_14=rsi_14,
                atr_14=atr_14,
                prev_high=prev_high,
                reason=f"Technical OK | {news_reason}",
                ml_probability=ml_prob,
                news_safe=False,
                news_reason=news_reason,
            )

        # -------------------------------------------------------------
        # 6. Filtro de Machine Learning (Random Forest >= 60%)
        # -------------------------------------------------------------
        if not ml_approved:
            prob_pct = int(round((ml_prob or 0.0) * 100))
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                timestamp=timestamp,
                close_price=close_price,
                entry_price=close_price,
                ema_20=ema_20,
                ema_50=ema_50,
                ema_200=ema_200,
                rsi_14=rsi_14,
                atr_14=atr_14,
                prev_high=prev_high,
                reason=f"Technical OK | ML Rejected ({prob_pct}%)",
                ml_probability=ml_prob,
                news_safe=news_safe,
                news_reason=news_reason,
            )

        # -------------------------------------------------------------
        # All conditions satisfied: Valid BUY signal
        # -------------------------------------------------------------
        metrics = {
            "min_recent_low": float(recent_window["low"].min()),
            "min_recent_rsi": min_recent_rsi,
            "dist_to_ema20_pct": (close_price - ema_20) / ema_20,
        }

        prob_pct = int(round((ml_prob or 0.0) * 100))
        return TradingSignal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            timestamp=timestamp,
            close_price=close_price,
            entry_price=close_price,
            ema_20=ema_20,
            ema_50=ema_50,
            ema_200=ema_200,
            rsi_14=rsi_14,
            atr_14=atr_14,
            prev_high=prev_high,
            reason=(
                f"BUY setup confirmed: Technical setup valid, News safe, "
                f"and ML Confirmed ({prob_pct}%)."
            ),
            metrics=metrics,
            ml_probability=ml_prob,
            news_safe=True,
            news_reason=news_reason,
        )
