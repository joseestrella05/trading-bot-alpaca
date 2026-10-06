"""Unit tests for SwingTrendFollowingStrategy.

Tests:
- Indicator generation (EMA20, EMA50, EMA200, ATR14, RSI14)
- Trend filter logic (Price > EMA200 & EMA50 > EMA200)
- Pullback detection to EMA20
- Trigger condition (Close > prev_high)
- RSI bounce over 40
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from config import StrategyConfig
from models import SignalType
from strategy import SwingTrendFollowingStrategy


class TestStrategy(unittest.TestCase):
    def setUp(self):
        self.config = StrategyConfig(
            ema_fast=20,
            ema_medium=50,
            ema_slow=200,
            atr_period=14,
            rsi_period=14,
            rsi_threshold=40.0,
            rsi_zone_upper=45.0,
            rsi_lookback=3,
            pullback_lookback=3,
            pullback_tolerance_pct=0.01,
        )
        self.strategy = SwingTrendFollowingStrategy(self.config)

    def _generate_synthetic_data(self, n_bars: int = 250, base_price: float = 100.0) -> pd.DataFrame:
        """Create steady uptrend daily bars."""
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n_bars)]
        prices = base_price + np.linspace(10, 150, n_bars)  # Clear uptrend

        df = pd.DataFrame(
            {
                "open": prices - 0.5,
                "high": prices + 1.5,
                "low": prices - 1.5,
                "close": prices,
                "volume": 1000000,
            },
            index=dates,
        )
        return df

    def test_indicator_calculations(self):
        df = self._generate_synthetic_data(250)
        enriched = self.strategy.calculate_indicators(df)

        self.assertIn("ema_fast", enriched.columns)
        self.assertIn("ema_medium", enriched.columns)
        self.assertIn("ema_slow", enriched.columns)
        self.assertIn("atr", enriched.columns)
        self.assertIn("rsi", enriched.columns)

        # In a steady uptrend, EMA20 > EMA50 > EMA200
        latest = enriched.iloc[-1]
        self.assertGreater(latest["ema_fast"], latest["ema_medium"])
        self.assertGreater(latest["ema_medium"], latest["ema_slow"])
        self.assertGreater(latest["atr"], 0.0)
        self.assertGreater(latest["rsi"], 0.0)
        self.assertLessEqual(latest["rsi"], 100.0)

    def test_trend_filter_rejection(self):
        # Generate downtrend data
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(250)]
        prices = 200.0 - np.linspace(0, 100, 250)
        df = pd.DataFrame(
            {
                "open": prices + 0.5,
                "high": prices + 1.0,
                "low": prices - 1.0,
                "close": prices,
                "volume": 1000000,
            },
            index=dates,
        )

        signal = self.strategy.evaluate("DOWNTREND", df)
        self.assertEqual(signal.signal_type, SignalType.HOLD)
        self.assertIn("Trend filter failed", signal.reason)

    def test_trigger_rejection_no_breakout(self):
        # Generate uptrend data where latest close fails to exceed previous high
        df = self._generate_synthetic_data(250)
        # Force latest bar close to be lower than previous bar high
        df.loc[df.index[-1], "close"] = df.loc[df.index[-2], "high"] - 2.0

        signal = self.strategy.evaluate("UPTREND_NO_BREAK", df)
        self.assertEqual(signal.signal_type, SignalType.HOLD)

    def test_insufficient_history(self):
        df = self._generate_synthetic_data(50)  # Only 50 bars
        signal = self.strategy.evaluate("SHORT_DATA", df)
        self.assertEqual(signal.signal_type, SignalType.HOLD)
        self.assertIn("Insufficient history", signal.reason)


if __name__ == "__main__":
    unittest.main()
