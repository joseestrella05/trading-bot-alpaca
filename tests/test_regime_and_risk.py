"""Unit tests for Market Regime Detection and Adaptive Risk Management Engine."""

from datetime import datetime, timezone
import math
import unittest
import numpy as np
import pandas as pd

from config import RiskConfig, StrategyConfig
from models import AccountState, MarketRegime, SignalType, TradingSignal
from regime_detector import MarketRegimeDetector, RegimeAnalysis
from risk_engine import AdaptiveRiskEngine, REGIME_RISK_SPECS
from strategy import SwingTrendFollowingStrategy


class TestMarketRegimeDetector(unittest.TestCase):
    def setUp(self):
        self.detector = MarketRegimeDetector(
            volatility_lookback=100,
            defensive_atr_percentile=85.0,
            bull_atr_percentile=70.0,
            slope_bars=5,
        )

    def _generate_synthetic_bars(
        self,
        n_bars: int = 150,
        trend: str = "bull",
        noise: float = 0.5,
    ) -> pd.DataFrame:
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=n_bars, freq="D")
        if trend == "bull":
            prices = 100.0 + np.linspace(0, 50, n_bars)
        elif trend == "bear":
            prices = 200.0 - np.linspace(0, 80, n_bars)
        else:  # range / flat
            prices = 150.0 + np.sin(np.linspace(0, 15, n_bars)) * 2.0

        np.random.seed(42)
        jitter = np.random.normal(0, noise, n_bars)
        close = prices + jitter
        high = close + np.abs(np.random.normal(1.0, 0.2, n_bars))
        low = close - np.abs(np.random.normal(1.0, 0.2, n_bars))
        open_p = (close + low) / 2.0

        return pd.DataFrame(
            {"open": open_p, "high": high, "low": low, "close": close, "volume": 1000000},
            index=dates,
        )

    def test_bull_trend_classification(self):
        df = self._generate_synthetic_bars(n_bars=220, trend="bull", noise=0.2)
        analysis = self.detector.analyze(df, symbol="BULL_TEST")

        self.assertEqual(analysis.regime, MarketRegime.BULL_TREND)
        self.assertEqual(analysis.risk_per_trade_pct, 0.0125)
        self.assertEqual(analysis.risk_reward_ratio, 3.0)
        self.assertTrue(analysis.allow_entries)
        self.assertIn("AGRESIVO", analysis.mode_text)

    def test_defensive_mode_via_price_below_ema200(self):
        # Generate downtrend data where price is below EMA200
        df = self._generate_synthetic_bars(n_bars=220, trend="bear", noise=0.5)
        analysis = self.detector.analyze(df, symbol="BEAR_TEST")

        self.assertEqual(analysis.regime, MarketRegime.HIGH_VOLATILITY_DEFENSIVE)
        self.assertEqual(analysis.risk_per_trade_pct, 0.0)
        self.assertFalse(analysis.allow_entries)
        self.assertIn("DEFENSIVE", analysis.mode_text)

    def test_defensive_mode_via_high_volatility(self):
        # Generate bull trend bars but spike volatility in recent bars
        df = self._generate_synthetic_bars(n_bars=220, trend="bull", noise=0.1)
        # Inject extreme wide range bars at the end to shoot ATR above 85th percentile
        for idx in range(-5, 0):
            df.iloc[idx, df.columns.get_loc("high")] += 25.0
            df.iloc[idx, df.columns.get_loc("low")] -= 25.0

        analysis = self.detector.analyze(df, symbol="VOL_SPIKE")
        self.assertEqual(analysis.regime, MarketRegime.HIGH_VOLATILITY_DEFENSIVE)
        self.assertFalse(analysis.allow_entries)
        self.assertGreaterEqual(analysis.atr_percentile, 85.0)

    def test_range_chop_classification(self):
        df = self._generate_synthetic_bars(n_bars=220, trend="range", noise=0.3)
        analysis = self.detector.analyze(df, symbol="CHOP_TEST")

        self.assertEqual(analysis.regime, MarketRegime.RANGE_CHOP)
        self.assertEqual(analysis.risk_per_trade_pct, 0.0050)
        self.assertEqual(analysis.risk_reward_ratio, 1.5)
        self.assertTrue(analysis.allow_entries)
        self.assertIn("CONSERVADOR", analysis.mode_text)


class TestAdaptiveRiskEngine(unittest.TestCase):
    def setUp(self):
        self.config = RiskConfig(
            max_risk_per_trade_pct=0.01,
            max_intraday_drawdown_pct=0.025,
            risk_reward_ratio=2.0,
            atr_sl_multiplier=1.5,
            max_position_equity_pct=0.50,
            allow_fractional=False,
        )
        self.engine = AdaptiveRiskEngine(self.config)

    def _sample_signal(self, entry: float = 100.0, atr: float = 2.0) -> TradingSignal:
        return TradingSignal(
            symbol="TEST",
            signal_type=SignalType.BUY,
            timestamp=datetime.now(timezone.utc),
            close_price=entry,
            entry_price=entry,
            ema_20=95.0,
            ema_50=90.0,
            ema_200=80.0,
            rsi_14=45.0,
            atr_14=atr,
            prev_high=98.0,
            reason="Valid setup",
        )

    def test_bull_trend_aggressive_sizing(self):
        # Equity = $100,000 -> 1.25% risk = $1,250
        # Entry = $100.00, ATR = $2.00 -> SL distance = 1.5 * 2.00 = $3.00
        # Stop Loss = $97.00
        # R:R = 3:1 -> Take Profit = $100.00 + (3.0 * $3.00) = $109.00
        # Shares = $1,250 / $3.00 = 416.666... -> floored to 416
        account = AccountState(
            equity=100000.0,
            last_equity=100000.0,
            cash=80000.0,
            buying_power=300000.0,
            intraday_drawdown_pct=0.0,
            kill_switch_active=False,
        )
        signal = self._sample_signal(entry=100.0, atr=2.0)

        pos_size = self.engine.calculate_position_size(
            signal, account, regime=MarketRegime.BULL_TREND
        )

        self.assertTrue(pos_size.is_valid)
        self.assertEqual(pos_size.stop_loss_price, 97.00)
        self.assertEqual(pos_size.take_profit_price, 109.00)
        self.assertEqual(pos_size.risk_reward_ratio, 3.0)
        self.assertEqual(pos_size.shares, 416.0)
        self.assertAlmostEqual(pos_size.risk_pct_used, 0.0125, places=4)

    def test_range_chop_conservative_sizing(self):
        # Equity = $100,000 -> 0.50% risk = $500
        # Entry = $100.00, ATR = $2.00 -> SL distance = $3.00, Stop Loss = $97.00
        # R:R = 1.5:1 -> Take Profit = $100.00 + (1.5 * $3.00) = $104.50
        # Shares = $500 / $3.00 = 166.666... -> floored to 166
        account = AccountState(
            equity=100000.0,
            last_equity=100000.0,
            cash=80000.0,
            buying_power=300000.0,
            intraday_drawdown_pct=0.0,
            kill_switch_active=False,
        )
        signal = self._sample_signal(entry=100.0, atr=2.0)

        pos_size = self.engine.calculate_position_size(
            signal, account, regime=MarketRegime.RANGE_CHOP
        )

        self.assertTrue(pos_size.is_valid)
        self.assertEqual(pos_size.stop_loss_price, 97.00)
        self.assertEqual(pos_size.take_profit_price, 104.50)
        self.assertEqual(pos_size.risk_reward_ratio, 1.5)
        self.assertEqual(pos_size.shares, 166.0)
        self.assertAlmostEqual(pos_size.risk_pct_used, 0.0050, places=4)

    def test_defensive_mode_total_block(self):
        account = AccountState(
            equity=100000.0,
            last_equity=100000.0,
            cash=80000.0,
            buying_power=300000.0,
            intraday_drawdown_pct=0.0,
            kill_switch_active=False,
        )
        signal = self._sample_signal(entry=100.0, atr=2.0)

        pos_size = self.engine.calculate_position_size(
            signal, account, regime=MarketRegime.HIGH_VOLATILITY_DEFENSIVE
        )

        self.assertFalse(pos_size.is_valid)
        self.assertEqual(pos_size.shares, 0.0)
        self.assertIn("HIGH_VOLATILITY_DEFENSIVE", pos_size.rejection_reason)

    def test_intraday_circuit_breaker_guardrail(self):
        # Drawdown = 2.6% (>= 2.5% max guardrail) -> must block
        account = AccountState(
            equity=97400.0,
            last_equity=100000.0,
            cash=50000.0,
            buying_power=200000.0,
            intraday_drawdown_pct=0.026,
            kill_switch_active=True,
        )
        signal = self._sample_signal(entry=100.0, atr=2.0)

        pos_size = self.engine.calculate_position_size(
            signal, account, regime=MarketRegime.BULL_TREND
        )

        self.assertFalse(pos_size.is_valid)
        self.assertEqual(pos_size.shares, 0.0)
        self.assertIn("Circuit breaker", pos_size.rejection_reason)


if __name__ == "__main__":
    unittest.main()
