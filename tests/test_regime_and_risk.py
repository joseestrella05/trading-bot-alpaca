"""Unit tests for Market Regime Detection and Adaptive Risk Management Engine."""

from dataclasses import replace
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
            defensive_atr_percentile=88.0,
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
        self.assertEqual(analysis.risk_multiplier, 1.25)
        self.assertEqual(analysis.risk_per_trade_pct, 0.0125)
        self.assertEqual(analysis.risk_reward_ratio, 3.0)
        self.assertTrue(analysis.allow_entries)
        self.assertIn("AGRESIVO", analysis.mode_text)
        self.assertEqual(analysis.display_badge, "BULL TREND [1.25%]")

    def test_defensive_active_via_price_below_ema200(self):
        # Trend broken (price < EMA200) with normal volatility -> reduced size, not a block
        df = self._generate_synthetic_bars(n_bars=220, trend="bear", noise=0.5)
        analysis = self.detector.analyze(df, symbol="BEAR_TEST")

        self.assertEqual(analysis.regime, MarketRegime.HIGH_VOLATILITY_DEFENSIVE)
        self.assertEqual(analysis.risk_multiplier, 0.40)
        self.assertAlmostEqual(analysis.risk_per_trade_pct, 0.0040)
        self.assertTrue(analysis.allow_entries)
        self.assertEqual(analysis.display_badge, "DEFENSIVE [0.40%]")

    def test_halt_via_extreme_volatility(self):
        # Generate bull trend bars but spike volatility in recent bars
        df = self._generate_synthetic_bars(n_bars=220, trend="bull", noise=0.1)
        # Inject extreme wide range bars at the end to shoot ATR above the 88th percentile
        for idx in range(-5, 0):
            df.iloc[idx, df.columns.get_loc("high")] += 25.0
            df.iloc[idx, df.columns.get_loc("low")] -= 25.0

        analysis = self.detector.analyze(df, symbol="VOL_SPIKE")
        self.assertEqual(analysis.regime, MarketRegime.HIGH_VOLATILITY_DEFENSIVE)
        self.assertEqual(analysis.risk_multiplier, 0.0)
        self.assertFalse(analysis.allow_entries)
        self.assertGreater(analysis.atr_percentile, 88.0)

    def _with_atr_percentile(self, pct: float):
        df = self._generate_synthetic_bars(n_bars=220, trend="bull", noise=0.2)
        self.detector.compute_atr_percentile = lambda _atr: pct
        return self.detector.analyze(df, symbol="PCT")

    def test_multiplier_bands_by_atr_percentile(self):
        # Same bullish structure; only the volatility percentile changes the multiplier
        self.assertEqual(self._with_atr_percentile(60.0).risk_multiplier, 1.25)
        self.assertEqual(self._with_atr_percentile(70.0).risk_multiplier, 1.25)
        self.assertEqual(self._with_atr_percentile(75.0).risk_multiplier, 0.40)
        self.assertEqual(self._with_atr_percentile(88.0).risk_multiplier, 0.40)
        self.assertEqual(self._with_atr_percentile(88.5).risk_multiplier, 0.0)

    def test_halt_via_five_bar_drawdown(self):
        df = self._generate_synthetic_bars(n_bars=220, trend="bull", noise=0.2)
        df.iloc[-1, df.columns.get_loc("close")] *= 0.92  # -8% from the 5-bar high
        detector = MarketRegimeDetector(volatility_lookback=100)
        detector.compute_atr_percentile = lambda _atr: 50.0  # isolate the drawdown rule
        analysis = detector.analyze(df, symbol="CRASH")
        self.assertEqual(analysis.risk_multiplier, 0.0)
        self.assertIn("drawdown", analysis.description)

    def test_range_chop_classification(self):
        df = self._generate_synthetic_bars(n_bars=220, trend="range", noise=0.3)
        analysis = self.detector.analyze(df, symbol="CHOP_TEST")

        self.assertEqual(analysis.regime, MarketRegime.RANGE_CHOP)
        self.assertEqual(analysis.risk_multiplier, 0.75)
        self.assertAlmostEqual(analysis.risk_per_trade_pct, 0.0075)
        self.assertEqual(analysis.risk_reward_ratio, 1.5)
        self.assertTrue(analysis.allow_entries)
        self.assertIn("MODERADO", analysis.mode_text)


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

    def test_range_chop_moderate_sizing(self):
        # Equity = $100,000 -> 1.0% base * 0.75 = 0.75% risk = $750
        # Entry = $100.00, ATR = $2.00 -> SL distance = $3.00, Stop Loss = $97.00
        # R:R = 1.5:1 -> Take Profit = $100.00 + (1.5 * $3.00) = $104.50
        # Shares = $750 / $3.00 = 250
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
        self.assertEqual(pos_size.shares, 250.0)
        self.assertAlmostEqual(pos_size.risk_pct_used, 0.0075, places=4)

    def test_defensive_active_small_size(self):
        # Defensive multiplier 0.40 -> 0.40% risk = $400 / $3.00 = 133.3 -> 133 shares (not blocked)
        account = AccountState(
            equity=100000.0, last_equity=100000.0, cash=80000.0, buying_power=300000.0,
            intraday_drawdown_pct=0.0, kill_switch_active=False,
        )
        signal = replace(self._sample_signal(entry=100.0, atr=2.0), risk_multiplier=0.40)
        pos_size = self.engine.calculate_position_size(
            signal, account, regime=MarketRegime.HIGH_VOLATILITY_DEFENSIVE
        )
        self.assertTrue(pos_size.is_valid)
        self.assertEqual(pos_size.shares, 133.0)
        self.assertAlmostEqual(pos_size.risk_pct_used, 0.0040, places=4)
        self.assertEqual(pos_size.take_profit_price, 106.00)  # 2:1 in defensive mode

    def test_signal_multiplier_overrides_regime_default(self):
        # Continuous scaling: multiplier 0.60 on a BULL_TREND regime -> 0.60% risk = $600 / $3 = 200
        account = AccountState(
            equity=100000.0, last_equity=100000.0, cash=80000.0, buying_power=300000.0,
            intraday_drawdown_pct=0.0, kill_switch_active=False,
        )
        signal = replace(self._sample_signal(entry=100.0, atr=2.0), risk_multiplier=0.60)
        pos_size = self.engine.calculate_position_size(signal, account, regime=MarketRegime.BULL_TREND)
        self.assertEqual(pos_size.shares, 200.0)

    def test_regime_absolute_halt(self):
        account = AccountState(
            equity=100000.0,
            last_equity=100000.0,
            cash=80000.0,
            buying_power=300000.0,
            intraday_drawdown_pct=0.0,
            kill_switch_active=False,
        )
        signal = replace(self._sample_signal(entry=100.0, atr=2.0), risk_multiplier=0.0)

        pos_size = self.engine.calculate_position_size(
            signal, account, regime=MarketRegime.HIGH_VOLATILITY_DEFENSIVE
        )

        self.assertFalse(pos_size.is_valid)
        self.assertEqual(pos_size.shares, 0.0)
        self.assertEqual(pos_size.rejection_reason, "Regime absolute halt: Extreme market volatility")

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


class TestStrategyRegimeModulation(unittest.TestCase):
    """Strategy propagates the most conservative multiplier and only blocks on an absolute halt."""

    def setUp(self):
        self.df = TestMarketRegimeDetector()._generate_synthetic_bars(n_bars=220, trend="bull", noise=0.2)
        self.strategy = SwingTrendFollowingStrategy(StrategyConfig())

    def _macro(self, multiplier: float, regime: MarketRegime) -> RegimeAnalysis:
        return MarketRegimeDetector()._build(
            regime, multiplier, "SPY", "synthetic macro",
            price=1.0, ema_20=1.0, ema_50=1.0, ema_200=1.0, ema20_slope=0.0, ema50_slope=0.0,
            atr_14=1.0, atr_percentile=50.0, adx_14=20.0,
        )

    def test_macro_halt_forces_hold(self):
        signal = self.strategy.evaluate(
            "TEST", self.df, macro_regime=self._macro(0.0, MarketRegime.HIGH_VOLATILITY_DEFENSIVE)
        )
        self.assertEqual(signal.signal_type, SignalType.HOLD)
        self.assertIn("Regime absolute halt", signal.reason)
        self.assertEqual(signal.risk_multiplier, 0.0)

    def test_macro_defensive_no_longer_blocks(self):
        signal = self.strategy.evaluate(
            "TEST", self.df, macro_regime=self._macro(0.40, MarketRegime.HIGH_VOLATILITY_DEFENSIVE)
        )
        self.assertNotIn("Regime", signal.reason.split(":")[0])
        self.assertEqual(signal.regime, MarketRegime.HIGH_VOLATILITY_DEFENSIVE.value)
        self.assertEqual(signal.risk_multiplier, 0.40)  # min(macro 0.40, asset bull 1.25)


if __name__ == "__main__":
    unittest.main()
