"""Unit tests for RiskManager.

Verifies:
- 1.0% maximum risk sizing formula
- 1.5 * ATR(14) Stop Loss placement
- 2:1 Take Profit target
- Intraday drawdown kill-switch at 2.5%
- Whole share flooring for Alpaca bracket orders
- Rejections on insufficient buying power or capital
"""

import math
import unittest
from datetime import datetime

from config import RiskConfig
from models import AccountState, SignalType, TradingSignal
from risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.config = RiskConfig(
            max_risk_per_trade_pct=0.01,
            max_intraday_drawdown_pct=0.025,
            risk_reward_ratio=2.0,
            atr_sl_multiplier=1.5,
            max_position_equity_pct=0.30,
            allow_fractional=False,
        )
        self.risk_manager = RiskManager(self.config)

    def test_kill_switch_activation(self):
        # 1. Normal state: 1.0% drawdown -> kill switch False
        healthy_account = self.risk_manager.evaluate_account_health(
            equity=99000.0,
            last_equity=100000.0,
            cash=50000.0,
            buying_power=200000.0,
        )
        self.assertFalse(healthy_account.kill_switch_active)
        self.assertAlmostEqual(healthy_account.intraday_drawdown_pct, 0.01, places=4)

        # 2. Drawdown >= 2.5% -> kill switch True
        breached_account = self.risk_manager.evaluate_account_health(
            equity=97400.0,
            last_equity=100000.0,
            cash=50000.0,
            buying_power=200000.0,
        )
        self.assertTrue(breached_account.kill_switch_active)
        self.assertAlmostEqual(breached_account.intraday_drawdown_pct, 0.026, places=3)

    def test_position_sizing_and_bracket_levels(self):
        # Account with $100,000 equity -> Max risk: $1,000 (1%)
        account = AccountState(
            equity=100000.0,
            last_equity=100000.0,
            cash=100000.0,
            buying_power=400000.0,
            intraday_drawdown_pct=0.0,
            kill_switch_active=False,
        )

        # Allow higher position equity cap to test pure risk formula
        self.risk_manager.config = RiskConfig(
            max_risk_per_trade_pct=0.01,
            max_intraday_drawdown_pct=0.025,
            risk_reward_ratio=2.0,
            atr_sl_multiplier=1.5,
            max_position_equity_pct=0.50,  # 50% max exposure
            allow_fractional=False,
        )

        # Signal with Entry=$200.00, ATR=$4.00
        # Expected SL distance: 1.5 * 4.00 = $6.00
        # Expected SL price: $200.00 - $6.00 = $194.00
        # Expected TP price: $200.00 + (2.0 * $6.00) = $212.00
        # Expected shares: $1,000 / $6.00 = 166.666... -> floored to 166 shares
        signal = TradingSignal(
            symbol="TEST",
            signal_type=SignalType.BUY,
            timestamp=datetime.utcnow(),
            close_price=200.0,
            entry_price=200.0,
            ema_20=195.0,
            ema_50=190.0,
            ema_200=170.0,
            rsi_14=42.0,
            atr_14=4.0,
            prev_high=198.0,
            reason="Valid setup",
        )

        pos_size = self.risk_manager.calculate_position_size(signal, account)

        self.assertTrue(pos_size.is_valid)
        self.assertEqual(pos_size.stop_loss_price, 194.00)
        self.assertEqual(pos_size.take_profit_price, 212.00)
        self.assertAlmostEqual(pos_size.risk_per_share, 6.00, places=2)
        self.assertAlmostEqual(pos_size.reward_per_share, 12.00, places=2)
        self.assertEqual(pos_size.risk_reward_ratio, 2.0)
        self.assertEqual(pos_size.shares, 166.0)
        self.assertFalse(pos_size.is_fractional)

        # Actual risk must never exceed max 1% ($1,000)
        self.assertLessEqual(pos_size.risk_amount_usd, 1000.0)
        self.assertEqual(pos_size.risk_amount_usd, 166 * 6.00)

    def test_concentration_cap_application(self):
        account = AccountState(
            equity=100000.0,
            last_equity=100000.0,
            cash=100000.0,
            buying_power=400000.0,
            intraday_drawdown_pct=0.0,
            kill_switch_active=False,
        )

        # Position cap of 20% = $20,000 max notional
        self.risk_manager.config = RiskConfig(
            max_risk_per_trade_pct=0.01,
            max_intraday_drawdown_pct=0.025,
            risk_reward_ratio=2.0,
            atr_sl_multiplier=1.5,
            max_position_equity_pct=0.20,
            allow_fractional=False,
        )

        signal = TradingSignal(
            symbol="TEST",
            signal_type=SignalType.BUY,
            timestamp=datetime.utcnow(),
            close_price=200.0,
            entry_price=200.0,
            ema_20=195.0,
            ema_50=190.0,
            ema_200=170.0,
            rsi_14=42.0,
            atr_14=4.0,
            prev_high=198.0,
            reason="Valid setup",
        )

        pos_size = self.risk_manager.calculate_position_size(signal, account)
        # Expected max shares: $20,000 / $200 = 100 shares
        self.assertEqual(pos_size.shares, 100.0)
        self.assertEqual(pos_size.total_exposure_usd, 20000.0)

    def test_rejection_when_kill_switch_active(self):
        account = AccountState(
            equity=97000.0,
            last_equity=100000.0,
            cash=50000.0,
            buying_power=100000.0,
            intraday_drawdown_pct=0.03,  # 3.0% drawdown
            kill_switch_active=True,
        )

        signal = TradingSignal(
            symbol="SPY",
            signal_type=SignalType.BUY,
            timestamp=datetime.utcnow(),
            close_price=500.0,
            entry_price=500.0,
            ema_20=490.0,
            ema_50=480.0,
            ema_200=450.0,
            rsi_14=43.0,
            atr_14=5.0,
            prev_high=498.0,
            reason="Valid setup",
        )

        pos = self.risk_manager.calculate_position_size(signal, account)
        self.assertFalse(pos.is_valid)
        self.assertIn("Kill-switch active", pos.rejection_reason)
        self.assertEqual(pos.shares, 0.0)

    def test_small_capital_sub_share_rejection(self):
        # Account with small capital: $1,000 equity -> Max risk: $10 (1%)
        account = AccountState(
            equity=1000.0,
            last_equity=1000.0,
            cash=1000.0,
            buying_power=2000.0,
            intraday_drawdown_pct=0.0,
            kill_switch_active=False,
        )

        # Stock price $500, ATR=$10 -> SL distance = $15
        # Risk for 1 share = $15 > $10 (1% limit).
        # raw_shares = $10 / $15 = 0.666 -> floor = 0 shares
        signal = TradingSignal(
            symbol="HIGH_PRICE",
            signal_type=SignalType.BUY,
            timestamp=datetime.utcnow(),
            close_price=500.0,
            entry_price=500.0,
            ema_20=490.0,
            ema_50=480.0,
            ema_200=450.0,
            rsi_14=42.0,
            atr_14=10.0,
            prev_high=495.0,
            reason="Valid setup",
        )

        pos = self.risk_manager.calculate_position_size(signal, account)
        self.assertFalse(pos.is_valid)
        self.assertEqual(pos.shares, 0.0)
        self.assertIn("< 1.0 share", pos.rejection_reason)


if __name__ == "__main__":
    unittest.main()
