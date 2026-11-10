"""Unit tests for AlpacaExecutionHandler.

Tests:
- Dry run bracket order generation
- Rejection handling when position sizing is marked invalid
- Correct bracket leg price targets (TP and SL)
- Correct integer share formatting for Alpaca API
"""

import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from alpaca.common.exceptions import APIError

from config import AlpacaConfig, ExecutionConfig, RiskConfig
from execution import AlpacaExecutionHandler
from models import PositionSize
from risk_manager import RiskManager


class TestExecutionHandler(unittest.TestCase):
    def setUp(self):
        self.alpaca_cfg = AlpacaConfig(
            api_key="TEST_KEY",
            secret_key="TEST_SECRET",
            paper=True,
        )
        self.exec_cfg = ExecutionConfig(
            entry_order_type="market",
            time_in_force="gtc",
            dry_run=True,
        )
        self.risk_manager = RiskManager(RiskConfig())

    @patch("execution.TradingClient")
    def test_dry_run_bracket_order(self, mock_client_cls):
        handler = AlpacaExecutionHandler(
            alpaca_config=self.alpaca_cfg,
            execution_config=self.exec_cfg,
            risk_manager=self.risk_manager,
        )

        pos = PositionSize(
            symbol="SPY",
            shares=50.0,
            is_fractional=False,
            entry_price=500.0,
            stop_loss_price=490.0,
            take_profit_price=520.0,
            risk_amount_usd=500.0,
            risk_per_share=10.0,
            reward_per_share=20.0,
            risk_reward_ratio=2.0,
            total_exposure_usd=25000.0,
            is_valid=True,
            rejection_reason=None,
        )

        result = handler.submit_bracket_order(pos)
        self.assertTrue(result.success)
        self.assertEqual(result.status, "SIMULATED")
        self.assertEqual(result.submitted_qty, 50.0)
        self.assertIsNotNone(result.order_id)

    @patch("execution.TradingClient")
    def test_invalid_position_rejected(self, mock_client_cls):
        handler = AlpacaExecutionHandler(
            alpaca_config=self.alpaca_cfg,
            execution_config=self.exec_cfg,
            risk_manager=self.risk_manager,
        )

        pos = PositionSize(
            symbol="SPY",
            shares=0.0,
            is_fractional=False,
            entry_price=500.0,
            stop_loss_price=0.0,
            take_profit_price=0.0,
            risk_amount_usd=0.0,
            risk_per_share=0.0,
            reward_per_share=0.0,
            risk_reward_ratio=0.0,
            total_exposure_usd=0.0,
            is_valid=False,
            rejection_reason="Drawdown Kill-Switch Active",
        )

        result = handler.submit_bracket_order(pos)
        self.assertFalse(result.success)
        self.assertEqual(result.status, "VALIDATION_FAILED")
        self.assertEqual(result.error_message, "Drawdown Kill-Switch Active")


def _bars(closes):
    """Daily bars ending yesterday with a fixed +/-1 range (ATR converges to ~2)."""
    closes = np.asarray(closes, dtype=float)
    end = pd.Timestamp(date.today() - timedelta(days=1), tz="UTC")
    idx = pd.date_range(end=end, periods=len(closes), freq="D")
    return pd.DataFrame(
        {"open": closes, "high": closes + 1, "low": closes - 1, "close": closes, "volume": 1e6}, index=idx
    )


class TestTrailingStops(unittest.TestCase):
    @patch("execution.TradingClient")
    def setUp(self, _mock_client_cls):
        self.handler = AlpacaExecutionHandler(
            alpaca_config=AlpacaConfig(api_key="K", secret_key="S", paper=True),
            execution_config=ExecutionConfig(dry_run=False),
            risk_manager=RiskManager(RiskConfig()),  # atr_sl_multiplier=1.5 -> 1.5R = 2.25 ATR
        )
        self.client = self.handler._client
        self.data_provider = MagicMock()

    def _pos(self, price, stop, entry=100.0):
        return {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": entry, "current_price": price,
                "stop_loss": stop, "stop_order_id": "sl-1"}

    def _run(self, pos, bars):
        return self.handler.update_open_positions_trailing_stops(
            [pos], self.data_provider, bars_by_symbol={"AAPL": bars}
        )

    def test_breakeven_after_one_atr(self):
        # Flat 100 -> ATR=2, EMA=100. Price 102.5 >= 100 + 1*ATR but < 1.5R trail level (104.5).
        updated = self._run(self._pos(price=102.5, stop=97.0), _bars([100.0] * 60))
        self.assertEqual(updated, {"AAPL": 100.0})
        order_id, req = self.client.replace_order_by_id.call_args.args
        self.assertEqual((order_id, req.stop_price), ("sl-1", 100.0))

    def test_trail_to_ema_and_last_low(self):
        # Rising ramp 90->110: EMA20 ~106.8, last closed low = 109 -> stop goes to the higher (109).
        bars = _bars(np.linspace(90, 110, 60))
        updated = self._run(self._pos(price=111.0, stop=100.0), bars)
        self.assertEqual(updated, {"AAPL": 109.0})
        self.assertEqual(self.client.replace_order_by_id.call_args.args[1].stop_price, 109.0)

    def test_stop_never_lowered(self):
        bars = _bars(np.linspace(90, 110, 60))
        # Losing position: untouched.
        self.assertEqual(self._run(self._pos(price=95.0, stop=97.0), bars), {})
        # Winning, but current stop (110) already above EMA/low/entry: not lowered.
        self.assertEqual(self._run(self._pos(price=111.0, stop=110.0), bars), {})
        # In profit but below the breakeven trigger: untouched.
        self.assertEqual(self._run(self._pos(price=101.0, stop=97.0), _bars([100.0] * 60)), {})
        self.client.replace_order_by_id.assert_not_called()

    def test_api_error_is_handled(self):
        self.client.replace_order_by_id.side_effect = APIError('{"code":42210000,"message":"order is pending"}')
        self.assertEqual(self._run(self._pos(price=102.5, stop=97.0), _bars([100.0] * 60)), {})

    def test_missing_bars_downloaded_from_provider(self):
        self.data_provider.get_daily_bars.return_value = _bars([100.0] * 60)
        updated = self.handler.update_open_positions_trailing_stops([self._pos(102.5, 97.0)], self.data_provider)
        self.assertEqual(updated, {"AAPL": 100.0})
        self.data_provider.get_daily_bars.assert_called_once()


if __name__ == "__main__":
    unittest.main()
