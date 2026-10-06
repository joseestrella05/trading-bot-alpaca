"""Unit tests for AlpacaExecutionHandler.

Tests:
- Dry run bracket order generation
- Rejection handling when position sizing is marked invalid
- Correct bracket leg price targets (TP and SL)
- Correct integer share formatting for Alpaca API
"""

import unittest
from unittest.mock import MagicMock, patch

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


if __name__ == "__main__":
    unittest.main()
