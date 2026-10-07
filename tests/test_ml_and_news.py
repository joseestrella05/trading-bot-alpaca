"""Unit tests for ML Predictor, News & Earnings Filter, and Integrated Strategy."""

from datetime import datetime, date, timedelta
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SignalType
from news_filter import NewsFilter, NewsFilterResult
from ml_model import StockMLPredictor, MLPredictionResult
from strategy import SwingTrendFollowingStrategy


class TestNewsAndML(unittest.TestCase):
    def test_news_filter_etf_bypass(self):
        filter_inst = NewsFilter()
        res = filter_inst.evaluate("SPY")
        self.assertTrue(res.earnings_safe)

    @patch("news_filter.NewsFilter.fetch_macro_calendar")
    def test_news_filter_macro_risk(self, mock_fetch):
        from datetime import timezone
        # Event in 30 minutes (UTC)
        imminent_time = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        mock_fetch.return_value = [
            {"country": "USD", "impact": "High", "title": "FOMC Rate Decision", "date": imminent_time}
        ]
        filter_inst = NewsFilter()
        safe, desc = filter_inst.check_macro_risk()
        self.assertFalse(safe)
        self.assertIn("FOMC", desc)

    def test_ml_feature_computation(self):
        predictor = StockMLPredictor()
        # Synthetic data with 30 bars
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(35)]
        close_vals = np.linspace(100, 130, 35)
        df = pd.DataFrame({
            "close": close_vals,
            "ema_fast": close_vals * 0.98,
            "ema_medium": close_vals * 0.95,
            "rsi": 55.0,
            "atr": 2.5,
        }, index=dates)

        feat_df = predictor.compute_features(df)
        self.assertIn("ret_5d", feat_df.columns)
        self.assertIn("ret_20d", feat_df.columns)
        self.assertIn("dist_ema20", feat_df.columns)
        self.assertIn("dist_ema50", feat_df.columns)
        self.assertIn("rsi_14", feat_df.columns)
        self.assertIn("norm_atr", feat_df.columns)

    def test_ml_approval_threshold(self):
        predictor = StockMLPredictor(approval_threshold=0.60)
        # Mock pipeline
        mock_pipe = MagicMock()
        # Case 1: Probability 0.65 >= 0.60 -> Approved
        mock_pipe.predict_proba.return_value = np.array([[0.35, 0.65]])
        predictor.pipeline = mock_pipe

        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(35)]
        close_vals = np.linspace(100, 130, 35)
        df = pd.DataFrame({
            "close": close_vals,
            "ema_fast": close_vals * 0.98,
            "ema_medium": close_vals * 0.95,
            "rsi": 55.0,
            "atr": 2.5,
        }, index=dates)

        res_approved = predictor.predict("AAPL", df)
        self.assertTrue(res_approved.is_approved)
        self.assertAlmostEqual(res_approved.probability, 0.65, places=2)

        # Case 2: Probability 0.52 < 0.60 -> Rejected
        mock_pipe.predict_proba.return_value = np.array([[0.48, 0.52]])
        res_rejected = predictor.predict("AAPL", df)
        self.assertFalse(res_rejected.is_approved)
        self.assertIn("ML Rejected (52%)", res_rejected.reason)

    def test_strategy_integration_rejections(self):
        mock_news = MagicMock()
        mock_ml = MagicMock()

        strategy = SwingTrendFollowingStrategy(news_filter=mock_news, ml_predictor=mock_ml)

        # Setup where technical would pass (or mock indicators)
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(250)]
        prices = 100.0 + np.linspace(10, 150, 250)
        df = pd.DataFrame({
            "open": prices - 0.5,
            "high": prices + 1.5,
            "low": prices - 1.5,
            "close": prices,
            "volume": 1000000,
        }, index=dates)

        # If news blocks trade
        mock_news.evaluate.return_value = NewsFilterResult(
            symbol="TEST", safe=False, earnings_safe=False, macro_safe=True, reason="Blocked by Earnings"
        )
        mock_ml.predict.return_value = MLPredictionResult(
            symbol="TEST", probability=0.70, is_approved=True, reason="ML Confirmed (70%)"
        )

        signal = strategy.evaluate("TEST", df)
        # Must be HOLD
        self.assertEqual(signal.signal_type, SignalType.HOLD)
        self.assertFalse(signal.news_safe)


if __name__ == "__main__":
    unittest.main()
