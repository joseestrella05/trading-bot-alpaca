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
import tempfile

from ml_model import StockMLPredictor, MLPredictionResult, create_labeled_dataset
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


def _barrier_path(future_highs, future_lows, future_closes):
    """25 flat warm-up bars + entry bar at 100 (ATR=1 -> TP=102, SL=98.5) + the given future bars."""
    n = 26
    df = pd.DataFrame({
        "open": [100.0] * n + list(future_closes),
        "high": [100.5] * n + list(future_highs),
        "low": [99.5] * n + list(future_lows),
        "close": [100.0] * n + list(future_closes),
        "volume": 1e6,
        "ema_fast": 100.0, "ema_medium": 100.0, "rsi": 50.0, "atr": 1.0,
    }, index=pd.date_range("2025-01-01", periods=n + len(future_closes), freq="D"))
    labeled = create_labeled_dataset(df, holding_period=len(future_closes), profit_take_atr=2.0, stop_loss_atr=1.5)
    assert labeled.index[-1] == df.index[n - 1]  # entry bar is the last one with a complete window
    return int(labeled["target"].iloc[-1])


class TestLightGBMPipeline(unittest.TestCase):
    def test_triple_barrier_labels(self):
        # Take profit (102) touched on bar 3, stop (98.5) never -> 1
        self.assertEqual(_barrier_path([101, 101.5, 102.2, 101, 101], [99.5, 99.6, 100, 100, 100], [100.5, 101, 102, 100.5, 100.5]), 1)
        # Stop touched on bar 2 before take profit on bar 4 -> 0
        self.assertEqual(_barrier_path([100.5, 100.5, 101, 103, 101], [99.5, 98.4, 99, 100, 100], [100, 99, 100, 102, 100]), 0)
        # Both barriers inside the same bar -> conservative 0
        self.assertEqual(_barrier_path([102.5, 101, 101, 101, 101], [98.0, 100, 100, 100, 100], [100, 100, 100, 100, 100]), 0)
        # No barrier touched, time expiry in loss -> 0; in profit -> 1
        self.assertEqual(_barrier_path([100.5] * 5, [99.0] * 5, [100, 100, 99.8, 99.6, 99.5]), 0)
        self.assertEqual(_barrier_path([101.5] * 5, [99.5] * 5, [100.5, 101, 101, 101.2, 101.4]), 1)

    def test_features_have_no_lookahead(self):
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80)))
        df = pd.DataFrame({"close": close, "volume": rng.uniform(1e6, 2e6, 80),
                           "ema_fast": close, "ema_medium": close, "rsi": 50.0, "atr": 1.0},
                          index=pd.date_range("2025-01-01", periods=80, freq="D"))
        full = StockMLPredictor(model_path="/nonexistent").compute_features(df)
        truncated = StockMLPredictor(model_path="/nonexistent").compute_features(df.iloc[:50])
        # Features at bar 49 must not change when future bars are appended
        pd.testing.assert_series_equal(full.iloc[49][list(truncated.columns)], truncated.iloc[49], check_names=False)

    def test_fallback_when_model_missing_or_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = StockMLPredictor(model_path=os.path.join(tmp, "nope.joblib"))
            corrupt_path = os.path.join(tmp, "corrupt.joblib")
            with open(corrupt_path, "wb") as fh:
                fh.write(b"not a model")
            corrupt = StockMLPredictor(model_path=corrupt_path)
            for predictor in (missing, corrupt):
                res = predictor.predict("AAPL", pd.DataFrame())
                self.assertEqual(res.probability, 0.50)
                self.assertTrue(res.is_approved)
                self.assertEqual(res.reason, "ML Model unavailable (fallback safe)")

    def test_train_and_predict_with_lightgbm(self):
        try:
            import lightgbm  # noqa: F401
        except (ImportError, OSError) as exc:  # OSError: missing libomp on macOS
            self.skipTest(f"lightgbm unavailable: {exc}")
        rng = np.random.default_rng(7)
        strat = SwingTrendFollowingStrategy()
        data = {}
        for sym in ("AAA", "BBB", "CCC"):
            close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, 300)))
            df = pd.DataFrame({
                "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                "volume": rng.uniform(1e6, 3e6, 300),
            }, index=pd.date_range("2024-01-01", periods=300, freq="B"))
            data[sym] = strat.calculate_indicators(df)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stock_lgbm_model.joblib")
            StockMLPredictor(model_path=path).train_model(data, n_splits=5)
            self.assertTrue(os.path.exists(path))
            res = StockMLPredictor(model_path=path).predict("AAA", data["AAA"])  # reload from disk

        self.assertIsInstance(res, MLPredictionResult)
        self.assertIsInstance(res.probability, float)
        self.assertGreaterEqual(res.probability, 0.0)
        self.assertLessEqual(res.probability, 1.0)
        self.assertEqual(res.is_approved, res.probability >= 0.60)
        self.assertRegex(res.reason, r"ML (Confirmed|Rejected) \(\d+%\)")


if __name__ == "__main__":
    unittest.main()
