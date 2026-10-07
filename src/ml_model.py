"""Machine Learning Swing Direction Predictor using Random Forest.

Evaluates daily (1D) market bars using a feature pipeline:
- Past 5-day and 20-day returns
- Normalized distances to EMA(20) and EMA(50)
- Wilder's RSI(14)
- Normalized ATR(14) volatility

Confirms BUY entries only when bullish probability >= 0.60 (60%).
Automatically trains and persists 'models/stock_rf_model.joblib' if not present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("MLModel")

FEATURE_NAMES = [
    "ret_5d",
    "ret_20d",
    "dist_ema20",
    "dist_ema50",
    "rsi_14",
    "norm_atr",
]


@dataclass(frozen=True)
class MLPredictionResult:
    """Inference output from the Random Forest model.

    Attributes:
        symbol: Market ticker evaluated.
        probability: Probability of bullish swing outcome (0.0 to 1.0).
        is_approved: True if probability meets or exceeds threshold (>= 0.60).
        threshold: Required minimum probability threshold (0.60).
        features: Computed feature values for the evaluated bar.
        reason: Diagnostic explanation.
    """
    symbol: str
    probability: float
    is_approved: bool
    threshold: float = 0.60
    features: Dict[str, float] = field(default_factory=dict)
    reason: str = "ML Confirmed"


class StockMLPredictor:
    """Manages feature engineering, training, serialization, and inference for swing direction."""

    def __init__(
        self,
        model_path: str = "models/stock_rf_model.joblib",
        approval_threshold: float = 0.60,
    ) -> None:
        """Initialize predictor.

        Args:
            model_path: File path to persisted joblib model.
            approval_threshold: Minimum bullish probability required for BUY confirmation.
        """
        self.model_path = model_path
        self.approval_threshold = approval_threshold
        self.pipeline: Optional[Pipeline] = None

        # Load existing model if available
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Load model from disk if present."""
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                logger.info("Loaded pre-trained ML model from %s", self.model_path)
            except Exception as exc:
                logger.warning("Failed to load existing model from %s (%s). Will re-train.", self.model_path, exc)
                self.pipeline = None

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the 6 model features from an OHLCV DataFrame with indicators.

        Args:
            df: DataFrame containing 'close', 'ema_fast', 'ema_medium', 'rsi', 'atr'.

        Returns:
            pd.DataFrame: Enriched DataFrame with feature columns.
        """
        out = df.copy()

        # 1. Past 5-day and 20-day returns
        out["ret_5d"] = out["close"].pct_change(5)
        out["ret_20d"] = out["close"].pct_change(20)

        # 2. Normalized distance to EMA20 and EMA50
        ema_fast_col = "ema_fast" if "ema_fast" in out.columns else "ema_20"
        ema_med_col = "ema_medium" if "ema_medium" in out.columns else "ema_50"

        if ema_fast_col in out.columns:
            out["dist_ema20"] = (out["close"] - out[ema_fast_col]) / out[ema_fast_col]
        else:
            out["dist_ema20"] = 0.0

        if ema_med_col in out.columns:
            out["dist_ema50"] = (out["close"] - out[ema_med_col]) / out[ema_med_col]
        else:
            out["dist_ema50"] = 0.0

        # 3. RSI(14)
        rsi_col = "rsi" if "rsi" in out.columns else "rsi_14"
        out["rsi_14"] = out[rsi_col] if rsi_col in out.columns else 50.0

        # 4. Normalized ATR volatility (ATR / Close)
        atr_col = "atr" if "atr" in out.columns else "atr_14"
        if atr_col in out.columns:
            out["norm_atr"] = out[atr_col] / out["close"]
        else:
            out["norm_atr"] = 0.02

        return out

    def train_model(
        self,
        training_data_dict: Dict[str, pd.DataFrame],
        forward_days: int = 5,
        target_return_threshold: float = 0.010,
    ) -> Pipeline:
        """Train a regularized Random Forest model on historical multi-asset data.

        Target Definition:
            Binary label = 1 if forward 5-day return > 1.0% (profitable swing breakout), else 0.

        Args:
            training_data_dict: Mapping of ticker to indicator-enriched DataFrame.
            forward_days: Prediction horizon in sessions.
            target_return_threshold: Minimum fractional gain to classify as bullish.

        Returns:
            Trained scikit-learn Pipeline.
        """
        X_all: List[pd.DataFrame] = []
        y_all: List[pd.Series] = []

        logger.info("Building training dataset from %d historical assets...", len(training_data_dict))

        for sym, raw_df in training_data_dict.items():
            if raw_df is None or len(raw_df) < 50:
                continue

            df = self.compute_features(raw_df)

            # Construct forward return target
            fwd_ret = (df["close"].shift(-forward_days) - df["close"]) / df["close"]
            df["target"] = (fwd_ret >= target_return_threshold).astype(int)

            # Drop rows with NaN in features or target
            cols_needed = FEATURE_NAMES + ["target"]
            clean_df = df.dropna(subset=cols_needed)

            if len(clean_df) > 30:
                X_all.append(clean_df[FEATURE_NAMES])
                y_all.append(clean_df["target"])

        if not X_all:
            raise ValueError("Insufficient training samples available across provided datasets.")

        X = pd.concat(X_all, ignore_index=True)
        y = pd.concat(y_all, ignore_index=True)

        logger.info(
            "Training Random Forest on %d samples (Bullish class ratio: %.1f%%)",
            len(X),
            (y.mean() * 100),
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ])

        pipeline.fit(X, y)

        # Save model
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump(pipeline, self.model_path)
        logger.info("Random Forest model successfully saved to %s", self.model_path)

        self.pipeline = pipeline
        return pipeline

    def ensure_model_ready(self, data_provider: Any, symbols: Optional[List[str]] = None) -> None:
        """Verify model exists; if not, automatically fetch data and train.

        Args:
            data_provider: Instance of AlpacaDataProvider.
            symbols: Seed tickers to train on (defaults to core ETFs and market leaders).
        """
        if self.pipeline is not None and os.path.exists(self.model_path):
            return

        seed_symbols = symbols or [
            "SPY", "QQQ", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
            "META", "JPM", "XOM", "CAT", "V", "LLY", "UNH"
        ]
        logger.info("Model not found at %s. Automatically training on %s...", self.model_path, seed_symbols)

        training_data: Dict[str, pd.DataFrame] = {}
        from strategy import SwingTrendFollowingStrategy
        strat = SwingTrendFollowingStrategy()

        for sym in seed_symbols:
            try:
                bars = data_provider.get_daily_bars(sym, lookback_days=450)
                if not bars.empty and len(bars) >= 150:
                    enriched = strat.calculate_indicators(bars)
                    training_data[sym] = enriched
            except Exception as exc:
                logger.warning("Could not fetch training bars for %s: %s", sym, exc)

        if training_data:
            self.train_model(training_data)
        else:
            logger.error("Failed to collect historical bars for initial ML training.")

    def predict(self, symbol: str, df: pd.DataFrame) -> MLPredictionResult:
        """Evaluate latest bar of symbol and predict bullish probability.

        Args:
            symbol: Ticker symbol.
            df: Historical DataFrame with technical indicators.

        Returns:
            MLPredictionResult with probability and approval verdict.
        """
        sym = symbol.upper()

        if self.pipeline is None:
            if os.path.exists(self.model_path):
                self._load_or_initialize()

        if self.pipeline is None:
            logger.warning("ML model not initialized. Defaulting to neutral probability (0.50).")
            return MLPredictionResult(
                symbol=sym,
                probability=0.50,
                is_approved=False,
                threshold=self.approval_threshold,
                reason="ML Model Not Loaded",
            )

        if df is None or len(df) < 25:
            return MLPredictionResult(
                symbol=sym,
                probability=0.50,
                is_approved=False,
                threshold=self.approval_threshold,
                reason="Insufficient bars for ML feature calculation",
            )

        feat_df = self.compute_features(df)
        latest_row = feat_df.iloc[-1]

        # Check for NaN in features
        feature_vals: Dict[str, float] = {}
        has_nan = False
        for f in FEATURE_NAMES:
            val = float(latest_row[f]) if f in latest_row else np.nan
            if np.isnan(val):
                has_nan = True
            feature_vals[f] = val

        if has_nan:
            return MLPredictionResult(
                symbol=sym,
                probability=0.50,
                is_approved=False,
                threshold=self.approval_threshold,
                features=feature_vals,
                reason="Features contain NaN",
            )

        X_input = pd.DataFrame([feature_vals])[FEATURE_NAMES]

        try:
            probabilities = self.pipeline.predict_proba(X_input)[0]
            # Probability for class 1 (bullish swing)
            bullish_prob = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
            prob_pct = int(round(bullish_prob * 100))

            is_approved = bullish_prob >= self.approval_threshold

            if is_approved:
                reason = f"ML Confirmed ({prob_pct}%)"
            else:
                reason = f"ML Rejected ({prob_pct}%)"

            return MLPredictionResult(
                symbol=sym,
                probability=bullish_prob,
                is_approved=is_approved,
                threshold=self.approval_threshold,
                features=feature_vals,
                reason=reason,
            )
        except Exception as exc:
            logger.error("Error during ML inference for %s: %s", sym, exc)
            return MLPredictionResult(
                symbol=sym,
                probability=0.50,
                is_approved=False,
                threshold=self.approval_threshold,
                features=feature_vals,
                reason=f"ML Inference Error: {exc}",
            )
