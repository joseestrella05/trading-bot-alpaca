"""Machine Learning Swing Trade Predictor using LightGBM.

Features (all computed from past/current bars only, no look-ahead):
- Cumulative log returns over 5 and 20 bars
- Normalized distances to EMA(20) and EMA(50)
- RSI(14) scaled to [0, 1]
- Normalized ATR(14) volatility (ATR / close)
- Volume ratio vs 20-bar average

Labels use a triple-barrier scheme aligned with the strategy's bracket orders:
take profit at +profit_take_atr * ATR, stop loss at -stop_loss_atr * ATR, time barrier at N bars.

Training uses TimeSeriesSplit cross-validation (no shuffling). If the model file is missing or
fails to load, predict() degrades to a neutral, non-blocking result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger("MLModel")

DEFAULT_MODEL_PATH = "models/stock_lgbm_model.joblib"

FEATURE_NAMES = [
    "ret_5d",
    "ret_20d",
    "dist_ema20",
    "dist_ema50",
    "rsi_14",
    "norm_atr",
    "vol_ratio",
]


@dataclass(frozen=True)
class MLPredictionResult:
    """Inference output from the ML model.

    Attributes:
        symbol: Market ticker evaluated.
        probability: Probability of a successful swing trade (0.0 to 1.0).
        is_approved: True if probability meets or exceeds threshold.
        threshold: Required minimum probability threshold.
        features: Computed feature values for the evaluated bar.
        reason: Diagnostic explanation.
    """
    symbol: str
    probability: float
    is_approved: bool
    threshold: float = 0.60
    features: Dict[str, float] = field(default_factory=dict)
    reason: str = "ML Confirmed"


def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    """First column name present in df (indicator columns differ between strategy and callers)."""
    return next((n for n in names if n in df.columns), None)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add FEATURE_NAMES columns. Every feature uses only bars <= t (positive shifts / trailing windows).

    Args:
        df: DataFrame with 'close' plus indicator columns ('ema_fast'/'ema_20', 'ema_medium'/'ema_50',
            'rsi'/'rsi_14', 'atr'/'atr_14') and optionally 'volume'.
    """
    out = df.copy()
    log_close = np.log(out["close"])
    out["ret_5d"] = log_close - log_close.shift(5)
    out["ret_20d"] = log_close - log_close.shift(20)

    ema20 = _col(out, "ema_fast", "ema_20")
    ema50 = _col(out, "ema_medium", "ema_50")
    out["dist_ema20"] = (out["close"] - out[ema20]) / out[ema20] if ema20 else 0.0
    out["dist_ema50"] = (out["close"] - out[ema50]) / out[ema50] if ema50 else 0.0

    rsi = _col(out, "rsi", "rsi_14")
    out["rsi_14"] = out[rsi] / 100.0 if rsi else 0.5

    atr = _col(out, "atr", "atr_14")
    out["norm_atr"] = out[atr] / out["close"] if atr else 0.02

    if "volume" in out.columns:
        out["vol_ratio"] = out["volume"] / out["volume"].rolling(20).mean()
    else:
        out["vol_ratio"] = 1.0
    out["vol_ratio"] = out["vol_ratio"].replace([np.inf, -np.inf], np.nan)
    return out


def create_labeled_dataset(
    df: pd.DataFrame,
    holding_period: int = 10,
    profit_take_atr: float = 2.0,
    stop_loss_atr: float = 1.5,
) -> pd.DataFrame:
    """Build features + triple-barrier 'target' for each bar t (entry at close[t]).

    target = 1 if high touches entry + profit_take_atr*ATR before low touches entry - stop_loss_atr*ATR
    within the next `holding_period` bars, or if neither is touched and close[t+N] > entry.
    target = 0 if the stop is touched first (a bar touching both counts as stop: conservative),
    or the time barrier expires with close[t+N] <= entry.
    The last `holding_period` bars (incomplete future window) are dropped.

    Args:
        df: OHLCV DataFrame enriched with indicators (needs 'high', 'low', 'close', 'atr'/'atr_14').
    """
    atr_name = _col(df, "atr", "atr_14")
    if atr_name is None:
        raise ValueError("create_labeled_dataset requires an 'atr' or 'atr_14' column.")

    out = compute_features(df)
    close, high, low, atr = out["close"], out["high"], out["low"], out[atr_name]
    take_profit = close + profit_take_atr * atr
    stop_loss = close - stop_loss_atr * atr

    target = pd.Series(np.nan, index=out.index)
    resolved = pd.Series(False, index=out.index)
    for k in range(1, holding_period + 1):
        hit_sl = (low.shift(-k) <= stop_loss) & ~resolved
        hit_tp = (high.shift(-k) >= take_profit) & ~resolved & ~hit_sl
        target[hit_sl] = 0.0
        target[hit_tp] = 1.0
        resolved |= hit_sl | hit_tp

    final_close = close.shift(-holding_period)
    expired = ~resolved
    target[expired] = (final_close[expired] > close[expired]).astype(float)
    target[final_close.isna()] = np.nan  # incomplete window: label unknown

    out["target"] = target
    return out.dropna(subset=FEATURE_NAMES + ["target", atr_name]).astype({"target": int})


class StockMLPredictor:
    """Manages feature engineering, training, serialization, and inference for swing trades."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        approval_threshold: float = 0.60,
    ) -> None:
        """Initialize predictor.

        Args:
            model_path: File path to persisted joblib model.
            approval_threshold: Minimum success probability required for BUY confirmation.
        """
        self.model_path = model_path
        self.approval_threshold = approval_threshold
        self.pipeline: Optional[Any] = None  # fitted classifier exposing predict_proba
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Load model from disk if present; any failure (incl. missing lightgbm/libomp) leaves it None."""
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                logger.info("Loaded pre-trained ML model from %s", self.model_path)
            except Exception as exc:
                logger.warning("Failed to load model from %s (%s). Using fallback.", self.model_path, exc)
                self.pipeline = None

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute model features (see module-level compute_features)."""
        return compute_features(df)

    def train_model(
        self,
        historical_data_dict: Dict[str, pd.DataFrame],
        n_splits: int = 5,
        holding_period: int = 10,
    ) -> Any:
        """Train LGBMClassifier with TimeSeriesSplit CV on triple-barrier labels, then persist it.

        Args:
            historical_data_dict: Mapping of ticker to indicator-enriched OHLCV DataFrame.
            n_splits: Number of chronological CV folds.
            holding_period: Triple-barrier time horizon in bars.

        Returns:
            The fitted LGBMClassifier.
        """
        from lightgbm import LGBMClassifier  # lazy: native lib (libomp) may be missing on the host

        datasets: List[pd.DataFrame] = []
        for sym, raw_df in historical_data_dict.items():
            if raw_df is None or len(raw_df) < 50:
                continue
            labeled = create_labeled_dataset(raw_df, holding_period=holding_period)
            if len(labeled) > 30:
                datasets.append(labeled[FEATURE_NAMES + ["target"]])

        if not datasets:
            raise ValueError("Insufficient training samples available across provided datasets.")

        # Stable sort by timestamp interleaves tickers chronologically so folds never train on the future.
        data = pd.concat(datasets).sort_index(kind="mergesort")
        X, y = data[FEATURE_NAMES], data["target"]

        def make_model() -> Any:
            return LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                subsample_freq=1,  # LightGBM ignores subsample unless bagging frequency > 0
                random_state=42,
                verbose=-1,
            )

        # Gap of one label horizon (in rows across all tickers) so test labels don't overlap train windows.
        splitter = TimeSeriesSplit(n_splits=n_splits, gap=holding_period * len(datasets))
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                logger.info("CV fold %d skipped: single class.", fold)
                continue
            fold_model = make_model().fit(X.iloc[train_idx], y_train)
            auc = roc_auc_score(y_test, fold_model.predict_proba(X.iloc[test_idx])[:, 1])
            logger.info("CV fold %d: train=%d test=%d AUC=%.3f", fold, len(train_idx), len(test_idx), auc)

        logger.info("Training LightGBM on %d samples (success ratio: %.1f%%)", len(X), y.mean() * 100)
        model = make_model().fit(X, y)

        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump(model, self.model_path)
        logger.info("LightGBM model saved to %s", self.model_path)

        self.pipeline = model
        return model

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
                    training_data[sym] = strat.calculate_indicators(bars)
            except Exception as exc:
                logger.warning("Could not fetch training bars for %s: %s", sym, exc)

        if training_data:
            self.train_model(training_data)
        else:
            logger.error("Failed to collect historical bars for initial ML training.")

    def predict(self, symbol: str, df: pd.DataFrame) -> MLPredictionResult:
        """Evaluate latest bar of symbol and predict trade success probability.

        Args:
            symbol: Ticker symbol.
            df: Historical DataFrame with technical indicators.

        Returns:
            MLPredictionResult with probability and approval verdict.
        """
        sym = symbol.upper()

        if self.pipeline is None:
            self._load_or_initialize()

        if self.pipeline is None:
            logger.warning("ML model unavailable. Using neutral fallback (0.50, approved).")
            return MLPredictionResult(
                symbol=sym,
                probability=0.50,
                is_approved=True,
                threshold=self.approval_threshold,
                reason="ML Model unavailable (fallback safe)",
            )

        if df is None or len(df) < 25:
            return MLPredictionResult(
                symbol=sym,
                probability=0.50,
                is_approved=False,
                threshold=self.approval_threshold,
                reason="Insufficient bars for ML feature calculation",
            )

        latest_row = self.compute_features(df).iloc[-1]
        feature_vals: Dict[str, float] = {f: float(latest_row[f]) for f in FEATURE_NAMES}

        if any(np.isnan(v) for v in feature_vals.values()):
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
            success_prob = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
            prob_pct = int(round(success_prob * 100))
            is_approved = success_prob >= self.approval_threshold
            return MLPredictionResult(
                symbol=sym,
                probability=success_prob,
                is_approved=is_approved,
                threshold=self.approval_threshold,
                features=feature_vals,
                reason=f"ML Confirmed ({prob_pct}%)" if is_approved else f"ML Rejected ({prob_pct}%)",
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
