"""Root alias re-exporting StockMLPredictor from src.ml_model."""
from src.ml_model import StockMLPredictor, MLPredictionResult, FEATURE_NAMES

__all__ = ["StockMLPredictor", "MLPredictionResult", "FEATURE_NAMES"]
