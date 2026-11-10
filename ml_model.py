"""Root alias re-exporting StockMLPredictor from src.ml_model."""
from src.ml_model import StockMLPredictor, MLPredictionResult, FEATURE_NAMES, create_labeled_dataset

__all__ = ["StockMLPredictor", "MLPredictionResult", "FEATURE_NAMES", "create_labeled_dataset"]
