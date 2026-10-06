"""Historical market data provider using Alpaca Historical Data API.

Retrieves daily aggregated bars (OHLCV) for equity assets, standardizes
DataFrame format, and validates sufficient history for indicator warmup.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from config import AlpacaConfig

logger = logging.getLogger(__name__)


class AlpacaDataProvider:
    """Encapsulates historical market data access via Alpaca-py SDK."""

    def __init__(self, config: AlpacaConfig) -> None:
        """Initialize the data client with Alpaca credentials and settings.

        Args:
            config: Alpaca connection configuration.
        """
        self.config = config
        self._client = StockHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
        )
        self._feed = DataFeed.SIP if config.data_feed.lower() == "sip" else DataFeed.IEX
        logger.info("AlpacaDataProvider initialized using feed: %s", self._feed.value)

    def _normalize_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardize the raw Alpaca DataFrame to a clean single-index DataFrame.

        Args:
            df: Raw DataFrame returned by Alpaca client.
            symbol: Symbol being normalized.

        Returns:
            pd.DataFrame: Sorted, indexed by timestamp, with standard OHLCV columns.
        """
        if df.empty:
            return pd.DataFrame()

        # Handle MultiIndex ('symbol', 'timestamp')
        if isinstance(df.index, pd.MultiIndex):
            if symbol in df.index.get_level_values(0):
                df_symbol = df.xs(symbol, level=0).copy()
            else:
                return pd.DataFrame()
        else:
            df_symbol = df.copy()

        # Normalize column names to lowercase
        df_symbol.columns = [str(col).lower() for col in df_symbol.columns]

        # Verify required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in df_symbol.columns]
        if missing:
            logger.warning("Symbol %s missing required OHLCV columns: %s", symbol, missing)
            return pd.DataFrame()

        # Ensure index is datetime and sorted chronologically
        if not isinstance(df_symbol.index, pd.DatetimeIndex):
            if "timestamp" in df_symbol.columns:
                df_symbol["timestamp"] = pd.to_datetime(df_symbol["timestamp"])
                df_symbol = df_symbol.set_index("timestamp")
            else:
                df_symbol.index = pd.to_datetime(df_symbol.index)

        df_symbol = df_symbol.sort_index()

        # Drop duplicate timestamps if any
        df_symbol = df_symbol[~df_symbol.index.duplicated(keep="last")]

        # Ensure numeric types
        for col in required_cols:
            df_symbol[col] = pd.to_numeric(df_symbol[col], errors="coerce")

        df_symbol = df_symbol.dropna(subset=required_cols)
        return df_symbol

    def get_daily_bars(
        self,
        symbol: str,
        lookback_days: int = 400,
        end_dt: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch daily bars for a single equity symbol.

        Args:
            symbol: Ticker symbol (e.g. 'SPY').
            lookback_days: Calendar days of history to request (defaults to 400 to cover 200 EMA).
            end_dt: End datetime for query (defaults to current UTC time).

        Returns:
            pd.DataFrame: Cleaned OHLCV DataFrame or empty DataFrame if retrieval fails.
        """
        if end_dt is None:
            end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=lookback_days)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                feed=self._feed,
            )
            bars = self._client.get_stock_bars(request)
            if bars is None or bars.df is None or bars.df.empty:
                logger.warning("No daily bars returned for %s", symbol)
                return pd.DataFrame()

            clean_df = self._normalize_dataframe(bars.df, symbol)
            logger.debug("Retrieved %d bars for symbol %s", len(clean_df), symbol)
            return clean_df

        except Exception as exc:
            logger.error("Error fetching daily bars for %s: %s", symbol, exc, exc_info=True)
            return pd.DataFrame()

    def get_multiple_daily_bars(
        self,
        symbols: List[str],
        lookback_days: int = 400,
        end_dt: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch daily bars for multiple equity symbols in a single batch request.

        Args:
            symbols: List of ticker symbols.
            lookback_days: Calendar days of history to request.
            end_dt: End datetime for query.

        Returns:
            Dict[str, pd.DataFrame]: Mapping from ticker symbol to cleaned OHLCV DataFrame.
        """
        if not symbols:
            return {}

        if end_dt is None:
            end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=lookback_days)

        result: Dict[str, pd.DataFrame] = {}

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                feed=self._feed,
            )
            bars = self._client.get_stock_bars(request)
            if bars is None or bars.df is None or bars.df.empty:
                logger.warning("Batch request returned no bars for %s", symbols)
                return {sym: pd.DataFrame() for sym in symbols}

            raw_df = bars.df
            for sym in symbols:
                clean_df = self._normalize_dataframe(raw_df, sym)
                result[sym] = clean_df

            return result

        except Exception as exc:
            logger.error("Batch fetch failed for %s: %s. Falling back to sequential fetch.", symbols, exc)
            for sym in symbols:
                result[sym] = self.get_daily_bars(sym, lookback_days, end_dt)
            return result
