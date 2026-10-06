"""Main orchestrator and CLI runner for the Alpaca Swing Trading Bot.

Runs on-demand or periodic scans evaluating a universe of liquid equities and ETFs
against the swing trend-following strategy and dispatching risk-managed bracket orders.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

import argparse
import logging
import sys
import time
from typing import Dict, List, Optional

from config import AppConfig, load_config
from data_provider import AlpacaDataProvider
from execution import AlpacaExecutionHandler
from models import AccountState, ExecutionResult, PositionSize, SignalType, TradingSignal
from risk_manager import RiskManager
from strategy import SwingTrendFollowingStrategy


def setup_logger(log_level_str: str) -> logging.Logger:
    """Configure structured logging output with timestamps and severity."""
    level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("TradingBot")


class SwingTradingBot:
    """Coordinates data fetching, strategy evaluation, risk controls, and execution."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize all subsystems with the given configuration.

        Args:
            config: Root application configuration.
        """
        self.config = config
        self.logger = setup_logger(config.log_level)

        self.logger.info("Initializing Swing Trading Bot...")
        self.data_provider = AlpacaDataProvider(config.alpaca)
        self.strategy = SwingTrendFollowingStrategy(config.strategy)
        self.risk_manager = RiskManager(config.risk)
        self.execution_handler = AlpacaExecutionHandler(
            alpaca_config=config.alpaca,
            execution_config=config.execution,
            risk_manager=self.risk_manager,
        )
        self.is_active: bool = True
        self.is_scanning: bool = False
        self.last_scan_time: Optional[datetime] = None
        self.last_results: List[Dict[str, Any]] = []

    def scan_symbol(
        self,
        symbol: str,
        account: AccountState,
        active_symbols: set[str],
    ) -> tuple[TradingSignal, Optional[PositionSize], Optional[ExecutionResult]]:
        """Run the quantitative pipeline on a single symbol.

        Args:
            symbol: Market ticker.
            account: Current account health state.
            active_symbols: Set of symbols with current exposure or pending orders.

        Returns:
            Tuple of (TradingSignal, PositionSize or None, ExecutionResult or None)
        """
        symbol = symbol.upper()
        self.logger.info("--> Evaluating symbol: %s", symbol)

        # 1. Check existing position or pending order
        if symbol in active_symbols:
            self.logger.info(
                "[%s] Already active in portfolio or has open order. Skipping new entry.",
                symbol,
            )
            dummy_signal = TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                timestamp=pd.Timestamp.now().to_pydatetime() if "pd" in sys.modules else None,  # type: ignore
                close_price=0.0,
                entry_price=0.0,
                ema_20=0.0,
                ema_50=0.0,
                ema_200=0.0,
                rsi_14=0.0,
                atr_14=0.0,
                prev_high=0.0,
                reason="Already active in portfolio or pending order exists",
            )
            return dummy_signal, None, None

        # 2. Fetch daily bars
        df = self.data_provider.get_daily_bars(symbol, lookback_days=400)
        if df.empty:
            self.logger.warning("[%s] Could not retrieve daily bars. Skipping.", symbol)
            return (
                TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    timestamp=None,  # type: ignore
                    close_price=0.0,
                    entry_price=0.0,
                    ema_20=0.0,
                    ema_50=0.0,
                    ema_200=0.0,
                    rsi_14=0.0,
                    atr_14=0.0,
                    prev_high=0.0,
                    reason="No historical data returned",
                ),
                None,
                None,
            )

        # 3. Strategy Evaluation
        signal = self.strategy.evaluate(symbol, df)
        self.logger.info(
            "[%s] Signal: %s | Price: $%.2f | RSI: %.1f | Reason: %s",
            symbol,
            signal.signal_type.value,
            signal.close_price,
            signal.rsi_14,
            signal.reason,
        )

        if signal.signal_type != SignalType.BUY:
            return signal, None, None

        # 4. Risk Assessment & Bracket Level Calculation
        position_size = self.risk_manager.calculate_position_size(signal, account)
        if not position_size.is_valid:
            self.logger.warning(
                "[%s] Sizing rejected by RiskManager: %s",
                symbol,
                position_size.rejection_reason,
            )
            return signal, position_size, None

        self.logger.info(
            "[%s] Risk Sizing Approved: %d shares | Entry: $%.2f | SL: $%.2f (-$%.2f/sh) | TP: $%.2f (+$%.2f/sh) | Risk: $%.2f (1.0%%)",
            symbol,
            int(position_size.shares),
            position_size.entry_price,
            position_size.stop_loss_price,
            position_size.risk_per_share,
            position_size.take_profit_price,
            position_size.reward_per_share,
            position_size.risk_amount_usd,
        )

        # 5. Order Execution (Bracket Order)
        exec_result = self.execution_handler.submit_bracket_order(position_size)
        return signal, position_size, exec_result

    def run_scan(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute a full market scan over target tickers.

        Args:
            symbols: Optional override list of tickers.

        Returns:
            Dict containing execution summary and metrics.
        """
        if not self.is_active:
            self.logger.warning("Bot is currently PAUSED. Scan aborted without placing orders.")
            account = self.execution_handler.get_account_state()
            return {
                "aborted": True,
                "reason": "Bot is paused",
                "account": account,
                "results": self.last_results,
            }

        scan_symbols = symbols or self.config.symbols
        self.logger.info("=" * 60)
        self.logger.info("STARTING MARKET SCAN: %s", scan_symbols)
        self.logger.info("=" * 60)

        self.is_scanning = True
        try:
            # 1. Query real-time account state & check Kill-Switch
            account = self.execution_handler.get_account_state()
            self.logger.info(
                "Account Equity: $%.2f | Cash: $%.2f | Buying Power: $%.2f | Intraday Drawdown: %.2f%%",
                account.equity,
                account.cash,
                account.buying_power,
                account.intraday_drawdown_pct * 100,
            )

            if account.kill_switch_active:
                self.logger.critical(
                    "ABORTING SCAN: Intraday Drawdown Kill-Switch is active (%.2f%% >= %.2f%%).",
                    account.intraday_drawdown_pct * 100,
                    self.config.risk.max_intraday_drawdown_pct * 100,
                )
                return {
                    "aborted": True,
                    "reason": "Kill-switch triggered",
                    "drawdown_pct": account.intraday_drawdown_pct,
                    "account": account,
                    "results": self.last_results,
                }

            # 2. Query currently active positions and orders
            active_symbols = self.execution_handler.get_active_symbols()
            self.logger.info("Active symbols already in portfolio/orders: %s", list(active_symbols))

            results_summary = []

            # 3. Process each ticker
            for sym in scan_symbols:
                try:
                    signal, pos_size, exec_res = self.scan_symbol(sym, account, active_symbols)
                    results_summary.append({
                        "symbol": sym,
                        "signal": signal.signal_type.value,
                        "price": signal.close_price,
                        "rsi": round(signal.rsi_14, 1),
                        "ema_20": round(signal.ema_20, 2),
                        "ema_50": round(signal.ema_50, 2),
                        "ema_200": round(signal.ema_200, 2),
                        "atr": round(signal.atr_14, 2),
                        "shares": int(pos_size.shares) if pos_size else 0,
                        "stop_loss": pos_size.stop_loss_price if pos_size else None,
                        "take_profit": pos_size.take_profit_price if pos_size else None,
                        "executed": exec_res.success if exec_res else False,
                        "order_id": exec_res.order_id if exec_res else None,
                        "status": exec_res.status if exec_res else ("NO_ORDER" if signal.signal_type != SignalType.BUY else "REJECTED"),
                        "reason": signal.reason if signal.signal_type != SignalType.BUY else (pos_size.rejection_reason if pos_size and not pos_size.is_valid else "Setup Valid & Order Dispatched"),
                    })
                except Exception as exc:
                    self.logger.error("Unexpected error scanning %s: %s", sym, exc, exc_info=True)

            from datetime import datetime, timezone
            self.last_scan_time = datetime.now(timezone.utc)
            self.last_results = results_summary

            # 4. Print Summary Report
            self.logger.info("=" * 60)
            self.logger.info("SCAN COMPLETE - SUMMARY REPORT")
            self.logger.info("=" * 60)
            for item in results_summary:
                status_tag = "[ORDER SENT]" if item["executed"] else f"[{item['status']}]"
                self.logger.info(
                    "%-6s | Signal: %-4s | Shares: %-3d | %-12s | %s",
                    item["symbol"],
                    item["signal"],
                    int(item["shares"]),
                    status_tag,
                    item["reason"][:75],
                )
            self.logger.info("=" * 60)

            return {
                "account": account,
                "results": results_summary,
                "aborted": False,
                "scan_time": self.last_scan_time.isoformat(),
            }
        finally:
            self.is_scanning = False


def parse_arguments() -> argparse.Namespace:
    """Parse command line flags."""
    parser = argparse.ArgumentParser(
        description="Alpaca Quantitative Swing Trading Bot (1D Pullback + Bracket Orders)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated ticker list override (e.g. 'SPY,AAPL,MSFT')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without sending orders to Alpaca",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run scans continuously in a loop",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Scan interval in seconds when running continuously (default: 3600s / 1 hour)",
    )
    return parser.parse_args()


def main() -> None:
    """Main application entry point."""
    args = parse_arguments()

    # Load configuration
    try:
        config = load_config()
    except Exception as exc:
        print(f"Configuration Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Apply CLI overrides
    if args.dry_run:
        from dataclasses import replace
        config = replace(config, execution=replace(config.execution, dry_run=True))

    tickers = [s.strip().upper() for s in args.tickers.split(",") if s.strip()] if args.tickers else None

    bot = SwingTradingBot(config)

    if args.continuous:
        bot.logger.info("Running in CONTINUOUS mode (interval: %ds)... Press Ctrl+C to stop.", args.interval)
        try:
            while True:
                bot.run_scan(symbols=tickers)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            bot.logger.info("Continuous scan stopped by user.")
    else:
        bot.logger.info("Running ON-DEMAND scan cycle...")
        bot.run_scan(symbols=tickers)


if __name__ == "__main__":
    main()
