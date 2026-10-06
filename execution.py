"""Order execution engine for Alpaca Trading API.

Manages broker connectivity, checks active positions and pending orders
to avoid double-allocation, and dispatches bracket orders with entry,
stop loss, and take profit legs simultaneously.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Set

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from config import AlpacaConfig, ExecutionConfig
from models import AccountState, ExecutionResult, PositionSize
from risk_manager import RiskManager

logger = logging.getLogger(__name__)


class AlpacaExecutionHandler:
    """Manages order execution and brokerage interaction with Alpaca."""

    def __init__(
        self,
        alpaca_config: AlpacaConfig,
        execution_config: ExecutionConfig,
        risk_manager: RiskManager,
    ) -> None:
        """Initialize the execution client.

        Args:
            alpaca_config: Alpaca API credentials and mode.
            execution_config: Execution preferences (order type, time in force, dry-run).
            risk_manager: Risk manager instance for account state assessment.
        """
        self.alpaca_config = alpaca_config
        self.execution_config = execution_config
        self.risk_manager = risk_manager

        self._client = TradingClient(
            api_key=alpaca_config.api_key,
            secret_key=alpaca_config.secret_key,
            paper=alpaca_config.paper,
        )
        logger.info(
            "AlpacaExecutionHandler initialized in %s mode",
            "PAPER" if alpaca_config.paper else "LIVE",
        )

    def get_account_state(self) -> AccountState:
        """Retrieve real-time account balances and evaluate health status.

        Returns:
            AccountState: Current account snapshot including drawdown and kill-switch status.
        """
        raw_account = self._client.get_account()
        equity = float(raw_account.equity)
        last_equity = float(raw_account.last_equity)
        cash = float(raw_account.cash)
        buying_power = float(raw_account.buying_power)

        return self.risk_manager.evaluate_account_health(
            equity=equity,
            last_equity=last_equity,
            cash=cash,
            buying_power=buying_power,
        )

    def get_active_symbols(self) -> Set[str]:
        """Fetch all symbols with an open position or active pending orders.

        Returns:
            Set[str]: Symbols currently exposed to market risk.
        """
        active_symbols: Set[str] = set()

        # 1. Fetch Open Positions
        try:
            positions = self._client.get_all_positions()
            for pos in positions:
                active_symbols.add(pos.symbol.upper())
        except Exception as exc:
            logger.error("Failed to query open positions from Alpaca: %s", exc)

        # 2. Fetch Open/Pending Orders
        try:
            filter_req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self._client.get_orders(filter=filter_req)
            for ord_item in orders:
                active_symbols.add(ord_item.symbol.upper())
        except Exception as exc:
            logger.error("Failed to query open orders from Alpaca: %s", exc)

        return active_symbols

    def get_positions_with_bracket_legs(self) -> List[Dict[str, Any]]:
        """Retrieve open positions enriched with Stop Loss and Take Profit levels from open orders.

        Returns:
            List[Dict[str, Any]]: Position details with stop loss and take profit targets.
        """
        try:
            positions = self._client.get_all_positions()
        except Exception as exc:
            logger.error("Failed to query positions: %s", exc)
            return []

        # Query open orders to match child Stop Loss and Take Profit legs
        orders_by_symbol: Dict[str, Dict[str, Optional[float]]] = {}
        try:
            open_orders = self._client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
            for ord_item in open_orders:
                sym = ord_item.symbol.upper()
                if sym not in orders_by_symbol:
                    orders_by_symbol[sym] = {"stop_loss": None, "take_profit": None}

                if getattr(ord_item, "stop_price", None) is not None:
                    orders_by_symbol[sym]["stop_loss"] = float(ord_item.stop_price)
                elif getattr(ord_item, "limit_price", None) is not None and ord_item.side == OrderSide.SELL:
                    orders_by_symbol[sym]["take_profit"] = float(ord_item.limit_price)
        except Exception as exc:
            logger.error("Failed to query open orders for bracket legs: %s", exc)

        results = []
        for pos in positions:
            sym = pos.symbol.upper()
            legs = orders_by_symbol.get(sym, {"stop_loss": None, "take_profit": None})
            unrealized_plpc = float(pos.unrealized_plpc) * 100.0 if pos.unrealized_plpc is not None else 0.0
            results.append({
                "symbol": sym,
                "qty": float(pos.qty),
                "side": str(pos.side),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": unrealized_plpc,
                "change_today": float(pos.change_today) * 100.0 if getattr(pos, "change_today", None) is not None else 0.0,
                "stop_loss": legs.get("stop_loss"),
                "take_profit": legs.get("take_profit"),
            })
        return results

    def has_open_position_or_order(self, symbol: str) -> bool:
        """Check if a ticker is already in portfolio or has pending orders.

        Args:
            symbol: Ticker symbol to verify.

        Returns:
            bool: True if symbol is already active.
        """
        return symbol.upper() in self.get_active_symbols()

    def submit_bracket_order(self, position: PositionSize) -> ExecutionResult:
        """Dispatch a Bracket Order (Entry + Take Profit + Stop Loss) to Alpaca.

        Args:
            position: Calculated position parameters and price targets.

        Returns:
            ExecutionResult: Execution details or error diagnostic.
        """
        symbol = position.symbol.upper()

        # Pre-execution validation
        if not position.is_valid:
            logger.warning("Order dispatch rejected for %s: %s", symbol, position.rejection_reason)
            return ExecutionResult(
                symbol=symbol,
                success=False,
                status="VALIDATION_FAILED",
                submitted_qty=0.0,
                error_message=position.rejection_reason,
            )

        # Idempotency client order identifier
        client_order_id = f"bot_{symbol}_{uuid.uuid4().hex[:8]}"

        # Resolve TimeInForce
        tif_map = {
            "gtc": TimeInForce.GTC,
            "day": TimeInForce.DAY,
        }
        tif = tif_map.get(self.execution_config.time_in_force.lower(), TimeInForce.GTC)

        # Format quantities: Whole shares for bracket orders in Alpaca
        order_qty = int(position.shares) if not position.is_fractional else position.shares

        # Construct bracket legs
        take_profit = TakeProfitRequest(limit_price=position.take_profit_price)
        stop_loss = StopLossRequest(stop_price=position.stop_loss_price)

        # Construct parent order request
        if self.execution_config.entry_order_type.lower() == "limit":
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=order_qty,
                side=OrderSide.BUY,
                time_in_force=tif,
                limit_price=position.entry_price,
                order_class=OrderClass.BRACKET,
                take_profit=take_profit,
                stop_loss=stop_loss,
                client_order_id=client_order_id,
            )
        else:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=order_qty,
                side=OrderSide.BUY,
                time_in_force=tif,
                order_class=OrderClass.BRACKET,
                take_profit=take_profit,
                stop_loss=stop_loss,
                client_order_id=client_order_id,
            )

        # Dry run simulation mode
        if self.execution_config.dry_run:
            logger.info(
                "[DRY-RUN] Bracket order simulated for %s: qty=%s, entry=$%.2f, TP=$%.2f, SL=$%.2f, risk=$%.2f (R:R=%.1f)",
                symbol,
                order_qty,
                position.entry_price,
                position.take_profit_price,
                position.stop_loss_price,
                position.risk_amount_usd,
                position.risk_reward_ratio,
            )
            return ExecutionResult(
                symbol=symbol,
                success=True,
                order_id=f"dry_run_{uuid.uuid4().hex[:12]}",
                client_order_id=client_order_id,
                status="SIMULATED",
                submitted_qty=float(order_qty),
                raw_response={
                    "dry_run": True,
                    "symbol": symbol,
                    "qty": order_qty,
                    "tp": position.take_profit_price,
                    "sl": position.stop_loss_price,
                },
            )

        # Live / Paper API submission
        try:
            logger.info(
                "Submitting Bracket Order to Alpaca: %s | Qty: %s | Entry: $%.2f | SL: $%.2f | TP: $%.2f | Max Risk: $%.2f",
                symbol,
                order_qty,
                position.entry_price,
                position.stop_loss_price,
                position.take_profit_price,
                position.risk_amount_usd,
            )
            response = self._client.submit_order(order_data=order_request)

            logger.info(
                "Bracket Order accepted by Alpaca! ID: %s | Status: %s | Symbol: %s",
                response.id,
                response.status,
                symbol,
            )

            return ExecutionResult(
                symbol=symbol,
                success=True,
                order_id=str(response.id),
                client_order_id=str(response.client_order_id),
                status=str(response.status),
                submitted_qty=float(response.qty) if response.qty is not None else float(order_qty),
                raw_response=dict(response) if hasattr(response, "__iter__") else {},
            )

        except APIError as api_err:
            error_msg = str(api_err)
            status_code = getattr(api_err, "status_code", "4xx")
            logger.error(
                "Alpaca API rejected bracket order for %s [Code %s]: %s",
                symbol,
                status_code,
                error_msg,
            )
            return ExecutionResult(
                symbol=symbol,
                success=False,
                client_order_id=client_order_id,
                status="REJECTED",
                submitted_qty=float(order_qty),
                error_code=str(status_code),
                error_message=error_msg,
            )

        except Exception as exc:
            logger.error("Unexpected error submitting order for %s: %s", symbol, exc, exc_info=True)
            return ExecutionResult(
                symbol=symbol,
                success=False,
                client_order_id=client_order_id,
                status="ERROR",
                submitted_qty=float(order_qty),
                error_message=str(exc),
            )
