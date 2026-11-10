"""Quantitative Risk Management Engine.

Enforces strict portfolio safety controls:
1. Risk per trade capped at 1.0% of portfolio equity.
2. Exact position sizing formula:
   shares = (capital * 0.01) / abs(entry_price - stop_loss_price)
3. Dynamic Bracket Order levels (Stop Loss = 1.5 * ATR(14), Take Profit = 2:1 ratio).
4. Intraday Drawdown Kill-Switch (halts execution if drawdown >= 2.5%).
5. Buying power and concentration limits.
6. Portfolio correlation gate (blocks BUYs highly correlated with open positions).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd

from config import RiskConfig
from models import AccountState, MarketRegime, PositionSize, TradingSignal, SignalType
from risk_engine import AdaptiveRiskEngine

logger = logging.getLogger(__name__)


class RiskManager:
    """Calculates risk-adjusted position sizing and monitors drawdown safety thresholds."""

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        """Initialize risk manager.

        Args:
            config: Risk configuration parameters.
        """
        self.config = config or RiskConfig()
        self.adaptive_engine = AdaptiveRiskEngine(self.config)

    def evaluate_account_health(
        self,
        equity: float,
        last_equity: float,
        cash: float,
        buying_power: float,
    ) -> AccountState:
        """Evaluate account state and determine intraday drawdown and kill-switch status.

        Args:
            equity: Real-time total portfolio equity.
            last_equity: Previous day closing equity.
            cash: Uninvested cash balance.
            buying_power: Current buying power.

        Returns:
            AccountState: Immutable snapshot of portfolio risk status.
        """
        if last_equity > 0:
            # Drawdown relative to yesterday's closing equity
            intraday_drawdown = max(0.0, (last_equity - equity) / last_equity)
        else:
            intraday_drawdown = 0.0

        kill_switch = intraday_drawdown >= self.config.max_intraday_drawdown_pct

        if kill_switch:
            logger.critical(
                "KILL-SWITCH ACTIVATED! Intraday drawdown: %.2f%% (threshold: %.2f%%). "
                "Current equity: $%.2f, Last equity: $%.2f. New orders are blocked.",
                intraday_drawdown * 100,
                self.config.max_intraday_drawdown_pct * 100,
                equity,
                last_equity,
            )

        return AccountState(
            equity=equity,
            last_equity=last_equity,
            cash=cash,
            buying_power=buying_power,
            intraday_drawdown_pct=intraday_drawdown,
            kill_switch_active=kill_switch,
        )

    def check_portfolio_correlation(
        self,
        candidate_symbol: str,
        candidate_returns: pd.Series,
        open_positions_returns: dict[str, pd.Series],
    ) -> tuple[bool, Optional[str], float]:
        """Check the candidate's log-return correlation against every open position.

        Series must be log returns indexed by timestamp; they are aligned on the index,
        NaNs dropped, and only the last `correlation_lookback_bars` common bars used.
        Pairs with < 3 common bars or zero variance are skipped (correlation undefined).

        Returns:
            (approved, rejection_reason, max_corr). max_corr is 0.0 when nothing is comparable.
        """
        max_corr = 0.0
        for ticker, position_returns in (open_positions_returns or {}).items():
            aligned = pd.concat([candidate_returns, position_returns], axis=1).dropna()
            aligned = aligned.iloc[-self.config.correlation_lookback_bars:]
            if len(aligned) < 3 or (aligned.std(ddof=0) == 0).any():
                logger.warning(
                    "Correlation %s vs %s skipped: %d usable bars or zero volatility.",
                    candidate_symbol, ticker, len(aligned),
                )
                continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if not math.isfinite(corr):
                continue
            max_corr = max(max_corr, corr)
            if corr >= self.config.max_portfolio_correlation:
                return False, f"Exceeded correlation limit with {ticker} (corr={corr:.2f})", max_corr
        return True, None, max_corr

    def calculate_position_size(
        self,
        signal: TradingSignal,
        account: AccountState,
        regime: Optional[MarketRegime] = None,
        candidate_returns: Optional[pd.Series] = None,
        open_positions_returns: Optional[dict[str, pd.Series]] = None,
    ) -> PositionSize:
        """Compute exact position sizing and bracket order price targets.

        If a regime is specified or present on the signal, delegates to the AdaptiveRiskEngine.
        Otherwise applies standard base configuration parameters.

        Args:
            signal: Trading signal generated by the strategy.
            account: Current account health snapshot.
            regime: Optional market regime override.
            candidate_returns: Optional log returns of the signal's symbol (enables correlation gate).
            open_positions_returns: Optional {ticker: log returns} of currently open positions.

        Returns:
            PositionSize: Strongly typed sizing specification with validation flags.
        """
        # Correlation gate runs before regime delegation so it covers both sizing paths.
        if (
            signal.signal_type == SignalType.BUY
            and candidate_returns is not None
            and open_positions_returns
        ):
            approved, reason, _ = self.check_portfolio_correlation(
                signal.symbol, candidate_returns, open_positions_returns
            )
            if not approved:
                return PositionSize(
                    symbol=signal.symbol,
                    shares=0.0,
                    is_fractional=False,
                    entry_price=signal.entry_price,
                    stop_loss_price=0.0,
                    take_profit_price=0.0,
                    risk_amount_usd=0.0,
                    risk_per_share=0.0,
                    reward_per_share=0.0,
                    risk_reward_ratio=0.0,
                    total_exposure_usd=0.0,
                    is_valid=False,
                    rejection_reason=f"Correlation gate blocked: {reason}",
                )

        # Delegate to adaptive regime engine if regime is explicitly specified
        if regime is not None:
            return self.adaptive_engine.calculate_position_size(signal, account, regime)

        # Delegate if signal has a recognized regime attached
        if signal.regime is not None:
            try:
                active_regime = MarketRegime(signal.regime)
                return self.adaptive_engine.calculate_position_size(signal, account, active_regime)
            except (ValueError, TypeError):
                pass

        symbol = signal.symbol
        entry_price = signal.entry_price

        # Check Kill Switch
        if account.kill_switch_active:
            return PositionSize(
                symbol=symbol,
                shares=0.0,
                is_fractional=False,
                entry_price=entry_price,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                risk_amount_usd=0.0,
                risk_per_share=0.0,
                reward_per_share=0.0,
                risk_reward_ratio=0.0,
                total_exposure_usd=0.0,
                is_valid=False,
                rejection_reason=f"Kill-switch active (Intraday drawdown {account.intraday_drawdown_pct*100:.2f}% >= {self.config.max_intraday_drawdown_pct*100:.2f}%)",
            )

        if signal.signal_type != SignalType.BUY:
            return PositionSize(
                symbol=symbol,
                shares=0.0,
                is_fractional=False,
                entry_price=entry_price,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                risk_amount_usd=0.0,
                risk_per_share=0.0,
                reward_per_share=0.0,
                risk_reward_ratio=0.0,
                total_exposure_usd=0.0,
                is_valid=False,
                rejection_reason=f"Signal is {signal.signal_type.value}, not BUY.",
            )

        if entry_price <= 0.0 or signal.atr_14 <= 0.0:
            return PositionSize(
                symbol=symbol,
                shares=0.0,
                is_fractional=False,
                entry_price=entry_price,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                risk_amount_usd=0.0,
                risk_per_share=0.0,
                reward_per_share=0.0,
                risk_reward_ratio=0.0,
                total_exposure_usd=0.0,
                is_valid=False,
                rejection_reason=f"Invalid pricing data: Entry=${entry_price:.2f}, ATR=${signal.atr_14:.2f}.",
            )

        # 1. Calculate Stop Loss distance (1.5 * ATR(14))
        sl_distance = self.config.atr_sl_multiplier * signal.atr_14
        stop_loss_price = round(entry_price - sl_distance, 2)

        if stop_loss_price <= 0.01:
            return PositionSize(
                symbol=symbol,
                shares=0.0,
                is_fractional=False,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=0.0,
                risk_amount_usd=0.0,
                risk_per_share=0.0,
                reward_per_share=0.0,
                risk_reward_ratio=0.0,
                total_exposure_usd=0.0,
                is_valid=False,
                rejection_reason=f"Calculated Stop Loss (${stop_loss_price:.2f}) is below or equal to zero.",
            )

        # 2. Risk per share
        risk_per_share = entry_price - stop_loss_price
        if risk_per_share <= 0:
            return PositionSize(
                symbol=symbol,
                shares=0.0,
                is_fractional=False,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=0.0,
                risk_amount_usd=0.0,
                risk_per_share=0.0,
                reward_per_share=0.0,
                risk_reward_ratio=0.0,
                total_exposure_usd=0.0,
                is_valid=False,
                rejection_reason="Risk per share is non-positive.",
            )

        # 3. Take Profit with 2:1 Ratio
        reward_per_share = self.config.risk_reward_ratio * risk_per_share
        take_profit_price = round(entry_price + reward_per_share, 2)
        actual_rr = reward_per_share / risk_per_share

        # 4. Core Position Sizing Formula: shares = (capital * 0.01) / abs(entry_price - stop_loss_price)
        risk_dollars = account.equity * self.config.max_risk_per_trade_pct
        raw_shares = risk_dollars / risk_per_share

        # 5. Alpaca Bracket Order enforcement:
        # Note: Alpaca API rejects bracket orders with fractional quantities ("fractional orders must be simple orders").
        # When allow_fractional is False, we floor to the nearest whole integer.
        if self.config.allow_fractional:
            shares = round(raw_shares, 4)
            is_fractional = (shares % 1.0) != 0.0
        else:
            shares = float(math.floor(raw_shares))
            is_fractional = False

        if shares < 1.0:
            # Cannot take even 1 whole share without exceeding the 1.0% maximum risk constraint
            one_share_risk = risk_per_share
            return PositionSize(
                symbol=symbol,
                shares=0.0,
                is_fractional=is_fractional,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                risk_amount_usd=risk_dollars,
                risk_per_share=risk_per_share,
                reward_per_share=reward_per_share,
                risk_reward_ratio=actual_rr,
                total_exposure_usd=0.0,
                is_valid=False,
                rejection_reason=(
                    f"Position size {raw_shares:.3f} < 1.0 share. Trading 1 share (${one_share_risk:.2f} risk) "
                    f"would exceed maximum allowed 1.0% capital risk (${risk_dollars:.2f})."
                ),
            )

        # 6. Buying Power and Concentration Constraints
        total_exposure = shares * entry_price

        # Check Buying Power
        if total_exposure > account.buying_power:
            max_shares_bp = math.floor(account.buying_power / entry_price)
            if max_shares_bp < 1.0:
                return PositionSize(
                    symbol=symbol,
                    shares=0.0,
                    is_fractional=is_fractional,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    risk_amount_usd=risk_dollars,
                    risk_per_share=risk_per_share,
                    reward_per_share=reward_per_share,
                    risk_reward_ratio=actual_rr,
                    total_exposure_usd=total_exposure,
                    is_valid=False,
                    rejection_reason=f"Insufficient buying power (${account.buying_power:.2f} available, ${total_exposure:.2f} required).",
                )
            logger.warning(
                "Reducing shares from %d to %d for %s due to available buying power ($%.2f)",
                int(shares),
                int(max_shares_bp),
                symbol,
                account.buying_power,
            )
            shares = float(max_shares_bp)
            total_exposure = shares * entry_price

        # Check Maximum Portfolio Concentration per asset
        max_allowed_exposure = account.equity * self.config.max_position_equity_pct
        if total_exposure > max_allowed_exposure:
            capped_shares = math.floor(max_allowed_exposure / entry_price)
            if capped_shares >= 1.0:
                logger.info(
                    "Capping shares from %d to %d for %s to maintain single-position exposure below %.1f%%",
                    int(shares),
                    int(capped_shares),
                    symbol,
                    self.config.max_position_equity_pct * 100,
                )
                shares = float(capped_shares)
                total_exposure = shares * entry_price

        actual_risk_dollars = shares * risk_per_share

        return PositionSize(
            symbol=symbol,
            shares=shares,
            is_fractional=is_fractional,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_amount_usd=actual_risk_dollars,
            risk_per_share=risk_per_share,
            reward_per_share=reward_per_share,
            risk_reward_ratio=actual_rr,
            total_exposure_usd=total_exposure,
            is_valid=True,
            rejection_reason=None,
        )
