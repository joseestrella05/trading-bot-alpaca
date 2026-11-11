"""Performance auditing and persistence on SQLite (WAL mode).

Stores daily equity snapshots and closed trades in `data/trading_history.db` and derives
institutional metrics (Sharpe, Sortino, Max Drawdown, Profit Factor, Win Rate).
A short-lived connection is opened per operation so the auditor is safe to call from the
scheduler thread and FastAPI worker threads concurrently; WAL lets readers run during writes.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import date, datetime, timedelta, timezone
import io
import logging
import math
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterator, List, Optional, Union

from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "trading_history.db"
TRADING_DAYS_PER_YEAR = 252

STOP_TYPES = ("stop", "stop_limit", "trailing_stop")


def _v(value: Any) -> str:
    """Normalize Alpaca enums / strings ('OrderSide.BUY', OrderType.LIMIT, 'limit') to lowercase values."""
    raw = getattr(value, "value", value)
    return str(raw).split(".")[-1].lower() if raw is not None else ""


def _f(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _otype(order: Any) -> str:
    return _v(getattr(order, "order_type", None) or getattr(order, "type", None))


TRADE_COLUMNS = [
    "id", "symbol", "entry_date", "exit_date", "entry_price", "exit_price", "shares", "side",
    "pnl_dollars", "pnl_pct", "exit_reason", "r_multiple", "timestamp",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_history (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    exit_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    shares REAL NOT NULL,
    side TEXT NOT NULL,
    pnl_dollars REAL NOT NULL,
    pnl_pct REAL NOT NULL,
    exit_reason TEXT,
    r_multiple REAL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trade_exit_date ON trade_history (exit_date);
CREATE TABLE IF NOT EXISTS equity_history (
    date TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    buying_power REAL NOT NULL,
    daily_return_pct REAL,
    drawdown_pct REAL NOT NULL,
    timestamp TEXT NOT NULL
);
"""


def _iso_date(value: Union[str, date, datetime, None]) -> str:
    if value is None:
        return datetime.now(timezone.utc).date().isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


class PerformanceAuditor:
    """Persists equity snapshots and closed trades; computes performance statistics."""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_sync_at: Optional[datetime] = None
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open, commit on success / rollback on error, and always close."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")  # per-connection setting; safe with WAL
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ writes
    def record_daily_equity(
        self,
        equity: float,
        cash: float,
        buying_power: float,
        as_of: Union[str, date, None] = None,
    ) -> None:
        """Upsert today's equity snapshot (one row per date; the latest call of the day wins)."""
        day = _iso_date(as_of or datetime.now(timezone.utc).date())
        with self._connect() as conn:
            prev = conn.execute(
                "SELECT equity FROM equity_history WHERE date < ? ORDER BY date DESC LIMIT 1", (day,)
            ).fetchone()
            peak_row = conn.execute(
                "SELECT MAX(equity) AS peak FROM equity_history WHERE date < ?", (day,)
            ).fetchone()
            daily_return = (equity / prev["equity"] - 1.0) * 100.0 if prev and prev["equity"] > 0 else None
            peak = max(equity, peak_row["peak"] or equity)
            drawdown = (1.0 - equity / peak) * 100.0 if peak > 0 else 0.0
            conn.execute(
                """
                INSERT INTO equity_history (date, equity, cash, buying_power, daily_return_pct, drawdown_pct, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    equity=excluded.equity, cash=excluded.cash, buying_power=excluded.buying_power,
                    daily_return_pct=excluded.daily_return_pct, drawdown_pct=excluded.drawdown_pct,
                    timestamp=excluded.timestamp
                """,
                (day, equity, cash, buying_power, daily_return, drawdown, datetime.now(timezone.utc).isoformat()),
            )

    def record_closed_trade(
        self,
        symbol: str,
        entry_date: Union[str, date, datetime],
        exit_date: Union[str, date, datetime],
        entry_price: float,
        exit_price: float,
        shares: float,
        side: str = "long",
        exit_reason: Optional[str] = None,
        initial_risk_per_share: Optional[float] = None,
        trade_id: Optional[str] = None,
    ) -> str:
        """Insert or update a closed trade. Returns its id (defaults to symbol:entry_date:side).

        Args:
            initial_risk_per_share: |entry - initial stop|; when given, r_multiple is stored.
            trade_id: Stable id (e.g. Alpaca parent order id) so re-syncing updates instead of duplicating.
        """
        side = side.lower()
        if side not in ("long", "short"):
            raise ValueError(f"side must be 'long' or 'short', got {side!r}")
        if entry_price <= 0 or shares <= 0:
            raise ValueError("entry_price and shares must be positive")

        direction = 1.0 if side == "long" else -1.0
        pnl_per_share = (exit_price - entry_price) * direction
        pnl_dollars = pnl_per_share * shares
        pnl_pct = pnl_per_share / entry_price * 100.0
        r_multiple = (
            pnl_per_share / initial_risk_per_share
            if initial_risk_per_share and initial_risk_per_share > 0 else None
        )
        entry_s, exit_s = _iso_date(entry_date), _iso_date(exit_date)
        trade_id = trade_id or f"{symbol.upper()}:{entry_s}:{side}"

        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO trade_history ({", ".join(TRADE_COLUMNS)})
                VALUES ({", ".join("?" * len(TRADE_COLUMNS))})
                ON CONFLICT(id) DO UPDATE SET
                    exit_date=excluded.exit_date, exit_price=excluded.exit_price, shares=excluded.shares,
                    pnl_dollars=excluded.pnl_dollars, pnl_pct=excluded.pnl_pct,
                    exit_reason=excluded.exit_reason, r_multiple=excluded.r_multiple,
                    timestamp=excluded.timestamp
                """,
                (
                    trade_id, symbol.upper(), entry_s, exit_s, entry_price, exit_price, shares, side,
                    pnl_dollars, pnl_pct, exit_reason, r_multiple, datetime.now(timezone.utc).isoformat(),
                ),
            )
        return trade_id

    # --------------------------------------------------------- alpaca sync
    def sync_closed_trades_from_alpaca(self, trading_client: Any, days_back: int = 120) -> int:
        """Reconcile closed long trades from Alpaca orders into trade_history. Idempotent (id = entry order id).

        Pairs each filled BUY entry with its exit:
        - Bracket legs (nested): the filled SELL leg(s). The OCO sibling arrives `canceled` and is ignored.
          Legs replaced by the trailing-stop logic are followed through `replaced_by` to the final order,
          and back through `replaces` to recover the ORIGINAL stop (initial risk for the R-multiple).
        - No filled leg (position closed by hand: Alpaca cancels the legs): the first later filled SELL of the
          same symbol that is not itself a bracket leg -> exit_reason MANUAL.
        Entries with no exit yet (position still open) are skipped. Partial exits are recorded with the
        exited quantity and updated on later syncs. Orders with incomplete data are logged and skipped.

        Returns:
            Number of trades recorded/updated in this run.
        """
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=500,
            nested=True,
            after=datetime.now(timezone.utc) - timedelta(days=days_back),
        )
        orders = list(trading_client.get_orders(filter=request) or [])
        trades = self._reconcile_orders(orders)
        for trade in trades:
            self.record_closed_trade(**trade)
        self.last_sync_at = datetime.now(timezone.utc)
        logger.info("Alpaca trade sync: %d closed orders scanned, %d trades recorded.", len(orders), len(trades))
        return len(trades)

    def sync_if_stale(self, trading_client: Any, max_age_seconds: float = 3600.0, days_back: int = 120) -> Optional[int]:
        """Run the Alpaca sync only if the last one is older than max_age_seconds (or never ran)."""
        if self.last_sync_at and (datetime.now(timezone.utc) - self.last_sync_at).total_seconds() < max_age_seconds:
            return None
        return self.sync_closed_trades_from_alpaca(trading_client, days_back=days_back)

    def sync_equity_history_from_alpaca(self, trading_client: Any, period: str = "3M") -> int:
        """Sync historical daily equity curve from Alpaca get_portfolio_history."""
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest
            history = trading_client.get_portfolio_history(GetPortfolioHistoryRequest(period=period, timeframe="1D"))
            if not history or not getattr(history, "equity", None):
                return 0
            count = 0
            for ts, eq in zip(history.timestamp, history.equity):
                if eq and eq > 0:
                    d = datetime.fromtimestamp(ts, timezone.utc).date()
                    self.record_daily_equity(equity=float(eq), cash=float(eq), buying_power=float(eq) * 4, as_of=d)
                    count += 1
            return count
        except Exception as exc:
            logger.warning("Failed to sync equity history from Alpaca: %s", exc)
            return 0

    @staticmethod
    def _reconcile_orders(orders: List[Any]) -> List[Dict[str, Any]]:
        """Pure pairing logic: Alpaca orders -> kwargs for record_closed_trade()."""
        by_id: Dict[str, Any] = {}

        def index(order: Any) -> None:
            by_id[str(order.id)] = order
            for leg in getattr(order, "legs", None) or []:
                index(leg)

        for order in orders:
            index(order)

        def chain(order: Any, attr: str) -> List[Any]:
            """order followed through replaced_by (forward) or replaces (backward), cycle-safe."""
            out = [order]
            while True:
                nxt = by_id.get(str(getattr(out[-1], attr, None) or ""))
                if nxt is None or nxt in out:
                    return out
                out.append(nxt)

        def filled(order: Any) -> bool:
            return (_f(order.filled_qty) or 0.0) > 0 and _f(order.filled_avg_price) is not None

        trades: List[Dict[str, Any]] = []
        consumed: set = set()  # ids belonging to bracket legs: never treat them as manual exits
        open_entries: List[Any] = []
        entries = [o for o in orders if _v(o.side) == "buy" and (_f(o.filled_qty) or 0.0) > 0]

        for entry in entries:
            try:
                entry_price = _f(entry.filled_avg_price)
                if not entry_price or entry.filled_at is None:
                    logger.warning("Trade sync: entry %s (%s) missing fill price/time, skipped.", entry.id, entry.symbol)
                    continue
                legs = []
                for leg in getattr(entry, "legs", None) or []:
                    forward = chain(leg, "replaced_by")
                    backward = chain(leg, "replaces")
                    consumed.update(str(o.id) for o in forward + backward)
                    legs.append((forward[-1], backward[-1]))  # (final order, original order)

                original_stops = [orig for final, orig in legs if _otype(orig) in STOP_TYPES]
                initial_stop = _f(original_stops[0].stop_price) if original_stops else None
                exits = [final for final, _ in legs if _v(final.side) == "sell" and filled(final)]
                if not exits:
                    open_entries.append((entry, initial_stop))
                    continue
                trades.append(_build_trade(entry, exits, initial_stop, manual=False))
            except Exception as exc:
                logger.warning("Trade sync: could not reconcile entry %s: %s", getattr(entry, "id", "?"), exc)

        # Manual closes: FIFO match remaining entries with later standalone SELL fills of the same symbol.
        sells = sorted(
            (o for o in orders if _v(o.side) == "sell" and filled(o) and o.filled_at and str(o.id) not in consumed),
            key=lambda o: o.filled_at,
        )
        for entry, initial_stop in sorted(open_entries, key=lambda e: e[0].filled_at):
            match = next((s for s in sells if s.symbol == entry.symbol and s.filled_at >= entry.filled_at), None)
            if match is None:
                continue  # still open
            sells.remove(match)
            try:
                trades.append(_build_trade(entry, [match], initial_stop, manual=True))
            except Exception as exc:
                logger.warning("Trade sync: could not reconcile manual exit %s: %s", match.id, exc)
        return trades

    # ------------------------------------------------------------------- reads
    def get_performance_summary(self, curve_days: int = 90) -> Dict[str, Any]:
        """Risk-adjusted statistics. Ratios are None when there is not enough data to define them."""
        with self._connect() as conn:
            equity_rows = conn.execute("SELECT date, equity FROM equity_history ORDER BY date").fetchall()
            trades = conn.execute("SELECT pnl_dollars FROM trade_history").fetchall()

        equities = [r["equity"] for r in equity_rows]
        returns = [b / a - 1.0 for a, b in zip(equities, equities[1:]) if a > 0]

        sharpe = sortino = None
        if len(returns) >= 2:
            mean = sum(returns) / len(returns)
            std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))
            if std > 0:
                sharpe = mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)
            downside = math.sqrt(sum(min(r, 0.0) ** 2 for r in returns) / len(returns))
            if downside > 0:
                sortino = mean / downside * math.sqrt(TRADING_DAYS_PER_YEAR)

        max_dd = peak = 0.0
        curve: List[Dict[str, Any]] = []
        for row in equity_rows:
            peak = max(peak, row["equity"])
            dd = (1.0 - row["equity"] / peak) * 100.0 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
            curve.append({"date": row["date"], "equity": row["equity"], "drawdown_pct": round(dd, 4)})

        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=curve_days)).isoformat()
        pnls = [t["pnl_dollars"] for t in trades]
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = -sum(p for p in pnls if p < 0)
        wins = sum(1 for p in pnls if p > 0)

        return {
            "sharpe_ratio": round(sharpe, 3) if sharpe is not None else None,
            "sortino_ratio": round(sortino, 3) if sortino is not None else None,
            "max_drawdown_pct": round(max_dd, 3),
            "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else None,
            "win_rate_pct": round(wins / len(pnls) * 100.0, 2) if pnls else None,
            "total_pnl": round(sum(pnls), 2),
            "total_trades": len(pnls),
            "equity_curve": [p for p in curve if p["date"] >= cutoff],
            "recent_trades": self.get_recent_trades(50),
        }

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent closed trades ordered by exit_date descending."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(TRADE_COLUMNS)} FROM trade_history ORDER BY exit_date DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def export_trades_csv(self) -> str:
        """All closed trades as CSV text (header row included), oldest exit first."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(TRADE_COLUMNS)} FROM trade_history ORDER BY exit_date, id"
            ).fetchall()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(TRADE_COLUMNS)
        writer.writerows(tuple(r) for r in rows)
        return buf.getvalue()

    def export_trades_xlsx(self) -> bytes:
        """Audited closed trades as a professional institutional Excel workbook (.xlsx)."""
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        summary = self.get_performance_summary()
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(TRADE_COLUMNS)} FROM trade_history ORDER BY exit_date ASC, id ASC"
            ).fetchall()

        wb = Workbook()
        ws = wb.active
        ws.title = "Audited Trades"
        ws.views.sheetView[0].showGridLines = True

        thin_side = Side(border_style="thin", color="CBD5E1")
        cell_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
        soft_grid_side = Side(border_style="thin", color="E2E8F0")
        row_border = Border(left=soft_grid_side, right=soft_grid_side, top=soft_grid_side, bottom=soft_grid_side)

        # ----------------------------------------------------------- Row 1: Title
        ws.merge_cells("A1:M1")
        title_cell = ws["A1"]
        title_cell.value = "QUANTBOT SYSTEMATIC ENGINE — AUDITED TRADE LEDGER"
        title_cell.font = Font(name="Calibri", size=13, bold=True, color="FFFFFF")
        title_cell.fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
        title_cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 36

        # Fill background of merged title cells
        for col_idx in range(1, 14):
            ws.cell(row=1, column=col_idx).fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")

        # ---------------------------------------------------- Rows 2-3: KPI Cards
        ws.row_dimensions[2].height = 18
        ws.row_dimensions[3].height = 26

        def _style_card(c1, c2, r_lbl, r_val, label, value, val_color="0F172A"):
            ws.merge_cells(start_row=r_lbl, start_column=c1, end_row=r_lbl, end_column=c2)
            lbl_cell = ws.cell(row=r_lbl, column=c1, value=label)
            lbl_cell.font = Font(name="Calibri", size=9, bold=True, color="64748B")
            lbl_cell.alignment = Alignment(horizontal="center", vertical="center")
            lbl_cell.fill = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")

            ws.merge_cells(start_row=r_val, start_column=c1, end_row=r_val, end_column=c2)
            v_cell = ws.cell(row=r_val, column=c1, value=value)
            v_cell.font = Font(name="Calibri", size=12, bold=True, color=val_color)
            v_cell.alignment = Alignment(horizontal="center", vertical="center")
            v_cell.fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")

            for col in range(c1, c2 + 1):
                ws.cell(row=r_lbl, column=col).border = cell_border
                ws.cell(row=r_val, column=col).border = cell_border

        pnl_val = summary.get("total_pnl", 0.0)
        pnl_str = f"${pnl_val:+,.2f}" if pnl_val != 0 else "$0.00"
        pnl_col = "059669" if pnl_val >= 0 else "DC2626"
        wr_str = f"{summary['win_rate_pct']:.1f}%" if summary.get("win_rate_pct") is not None else "--"
        pf_str = f"{summary['profit_factor']:.2f}" if summary.get("profit_factor") is not None else "--"
        dd_str = f"{summary['max_drawdown_pct']:.2f}%" if summary.get("max_drawdown_pct") is not None else "0.00%"
        trades_str = str(summary.get("total_trades", len(rows)))

        _style_card(2, 3, 2, 3, "TOTAL P&L", pnl_str, pnl_col)
        _style_card(4, 5, 2, 3, "WIN RATE", wr_str, "0F172A")
        _style_card(6, 7, 2, 3, "PROFIT FACTOR", pf_str, "0F172A")
        _style_card(8, 9, 2, 3, "MAX DRAWDOWN", dd_str, "DC2626" if summary.get("max_drawdown_pct", 0) > 0 else "0F172A")
        _style_card(10, 11, 2, 3, "TOTAL TRADES", trades_str, "0F172A")

        # Row 4: Spacer
        ws.row_dimensions[4].height = 14

        # ------------------------------------------------- Row 5: Table Header
        headers = [
            ("Trade #", "center"),
            ("Ticker", "center"),
            ("Tipo", "center"),
            ("Fecha Entrada", "center"),
            ("Fecha Salida", "center"),
            ("Días", "center"),
            ("Precio Entrada", "right"),
            ("Precio Salida", "right"),
            ("Acciones", "right"),
            ("P&L ($)", "right"),
            ("P&L (%)", "right"),
            ("Motivo", "center"),
            ("R-Múltiplo", "center"),
        ]

        ws.row_dimensions[5].height = 26
        header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
        header_font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")

        for col_idx, (col_title, align_h) in enumerate(headers, start=1):
            cell = ws.cell(row=5, column=col_idx, value=col_title)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal=align_h, vertical="center")
            cell.border = cell_border

        # ------------------------------------------------ Row 6+: Trade Rows
        def _clean_date_str(s: Any) -> str:
            if not s:
                return "--"
            try:
                dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
                return dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                return str(s)[:16]

        def _days_held(e1_s: Any, e2_s: Any) -> float:
            try:
                d1 = datetime.fromisoformat(str(e1_s).replace("Z", "+00:00"))
                d2 = datetime.fromisoformat(str(e2_s).replace("Z", "+00:00"))
                return round(max(0.0, (d2 - d1).total_seconds() / 86400.0), 1)
            except Exception:
                return 0.0

        win_fill = PatternFill(start_color="ECFDF5", end_color="ECFDF5", fill_type="solid")  # soft emerald
        loss_fill = PatternFill(start_color="FEF2F2", end_color="FEF2F2", fill_type="solid")  # soft red
        text_font = Font(name="Calibri", size=10, color="0F172A")
        bold_win_font = Font(name="Calibri", size=10, bold=True, color="047857")
        bold_loss_font = Font(name="Calibri", size=10, bold=True, color="B91C1C")

        for idx, r in enumerate(rows, start=1):
            row_num = idx + 5
            ws.row_dimensions[row_num].height = 22
            pnl_dollars = float(r["pnl_dollars"])
            is_win = pnl_dollars >= 0
            row_fill = win_fill if is_win else loss_fill
            pnl_font = bold_win_font if is_win else bold_loss_font

            r_mult = r["r_multiple"]
            r_mult_str = f"{float(r_mult):+.2f} R" if r_mult is not None else "--"

            values = [
                (idx, "center", None, text_font),
                (r["symbol"], "center", None, Font(name="Calibri", size=10, bold=True, color="0F172A")),
                (r["side"].upper(), "center", None, text_font),
                (_clean_date_str(r["entry_date"]), "center", None, text_font),
                (_clean_date_str(r["exit_date"]), "center", None, text_font),
                (_days_held(r["entry_date"], r["exit_date"]), "center", "0.0", text_font),
                (float(r["entry_price"]), "right", "$#,##0.00", text_font),
                (float(r["exit_price"]), "right", "$#,##0.00", text_font),
                (float(r["shares"]), "right", "#,##0", text_font),
                (pnl_dollars, "right", "$#,##0.00;($#,##0.00);\"$0.00\"", pnl_font),
                (float(r["pnl_pct"]) / 100.0, "right", "+0.00%;-0.00%;0.00%", pnl_font),
                (r["exit_reason"] or "MANUAL", "center", None, text_font),
                (r_mult_str, "center", None, pnl_font if r_mult is not None else text_font),
            ]

            for col_idx, (val, align_h, num_fmt, font_style) in enumerate(values, start=1):
                c = ws.cell(row=row_num, column=col_idx, value=val)
                c.fill = row_fill
                c.font = font_style
                c.alignment = Alignment(horizontal=align_h, vertical="center")
                c.border = row_border
                if num_fmt:
                    c.number_format = num_fmt

        # Auto-fit column widths with safety margin
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                # Skip title and card rows for column width auto-calculation
                if cell.row < 5:
                    continue
                v = cell.value
                if v is not None:
                    max_len = max(max_len, len(str(v)))
            ws.column_dimensions[col_letter].width = max(max_len + 4, 11)

        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()


def _build_trade(entry: Any, exits: List[Any], initial_stop: Optional[float], manual: bool) -> Dict[str, Any]:
    """Aggregate (possibly partial / multiple) exit fills of one entry into record_closed_trade kwargs."""
    entry_price = float(entry.filled_avg_price)
    entry_qty = float(entry.filled_qty)
    exit_qty = sum(float(o.filled_qty) for o in exits)
    exit_price = sum(float(o.filled_qty) * float(o.filled_avg_price) for o in exits) / exit_qty
    shares = min(entry_qty, exit_qty)
    if exit_qty < entry_qty:
        logger.info("Trade sync: %s partially closed (%.0f of %.0f shares).", entry.symbol, exit_qty, entry_qty)

    main_exit = max(exits, key=lambda o: float(o.filled_qty))
    otype = _otype(main_exit)
    if manual:
        reason = "MANUAL"
    elif otype == "limit":
        reason = "TAKE_PROFIT"
    elif otype in STOP_TYPES:
        trailed = otype == "trailing_stop" or getattr(main_exit, "replaces", None) or exit_price >= entry_price
        reason = "TRAILING_STOP" if trailed else "STOP_LOSS"
    else:
        reason = "MANUAL"

    risk = entry_price - initial_stop if initial_stop is not None and initial_stop < entry_price else None
    return {
        "symbol": entry.symbol,
        "entry_date": entry.filled_at,
        "exit_date": max(o.filled_at for o in exits if o.filled_at is not None),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "shares": shares,
        "side": "long",
        "exit_reason": reason,
        "initial_risk_per_share": risk,
        "trade_id": str(entry.id),
    }
