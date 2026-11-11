"""Unit tests for the SQLite PerformanceAuditor and its FastAPI endpoints."""

import csv
import io
import math
import os
import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.performance_auditor import PerformanceAuditor


class TestPerformanceAuditor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "trading_history.db")
        self.auditor = PerformanceAuditor(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _rows(self, table):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
        finally:
            conn.close()

    def test_wal_mode_enabled(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(conn.execute("PRAGMA journal_mode;").fetchone()[0], "wal")
        finally:
            conn.close()

    def test_daily_equity_dedup_returns_and_drawdown(self):
        d0 = date.today() - timedelta(days=2)
        self.auditor.record_daily_equity(100_000, 50_000, 200_000, as_of=d0)
        self.auditor.record_daily_equity(110_000, 50_000, 200_000, as_of=d0 + timedelta(days=1))
        # Same date twice -> one row, latest snapshot wins
        self.auditor.record_daily_equity(98_000, 50_000, 200_000, as_of=d0 + timedelta(days=2))
        self.auditor.record_daily_equity(99_000, 50_000, 200_000, as_of=d0 + timedelta(days=2))

        rows = self._rows("equity_history")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2]["equity"], 99_000)
        self.assertIsNone(rows[0]["daily_return_pct"])
        self.assertAlmostEqual(rows[1]["daily_return_pct"], 10.0)
        self.assertAlmostEqual(rows[2]["daily_return_pct"], (99_000 / 110_000 - 1) * 100)
        self.assertAlmostEqual(rows[2]["drawdown_pct"], (1 - 99_000 / 110_000) * 100)

    def test_closed_trade_upsert_pnl_and_r_multiple(self):
        tid = self.auditor.record_closed_trade(
            "aapl", "2026-10-01", "2026-10-10", 100.0, 106.0, 10, exit_reason="TP", initial_risk_per_share=3.0
        )
        # Re-sync of the same trade updates instead of duplicating
        self.auditor.record_closed_trade(
            "AAPL", "2026-10-01", "2026-10-11", 100.0, 109.0, 10, exit_reason="TRAIL", initial_risk_per_share=3.0
        )
        self.auditor.record_closed_trade("TSLA", "2026-10-02", "2026-10-05", 200.0, 190.0, 5, side="short")

        rows = {r["id"]: r for r in self._rows("trade_history")}
        self.assertEqual(len(rows), 2)
        aapl = rows[tid]
        self.assertEqual((aapl["exit_price"], aapl["exit_reason"]), (109.0, "TRAIL"))
        self.assertAlmostEqual(aapl["pnl_dollars"], 90.0)
        self.assertAlmostEqual(aapl["pnl_pct"], 9.0)
        self.assertAlmostEqual(aapl["r_multiple"], 3.0)
        tsla = rows["TSLA:2026-10-02:short"]
        self.assertAlmostEqual(tsla["pnl_dollars"], 50.0)  # short profits when price falls
        self.assertIsNone(tsla["r_multiple"])

        with self.assertRaises(ValueError):
            self.auditor.record_closed_trade("X", "2026-10-01", "2026-10-02", 10.0, 11.0, 1, side="sideways")

    def test_summary_metrics(self):
        start = date.today() - timedelta(days=10)
        equities = [100_000, 101_000, 99_000, 102_000, 103_000]
        for i, eq in enumerate(equities):
            self.auditor.record_daily_equity(eq, 0, 0, as_of=start + timedelta(days=i))
        self.auditor.record_daily_equity(90_000, 0, 0, as_of=start - timedelta(days=200))  # outside 90d curve
        for entry, exit_ in [(100, 110), (100, 104), (100, 95)]:
            self.auditor.record_closed_trade("SPY", "2026-09-01", "2026-10-01", entry, exit_, 10, trade_id=f"t{exit_}")

        s = self.auditor.get_performance_summary()

        full = [90_000] + equities
        rets = [b / a - 1 for a, b in zip(full, full[1:])]
        mean = sum(rets) / len(rets)
        std = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1))
        downside = math.sqrt(sum(min(r, 0) ** 2 for r in rets) / len(rets))
        self.assertAlmostEqual(s["sharpe_ratio"], round(mean / std * math.sqrt(252), 3))
        self.assertAlmostEqual(s["sortino_ratio"], round(mean / downside * math.sqrt(252), 3))
        self.assertAlmostEqual(s["max_drawdown_pct"], round((1 - 99_000 / 101_000) * 100, 3))
        self.assertAlmostEqual(s["profit_factor"], round(140 / 50, 3))
        self.assertAlmostEqual(s["win_rate_pct"], round(2 / 3 * 100, 2))
        self.assertEqual(s["total_pnl"], 90.0)
        self.assertEqual(s["total_trades"], 3)
        self.assertEqual([p["equity"] for p in s["equity_curve"]], equities)

    def test_summary_empty_db_is_json_safe(self):
        s = self.auditor.get_performance_summary()
        self.assertEqual(
            (s["sharpe_ratio"], s["sortino_ratio"], s["profit_factor"], s["win_rate_pct"]), (None, None, None, None)
        )
        self.assertEqual((s["total_pnl"], s["max_drawdown_pct"], s["equity_curve"]), (0, 0.0, []))

    def test_export_csv(self):
        self.auditor.record_closed_trade("MSFT", "2026-10-01", "2026-10-03", 400.0, 412.0, 2, exit_reason="TP")
        rows = list(csv.reader(io.StringIO(self.auditor.export_trades_csv(raw=True))))
        self.assertEqual(rows[0][:3], ["id", "symbol", "entry_date"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][1], "MSFT")


class TestPerformanceEndpoints(unittest.TestCase):
    """Endpoints exercised without the app lifespan (no Alpaca credentials needed)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.auditor = PerformanceAuditor(os.path.join(self.tmp.name, "t.db"))
        self.auditor.record_closed_trade("NVDA", "2026-10-01", "2026-10-02", 100.0, 105.0, 1)
        import web.app as web_app
        self.synced = []
        fake_bot = SimpleNamespace(auditor=self.auditor, sync_trade_history=lambda *a: self.synced.append(a))
        self.patcher = patch.object(web_app, "bot_instance", fake_bot)
        self.patcher.start()
        self.client = TestClient(web_app.app)  # not entered -> lifespan does not run

    def tearDown(self):
        self.patcher.stop()
        self.tmp.cleanup()

    def test_performance_endpoint(self):
        resp = self.client.get("/api/performance")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for key in ("sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "profit_factor",
                    "win_rate_pct", "total_pnl", "equity_curve"):
            self.assertIn(key, data)
        self.assertEqual(data["total_pnl"], 5.0)
        self.assertEqual(self.synced, [(3600,)])  # stale-guarded re-sync requested before computing

    def test_csv_export_endpoint(self):
        resp = self.client.get("/api/export/trades.csv")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.headers["content-type"].startswith("text/csv"))
        self.assertIn("attachment", resp.headers["content-disposition"])
        self.assertIn("NVDA", resp.text)


# ---------------------------------------------------------------------------
# Alpaca closed-trade reconciliation (mock orders shaped like alpaca-py Order)
# ---------------------------------------------------------------------------
T0 = datetime(2026, 10, 1, 14, 30, tzinfo=timezone.utc)


def _order(oid, symbol, side, otype, status, qty=None, price=None, filled_at=None,
           stop=None, limit=None, legs=None, replaces=None, replaced_by=None, order_class="simple"):
    return SimpleNamespace(
        id=oid, symbol=symbol, side=f"OrderSide.{side.upper()}", order_type=otype, type=otype,
        status=status, filled_qty=None if qty is None else str(qty),
        filled_avg_price=None if price is None else str(price), filled_at=filled_at,
        stop_price=stop, limit_price=limit, legs=legs, replaces=replaces, replaced_by=replaced_by,
        order_class=order_class,
    )


def _bracket(oid, symbol, entry, qty, sl, tp, exit_leg=None, exit_price=None, exit_qty=None, days=5):
    """Filled bracket entry; exit_leg in {'tp', 'sl', None}. The OCO sibling comes back canceled."""
    def leg(kind):
        filled = exit_leg == kind
        return _order(
            f"{oid}-{kind}", symbol, "sell", "limit" if kind == "tp" else "stop",
            "filled" if filled else "canceled",
            qty=(exit_qty or qty) if filled else 0, price=exit_price if filled else None,
            filled_at=T0 + timedelta(days=days) if filled else None,
            stop=sl if kind == "sl" else None, limit=tp if kind == "tp" else None,
        )
    return _order(oid, symbol, "buy", "market", "filled", qty=qty, price=entry, filled_at=T0,
                  legs=[leg("tp"), leg("sl")], order_class="bracket")


class FakeTradingClient:
    def __init__(self, orders):
        self.orders = orders
        self.requests = []

    def get_orders(self, filter=None):
        self.requests.append(filter)
        return self.orders


class TestAlpacaTradeSync(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "trading_history.db")
        self.auditor = PerformanceAuditor(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _trades(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            return {r["id"]: r for r in conn.execute("SELECT * FROM trade_history")}
        finally:
            conn.close()

    def test_take_profit_winner(self):
        client = FakeTradingClient([_bracket("e1", "AAPL", 100.0, 10, sl=97.0, tp=106.0, exit_leg="tp", exit_price=106.0)])
        self.assertEqual(self.auditor.sync_closed_trades_from_alpaca(client), 1)

        req = client.requests[0]
        self.assertEqual((req.status.value, req.limit, req.nested), ("closed", 500, True))
        t = self._trades()["e1"]
        self.assertEqual((t["symbol"], t["exit_reason"], t["side"]), ("AAPL", "TAKE_PROFIT", "long"))
        self.assertAlmostEqual(t["pnl_dollars"], 60.0)
        self.assertAlmostEqual(t["pnl_pct"], 6.0)
        self.assertAlmostEqual(t["r_multiple"], 2.0)  # +6 / initial risk 3
        self.assertTrue(t["entry_date"].startswith("2026-10-01"))
        self.assertTrue(t["exit_date"].startswith("2026-10-06"))

    def test_stop_loss_loser(self):
        client = FakeTradingClient([_bracket("e2", "TSLA", 200.0, 5, sl=194.0, tp=212.0, exit_leg="sl", exit_price=193.5)])
        self.auditor.sync_closed_trades_from_alpaca(client)
        t = self._trades()["e2"]
        self.assertEqual(t["exit_reason"], "STOP_LOSS")
        self.assertAlmostEqual(t["pnl_dollars"], -32.5)  # slippage below the stop is captured
        self.assertAlmostEqual(t["r_multiple"], -6.5 / 6.0)

    def test_idempotent_double_sync(self):
        orders = [
            _bracket("e1", "AAPL", 100.0, 10, sl=97.0, tp=106.0, exit_leg="tp", exit_price=106.0),
            _bracket("e2", "TSLA", 200.0, 5, sl=194.0, tp=212.0, exit_leg="sl", exit_price=193.5),
        ]
        client = FakeTradingClient(orders)
        self.auditor.sync_closed_trades_from_alpaca(client)
        self.auditor.sync_closed_trades_from_alpaca(client)
        self.assertEqual(len(self._trades()), 2)
        self.assertEqual(self.auditor.get_performance_summary()["total_trades"], 2)

    def test_trailing_stop_uses_original_stop_for_r(self):
        # Stop leg raised 97 -> 103 by the trailing logic (replace_order); the replacement filled at 103.
        entry = _bracket("e3", "MSFT", 100.0, 10, sl=97.0, tp=110.0)
        original_sl = entry.legs[1]
        original_sl.status, original_sl.replaced_by = "replaced", "e3-sl2"
        replacement = _order("e3-sl2", "MSFT", "sell", "stop", "filled", qty=10, price=103.0,
                             filled_at=T0 + timedelta(days=8), stop=103.0, replaces="e3-sl")
        self.auditor.sync_closed_trades_from_alpaca(FakeTradingClient([entry, replacement]))
        trades = self._trades()
        self.assertEqual(list(trades), ["e3"])  # the replacement leg is not a separate manual trade
        t = trades["e3"]
        self.assertEqual(t["exit_reason"], "TRAILING_STOP")
        self.assertAlmostEqual(t["r_multiple"], 1.0)  # +3 / original risk 3 (not the trailed stop)

    def test_manual_close_and_still_open(self):
        # Manual close: Alpaca cancels both legs and a standalone market SELL fills.
        manual = _bracket("e4", "NVDA", 120.0, 8, sl=115.0, tp=130.0)
        sell = _order("s4", "NVDA", "sell", "market", "filled", qty=8, price=125.0, filled_at=T0 + timedelta(days=2))
        # Position still open: entry filled, legs still working (not in the CLOSED listing).
        still_open = _order("e5", "AMD", "buy", "market", "filled", qty=3, price=150.0, filled_at=T0, legs=[])
        self.assertEqual(self.auditor.sync_closed_trades_from_alpaca(FakeTradingClient([manual, sell, still_open])), 1)
        t = self._trades()["e4"]
        self.assertEqual(t["exit_reason"], "MANUAL")
        self.assertAlmostEqual(t["pnl_dollars"], 40.0)
        self.assertAlmostEqual(t["r_multiple"], 1.0)

    def test_partial_close_and_incomplete_data(self):
        partial = _bracket("e6", "META", 500.0, 10, sl=490.0, tp=520.0, exit_leg="tp", exit_price=520.0, exit_qty=4)
        broken = _order("e7", "GOOG", "buy", "market", "filled", qty=5, price=None, filled_at=T0)  # no fill price
        with self.assertLogs("src.performance_auditor", level="WARNING") as logs:
            self.auditor.sync_closed_trades_from_alpaca(FakeTradingClient([partial, broken]))
        self.assertTrue(any("e7" in m for m in logs.output))
        t = self._trades()["e6"]
        self.assertEqual(t["shares"], 4.0)
        self.assertAlmostEqual(t["pnl_dollars"], 80.0)
        self.assertNotIn("e7", self._trades())

    def test_sync_if_stale(self):
        client = FakeTradingClient([])
        self.assertEqual(self.auditor.sync_if_stale(client), 0)
        self.assertIsNone(self.auditor.sync_if_stale(client, max_age_seconds=3600))  # fresh -> skipped
        self.assertEqual(len(client.requests), 1)


if __name__ == "__main__":
    unittest.main()
