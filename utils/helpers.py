"""Utility functions for CryptoTrader Pro."""

from datetime import datetime

from tabulate import tabulate

from core.risk_manager import TradeRecord


def format_trade_table(trades: list[TradeRecord]) -> str:
    if not trades:
        return "No trades to display."

    headers = ["Pair", "Side", "Entry", "Exit", "P&L", "P&L%", "Reason", "Time"]
    rows = []

    for t in trades:
        rows.append([
            t.pair,
            t.side,
            f"{t.entry_price:,.2f}",
            f"{t.exit_price:,.2f}",
            f"{t.pnl:+,.2f}",
            f"{t.pnl_pct:+.2f}%",
            t.exit_reason,
            t.closed_at.strftime("%m-%d %H:%M"),
        ])

    return tabulate(rows, headers=headers, tablefmt="grid")


def format_stats_table(stats: dict) -> str:
    rows = [[k.replace("_", " ").title(), v] for k, v in stats.items()]
    return tabulate(rows, headers=["Metric", "Value"], tablefmt="grid")


def format_currency(value: float, currency: str = "INR") -> str:
    if currency == "INR":
        return f"₹{value:,.2f}"
    return f"${value:,.2f}"


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
