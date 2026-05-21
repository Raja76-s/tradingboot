"""
Risk Management Module.

Handles:
- Position sizing (max % of capital per trade)
- Stop-loss calculation
- Max daily loss tracking
- Trailing stop-loss
- Max open positions
"""

from dataclasses import dataclass, field
from datetime import date, datetime

from config.settings import RiskConfig


@dataclass
class Position:
    pair: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    trailing_stop: float | None = None
    opened_at: datetime = field(default_factory=datetime.now)
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    pnl: float = 0.0

    def update_trailing_stop(
        self, current_price: float, trailing_pct: float
    ) -> None:
        if self.side == "BUY":
            if current_price > self.highest_price:
                self.highest_price = current_price
            new_stop = self.highest_price * (1 - trailing_pct / 100)
            if self.trailing_stop is None or new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
        else:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            new_stop = self.lowest_price * (1 + trailing_pct / 100)
            if self.trailing_stop is None or new_stop < self.trailing_stop:
                self.trailing_stop = new_stop

    def check_exit(self, current_price: float) -> str | None:
        if self.side == "BUY":
            if current_price <= self.stop_loss:
                return "STOP_LOSS"
            if self.trailing_stop and current_price <= self.trailing_stop:
                return "TRAILING_STOP"
            if current_price >= self.take_profit:
                return "TAKE_PROFIT"
        else:
            if current_price >= self.stop_loss:
                return "STOP_LOSS"
            if self.trailing_stop and current_price >= self.trailing_stop:
                return "TRAILING_STOP"
            if current_price <= self.take_profit:
                return "TAKE_PROFIT"
        return None

    def calculate_pnl(self, current_price: float) -> float:
        if self.side == "BUY":
            self.pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - current_price) * self.quantity
        return self.pnl


@dataclass
class TradeRecord:
    pair: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    opened_at: datetime
    closed_at: datetime


class RiskManager:
    """Manages trading risk."""

    def __init__(self, config: RiskConfig, initial_balance: float) -> None:
        self.config = config
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.open_positions: list[Position] = []
        self.trade_history: list[TradeRecord] = []
        self.daily_pnl: dict[str, float] = {}

    def can_open_position(self) -> tuple[bool, str]:
        if len(self.open_positions) >= self.config.max_open_positions:
            return False, f"Max open positions ({self.config.max_open_positions}) reached"

        today = date.today().isoformat()
        daily_loss = self.daily_pnl.get(today, 0.0)
        max_daily_loss = self.current_balance * (self.config.max_daily_loss_pct / 100)

        if daily_loss < 0 and abs(daily_loss) >= max_daily_loss:
            return False, f"Max daily loss ({self.config.max_daily_loss_pct}%) reached"

        return True, "OK"

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
    ) -> float:
        risk_amount = self.current_balance * (self.config.max_risk_per_trade_pct / 100)
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0.0

        quantity = risk_amount / price_risk
        position_value = quantity * entry_price

        max_position_value = self.current_balance * 0.3
        if position_value > max_position_value:
            quantity = max_position_value / entry_price

        return round(quantity, 8)

    def open_position(
        self,
        pair: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        quantity: float | None = None,
    ) -> Position | None:
        can_open, reason = self.can_open_position()
        if not can_open:
            return None

        if quantity is None:
            quantity = self.calculate_position_size(entry_price, stop_loss)

        if quantity <= 0:
            return None

        rr_ratio = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        if rr_ratio < self.config.min_risk_reward_ratio:
            return None

        position = Position(
            pair=pair,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            highest_price=entry_price,
            lowest_price=entry_price,
        )
        self.open_positions.append(position)
        return position

    def close_position(
        self,
        position: Position,
        exit_price: float,
        exit_reason: str,
    ) -> TradeRecord:
        pnl = position.calculate_pnl(exit_price)
        pnl_pct = (pnl / (position.entry_price * position.quantity)) * 100

        record = TradeRecord(
            pair=position.pair,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            exit_reason=exit_reason,
            opened_at=position.opened_at,
            closed_at=datetime.now(),
        )

        self.trade_history.append(record)
        self.current_balance += pnl
        self.open_positions.remove(position)

        today = date.today().isoformat()
        self.daily_pnl[today] = self.daily_pnl.get(today, 0.0) + pnl

        return record

    def update_positions(self, prices: dict[str, float]) -> list[TradeRecord]:
        closed_trades: list[TradeRecord] = []

        for position in list(self.open_positions):
            current_price = prices.get(position.pair)
            if current_price is None:
                continue

            position.update_trailing_stop(current_price, self.config.trailing_stop_pct)
            exit_reason = position.check_exit(current_price)

            if exit_reason:
                record = self.close_position(position, current_price, exit_reason)
                closed_trades.append(record)

        return closed_trades

    def get_stats(self) -> dict[str, float | int]:
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
                "profit_factor": 0.0,
                "current_balance": self.current_balance,
                "drawdown_pct": 0.0,
            }

        wins = [t for t in self.trade_history if t.pnl > 0]
        losses = [t for t in self.trade_history if t.pnl < 0]

        total_wins = sum(t.pnl for t in wins) if wins else 0
        total_losses = abs(sum(t.pnl for t in losses)) if losses else 0

        peak_balance = self.initial_balance
        running_balance = self.initial_balance
        max_drawdown = 0.0

        for trade in self.trade_history:
            running_balance += trade.pnl
            peak_balance = max(peak_balance, running_balance)
            drawdown = (peak_balance - running_balance) / peak_balance * 100
            max_drawdown = max(max_drawdown, drawdown)

        return {
            "total_trades": len(self.trade_history),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(self.trade_history) * 100 if self.trade_history else 0,
            "total_pnl": round(sum(t.pnl for t in self.trade_history), 2),
            "total_pnl_pct": round(
                (self.current_balance - self.initial_balance) / self.initial_balance * 100, 2
            ),
            "avg_win": round(total_wins / len(wins), 2) if wins else 0,
            "avg_loss": round(total_losses / len(losses), 2) if losses else 0,
            "max_win": round(max((t.pnl for t in wins), default=0), 2),
            "max_loss": round(min((t.pnl for t in losses), default=0), 2),
            "profit_factor": round(total_wins / total_losses, 2) if total_losses > 0 else float("inf"),
            "current_balance": round(self.current_balance, 2),
            "drawdown_pct": round(max_drawdown, 2),
        }
