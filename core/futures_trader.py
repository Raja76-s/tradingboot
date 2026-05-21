"""
Futures/Derivatives Trading Module.

Supports:
- Long and Short positions (profit from both directions)
- Leverage (2x, 5x, 10x, etc.)
- Margin management and liquidation price calculation
- Funding rate monitoring
- Futures-specific risk management
"""

import time
from dataclasses import dataclass, field
from datetime import datetime

from colorama import Fore, Style

from config.settings import AppConfig
from core.data_fetcher import DataFetcher
from core.signal_generator import SignalGenerator


@dataclass
class FuturesConfig:
    leverage: int = 5
    max_leverage: int = 10
    margin_type: str = "isolated"  # "isolated" or "cross"
    max_margin_usage_pct: float = 50.0
    liquidation_buffer_pct: float = 5.0


@dataclass
class FuturesPosition:
    pair: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    leverage: int
    margin: float
    stop_loss: float
    take_profit: float
    trailing_stop: float | None = None
    liquidation_price: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    unrealized_pnl: float = 0.0
    funding_paid: float = 0.0

    def __post_init__(self) -> None:
        self.liquidation_price = self._calculate_liquidation_price()
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price

    def _calculate_liquidation_price(self) -> float:
        """Calculate liquidation price based on leverage and margin."""
        if self.leverage <= 0:
            return 0.0

        margin_ratio = 1.0 / self.leverage
        maintenance_margin = 0.005  # 0.5% maintenance margin

        if self.side == "LONG":
            return self.entry_price * (1 - margin_ratio + maintenance_margin)
        else:
            return self.entry_price * (1 + margin_ratio - maintenance_margin)

    def update_trailing_stop(
        self, current_price: float, trailing_pct: float
    ) -> None:
        if self.side == "LONG":
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

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL including leverage."""
        if self.side == "LONG":
            price_change_pct = (current_price - self.entry_price) / self.entry_price
        else:
            price_change_pct = (self.entry_price - current_price) / self.entry_price

        self.unrealized_pnl = self.margin * self.leverage * price_change_pct
        return self.unrealized_pnl

    def calculate_roe(self, current_price: float) -> float:
        """Calculate Return on Equity (margin)."""
        pnl = self.calculate_pnl(current_price)
        return (pnl / self.margin * 100) if self.margin > 0 else 0.0

    def check_exit(self, current_price: float) -> str | None:
        # Check liquidation
        if self.side == "LONG" and current_price <= self.liquidation_price:
            return "LIQUIDATED"
        if self.side == "SHORT" and current_price >= self.liquidation_price:
            return "LIQUIDATED"

        # Check stop-loss
        if self.side == "LONG" and current_price <= self.stop_loss:
            return "STOP_LOSS"
        if self.side == "SHORT" and current_price >= self.stop_loss:
            return "STOP_LOSS"

        # Check trailing stop
        if self.trailing_stop:
            if self.side == "LONG" and current_price <= self.trailing_stop:
                return "TRAILING_STOP"
            if self.side == "SHORT" and current_price >= self.trailing_stop:
                return "TRAILING_STOP"

        # Check take-profit
        if self.side == "LONG" and current_price >= self.take_profit:
            return "TAKE_PROFIT"
        if self.side == "SHORT" and current_price <= self.take_profit:
            return "TAKE_PROFIT"

        return None


@dataclass
class FuturesTradeRecord:
    pair: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    margin: float
    pnl: float
    roe: float  # Return on equity/margin
    exit_reason: str
    opened_at: datetime
    closed_at: datetime
    funding_paid: float = 0.0


class FuturesRiskManager:
    """Risk management for futures trading."""

    def __init__(
        self,
        futures_config: FuturesConfig,
        risk_config: object,
        initial_balance: float,
    ) -> None:
        self.futures_config = futures_config
        self.risk_config = risk_config
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.available_margin = initial_balance
        self.open_positions: list[FuturesPosition] = []
        self.trade_history: list[FuturesTradeRecord] = []

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        leverage: int | None = None,
    ) -> tuple[float, float]:
        """Calculate position size and required margin.

        Returns (quantity, margin).
        """
        if leverage is None:
            leverage = self.futures_config.leverage

        max_margin = self.available_margin * (
            self.futures_config.max_margin_usage_pct / 100
        )
        risk_amount = self.current_balance * 0.015  # 1.5% risk per trade

        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            return 0.0, 0.0

        # Position size based on risk
        notional = risk_amount / (price_risk / entry_price)
        margin = notional / leverage

        # Cap margin to available
        if margin > max_margin:
            margin = max_margin
            notional = margin * leverage

        quantity = notional / entry_price
        return round(quantity, 8), round(margin, 2)

    def open_position(
        self,
        pair: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        leverage: int | None = None,
    ) -> FuturesPosition | None:
        if leverage is None:
            leverage = self.futures_config.leverage

        if leverage > self.futures_config.max_leverage:
            leverage = self.futures_config.max_leverage

        quantity, margin = self.calculate_position_size(
            entry_price, stop_loss, leverage
        )

        if quantity <= 0 or margin <= 0:
            return None

        if margin > self.available_margin:
            return None

        # Map BUY/SELL to LONG/SHORT for futures
        futures_side = "LONG" if side in ("BUY", "LONG") else "SHORT"

        position = FuturesPosition(
            pair=pair,
            side=futures_side,
            entry_price=entry_price,
            quantity=quantity,
            leverage=leverage,
            margin=margin,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.open_positions.append(position)
        self.available_margin -= margin
        return position

    def close_position(
        self,
        position: FuturesPosition,
        exit_price: float,
        exit_reason: str,
    ) -> FuturesTradeRecord:
        pnl = position.calculate_pnl(exit_price)
        roe = position.calculate_roe(exit_price)

        if exit_reason == "LIQUIDATED":
            pnl = -position.margin  # Lose entire margin

        record = FuturesTradeRecord(
            pair=position.pair,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            leverage=position.leverage,
            margin=position.margin,
            pnl=round(pnl, 2),
            roe=round(roe, 2),
            exit_reason=exit_reason,
            opened_at=position.opened_at,
            closed_at=datetime.now(),
            funding_paid=position.funding_paid,
        )

        self.trade_history.append(record)
        self.current_balance += pnl
        self.available_margin += position.margin + pnl
        self.open_positions.remove(position)

        return record

    def update_positions(
        self, prices: dict[str, float], trailing_pct: float = 1.5
    ) -> list[FuturesTradeRecord]:
        closed: list[FuturesTradeRecord] = []

        for position in list(self.open_positions):
            current_price = prices.get(position.pair)
            if current_price is None:
                continue

            position.update_trailing_stop(current_price, trailing_pct)
            exit_reason = position.check_exit(current_price)

            if exit_reason:
                record = self.close_position(position, current_price, exit_reason)
                closed.append(record)

        return closed

    def get_stats(self) -> dict[str, float | int]:
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "liquidations": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
                "avg_roe": 0.0,
                "best_roe": 0.0,
                "worst_roe": 0.0,
                "profit_factor": 0.0,
                "current_balance": self.current_balance,
                "available_margin": self.available_margin,
                "total_margin_used": sum(p.margin for p in self.open_positions),
            }

        wins = [t for t in self.trade_history if t.pnl > 0]
        losses = [t for t in self.trade_history if t.pnl <= 0]
        liquidations = [t for t in self.trade_history if t.exit_reason == "LIQUIDATED"]

        total_wins = sum(t.pnl for t in wins) if wins else 0
        total_losses = abs(sum(t.pnl for t in losses)) if losses else 0

        return {
            "total_trades": len(self.trade_history),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "liquidations": len(liquidations),
            "win_rate": len(wins) / len(self.trade_history) * 100,
            "total_pnl": round(sum(t.pnl for t in self.trade_history), 2),
            "total_pnl_pct": round(
                (self.current_balance - self.initial_balance)
                / self.initial_balance
                * 100,
                2,
            ),
            "avg_roe": round(
                sum(t.roe for t in self.trade_history) / len(self.trade_history), 2
            ),
            "best_roe": round(max(t.roe for t in self.trade_history), 2),
            "worst_roe": round(min(t.roe for t in self.trade_history), 2),
            "profit_factor": (
                round(total_wins / total_losses, 2) if total_losses > 0 else float("inf")
            ),
            "current_balance": round(self.current_balance, 2),
            "available_margin": round(self.available_margin, 2),
            "total_margin_used": round(
                sum(p.margin for p in self.open_positions), 2
            ),
        }


class FuturesPaperTrader:
    """Paper trading for futures with leverage."""

    def __init__(self, config: AppConfig, futures_config: FuturesConfig | None = None) -> None:
        self.config = config
        self.futures_config = futures_config or FuturesConfig()
        self.risk_manager = FuturesRiskManager(
            self.futures_config,
            config.risk,
            config.trading.initial_balance,
        )
        self.signal_generator = SignalGenerator(config)
        self.data_fetcher = DataFetcher(config)
        self.is_running = False

    def run(self, interval_seconds: int = 60) -> None:
        self.is_running = True
        print(f"\n{Fore.CYAN}{'=' * 60}")
        print("FUTURES PAPER TRADING MODE")
        print(f"{'=' * 60}{Style.RESET_ALL}")
        print(f"Initial Balance: {self.config.trading.initial_balance:,.2f}")
        print(f"Leverage: {self.futures_config.leverage}x")
        print(f"Margin Type: {self.futures_config.margin_type}")
        print(f"Max Margin Usage: {self.futures_config.max_margin_usage_pct}%")
        print(f"Pairs: {', '.join(self.config.trading.trading_pairs)}")
        print("-" * 60)

        while self.is_running:
            try:
                self._trading_cycle()
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                print(f"\n{Fore.RED}Futures paper trading stopped.{Style.RESET_ALL}")
                self._print_final_stats()
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)

    def _trading_cycle(self) -> None:
        prices = self._get_current_prices()
        closed = self.risk_manager.update_positions(prices)

        for trade in closed:
            self._print_trade_closed(trade)

        signals = self.signal_generator.scan_all_pairs()

        for signal in signals:
            if signal.action == "HOLD":
                continue
            if signal.confidence < self.config.strategy.min_confidence_score:
                continue

            # For futures: BUY = LONG, SELL = SHORT
            already_in = any(
                p.pair == signal.pair for p in self.risk_manager.open_positions
            )
            if already_in:
                continue

            position = self.risk_manager.open_position(
                pair=signal.pair,
                side=signal.action,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit_1,
            )

            if position:
                self._print_trade_opened(position, signal.confidence)

        self._print_status()

    def _get_current_prices(self) -> dict[str, float]:
        prices: dict[str, float] = {}
        for pair in self.config.trading.trading_pairs:
            try:
                df = self.data_fetcher.fetch_ohlcv(pair, "1m", 1)
                if not df.empty:
                    prices[pair] = df["close"].iloc[-1]
            except Exception:
                continue
        return prices

    def _print_trade_opened(self, position: FuturesPosition, confidence: int) -> None:
        color = Fore.GREEN if position.side == "LONG" else Fore.RED
        print(f"\n{color}[FUTURES OPENED] {position.side} {position.pair} "
              f"@ {position.leverage}x{Style.RESET_ALL}")
        print(f"  Entry: {position.entry_price:,.2f}")
        print(f"  Margin: {position.margin:,.2f}")
        print(f"  Notional: {position.margin * position.leverage:,.2f}")
        print(f"  Stop-Loss: {position.stop_loss:,.2f}")
        print(f"  Take-Profit: {position.take_profit:,.2f}")
        print(f"  Liquidation: {position.liquidation_price:,.2f}")
        print(f"  Confidence: {confidence}/100")

    def _print_trade_closed(self, trade: FuturesTradeRecord) -> None:
        color = Fore.GREEN if trade.pnl > 0 else Fore.RED
        emoji = "LIQUIDATED!" if trade.exit_reason == "LIQUIDATED" else trade.exit_reason
        print(f"\n{color}[FUTURES CLOSED] {trade.side} {trade.pair} "
              f"@ {trade.leverage}x{Style.RESET_ALL}")
        print(f"  Entry: {trade.entry_price:,.2f} -> Exit: {trade.exit_price:,.2f}")
        print(f"  P&L: {trade.pnl:+,.2f} (ROE: {trade.roe:+.2f}%)")
        print(f"  Reason: {emoji}")

    def _print_status(self) -> None:
        stats = self.risk_manager.get_stats()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"Balance: {stats['current_balance']:,.2f} | "
              f"Margin Used: {stats['total_margin_used']:,.2f} | "
              f"Open: {len(self.risk_manager.open_positions)} | "
              f"Trades: {stats['total_trades']} | "
              f"Win Rate: {stats['win_rate']:.1f}%")

    def _print_final_stats(self) -> None:
        stats = self.risk_manager.get_stats()
        print(f"\n{'=' * 60}")
        print("FUTURES PAPER TRADING RESULTS")
        print(f"{'=' * 60}")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Liquidations: {stats['liquidations']}")
        print(f"Total P&L: {stats['total_pnl']:+,.2f}")
        print(f"Return: {stats['total_pnl_pct']:+.2f}%")
        print(f"Avg ROE: {stats['avg_roe']:+.2f}%")
        print(f"Best ROE: {stats['best_roe']:+.2f}%")
        print(f"Worst ROE: {stats['worst_roe']:+.2f}%")
        print(f"Final Balance: {stats['current_balance']:,.2f}")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"{'=' * 60}")

    def stop(self) -> None:
        self.is_running = False
