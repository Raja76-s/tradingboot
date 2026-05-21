"""
Auto-Trading Bot.

Supports both paper trading (simulated) and live trading (CoinDCX API).
Uses signals from the scoring engine and risk management rules.
"""

import time
from datetime import datetime

from colorama import Fore, Style

from config.settings import AppConfig
from core.data_fetcher import CoinDCXClient, DataFetcher
from core.risk_manager import Position, RiskManager, TradeRecord
from core.signal_generator import SignalGenerator


class PaperTrader:
    """Simulated trading without real money."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.risk_manager = RiskManager(config.risk, config.trading.initial_balance)
        self.signal_generator = SignalGenerator(config)
        self.data_fetcher = DataFetcher(config)
        self.is_running = False

    def run(
        self,
        interval_seconds: int = 60,
        on_trade: object | None = None,
    ) -> None:
        self.is_running = True
        print(f"\n{Fore.CYAN}{'=' * 60}")
        print("PAPER TRADING MODE")
        print(f"{'=' * 60}{Style.RESET_ALL}")
        print(f"Initial Balance: {self.config.trading.initial_balance:,.2f} "
              f"{self.config.trading.currency}")
        print(f"Pairs: {', '.join(self.config.trading.trading_pairs)}")
        print(f"Max risk per trade: {self.config.risk.max_risk_per_trade_pct}%")
        print(f"Max daily loss: {self.config.risk.max_daily_loss_pct}%")
        print("-" * 60)

        while self.is_running:
            try:
                self._trading_cycle(on_trade)
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                print(f"\n{Fore.RED}Paper trading stopped.{Style.RESET_ALL}")
                self._print_final_stats()
                break
            except Exception as e:
                print(f"Error in trading cycle: {e}")
                time.sleep(5)

    def _trading_cycle(self, on_trade: object | None = None) -> None:
        prices = self._get_current_prices()
        closed_trades = self.risk_manager.update_positions(prices)

        for trade in closed_trades:
            self._print_trade_closed(trade)
            if on_trade and callable(on_trade):
                on_trade(trade)

        signals = self.signal_generator.scan_all_pairs()

        for signal in signals:
            if signal.action == "HOLD":
                continue
            if signal.confidence < self.config.strategy.min_confidence:
                continue

            can_open, reason = self.risk_manager.can_open_position()
            if not can_open:
                continue

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

    def _print_trade_opened(self, position: Position, confidence: int) -> None:
        color = Fore.GREEN if position.side == "BUY" else Fore.RED
        print(f"\n{color}[TRADE OPENED] {position.side} {position.pair}{Style.RESET_ALL}")
        print(f"  Entry: {position.entry_price:,.2f}")
        print(f"  Qty: {position.quantity:.6f}")
        print(f"  Stop-Loss: {position.stop_loss:,.2f}")
        print(f"  Take-Profit: {position.take_profit:,.2f}")
        print(f"  Confidence: {confidence}/100")

    def _print_trade_closed(self, trade: TradeRecord) -> None:
        color = Fore.GREEN if trade.pnl > 0 else Fore.RED
        print(f"\n{color}[TRADE CLOSED] {trade.side} {trade.pair}{Style.RESET_ALL}")
        print(f"  Entry: {trade.entry_price:,.2f} → Exit: {trade.exit_price:,.2f}")
        print(f"  P&L: {trade.pnl:+,.2f} ({trade.pnl_pct:+.2f}%)")
        print(f"  Reason: {trade.exit_reason}")

    def _print_status(self) -> None:
        stats = self.risk_manager.get_stats()
        positions = self.risk_manager.open_positions

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"Balance: {stats['current_balance']:,.2f} | "
              f"Open: {len(positions)} | "
              f"Trades: {stats['total_trades']} | "
              f"Win Rate: {stats['win_rate']:.1f}% | "
              f"P&L: {stats['total_pnl']:+,.2f}")

    def _print_final_stats(self) -> None:
        stats = self.risk_manager.get_stats()
        print(f"\n{'=' * 60}")
        print("PAPER TRADING FINAL RESULTS")
        print(f"{'=' * 60}")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Total P&L: {stats['total_pnl']:+,.2f}")
        print(f"Return: {stats['total_pnl_pct']:+.2f}%")
        print(f"Final Balance: {stats['current_balance']:,.2f}")
        print(f"Max Drawdown: {stats['drawdown_pct']:.2f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"{'=' * 60}")

    def stop(self) -> None:
        self.is_running = False

    def get_stats(self) -> dict[str, float | int]:
        return self.risk_manager.get_stats()


class LiveTrader:
    """Live trading via CoinDCX API."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = CoinDCXClient(config)
        self.risk_manager = RiskManager(config.risk, config.trading.initial_balance)
        self.signal_generator = SignalGenerator(config)
        self.is_running = False

        if not config.coindcx.api_key or not config.coindcx.api_secret:
            raise ValueError(
                "CoinDCX API key and secret are required for live trading. "
                "Set COINDCX_API_KEY and COINDCX_API_SECRET environment variables."
            )

    def run(self, interval_seconds: int = 60) -> None:
        self.is_running = True
        print(f"\n{Fore.RED}{'=' * 60}")
        print("⚠️  LIVE TRADING MODE - REAL MONEY ⚠️")
        print(f"{'=' * 60}{Style.RESET_ALL}")
        print(f"Pairs: {', '.join(self.config.trading.trading_pairs)}")
        print(f"Max risk per trade: {self.config.risk.max_risk_per_trade_pct}%")
        print(f"Max daily loss: {self.config.risk.max_daily_loss_pct}%")
        print("-" * 60)

        # Verify API connection
        try:
            balances = self.client.get_balances()
            print(f"API connection verified. Found {len(balances)} assets.")
        except Exception as e:
            print(f"API connection failed: {e}")
            return

        while self.is_running:
            try:
                self._trading_cycle()
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                print(f"\n{Fore.RED}Live trading stopped.{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)

    def _trading_cycle(self) -> None:
        signals = self.signal_generator.scan_all_pairs()

        for signal in signals:
            if signal.action == "HOLD":
                continue
            if signal.confidence < self.config.strategy.min_confidence:
                continue

            can_open, reason = self.risk_manager.can_open_position()
            if not can_open:
                continue

            quantity = self.risk_manager.calculate_position_size(
                signal.entry_price, signal.stop_loss
            )

            if quantity <= 0:
                continue

            try:
                order = self.client.place_order(
                    pair=signal.pair,
                    side=signal.action.lower(),
                    order_type="market_order",
                    quantity=quantity,
                )
                print(f"\n{Fore.GREEN}[LIVE ORDER] {signal.action} {signal.pair}")
                print(f"  Qty: {quantity}, Order: {order}{Style.RESET_ALL}")

                self.risk_manager.open_position(
                    pair=signal.pair,
                    side=signal.action,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit_1,
                    quantity=quantity,
                )
            except Exception as e:
                print(f"Order failed for {signal.pair}: {e}")

    def stop(self) -> None:
        self.is_running = False
