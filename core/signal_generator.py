"""
Real-time Signal Generator.

Monitors live market data and generates trading signals
using the scoring engine with all indicators and patterns.
"""

import time
from datetime import datetime

from colorama import Fore, Style

from config.settings import AppConfig
from core.data_fetcher import DataFetcher
from core.scoring_engine import MultiTimeframeScoringEngine, ScoringEngine, TradeSignal


class SignalGenerator:
    """Generates trading signals in real-time."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.data_fetcher = DataFetcher(config)
        self.scoring_engine = ScoringEngine(config.strategy)
        self.mtf_engine = MultiTimeframeScoringEngine(config.strategy)
        self.signal_history: list[TradeSignal] = []

    def scan_pair(
        self,
        pair: str,
        timeframe: str | None = None,
    ) -> TradeSignal:
        if timeframe is None:
            timeframe = self.config.strategy.default_timeframe

        df = self.data_fetcher.fetch_ohlcv(pair, timeframe, 500)
        signal = self.scoring_engine.analyze(df, pair, timeframe)
        self.signal_history.append(signal)
        return signal

    def scan_pair_multi_tf(self, pair: str) -> TradeSignal:
        data = self.data_fetcher.fetch_multi_timeframe(
            pair, ["5m", "15m", "1h", "4h"], limit=300
        )
        signal = self.mtf_engine.analyze_multi_tf(data, pair)
        self.signal_history.append(signal)
        return signal

    def scan_all_pairs(
        self,
        multi_timeframe: bool = False,
    ) -> list[TradeSignal]:
        signals: list[TradeSignal] = []

        for pair in self.config.trading.trading_pairs:
            try:
                if multi_timeframe:
                    signal = self.scan_pair_multi_tf(pair)
                else:
                    signal = self.scan_pair(pair)
                signals.append(signal)
            except Exception as e:
                print(f"Error scanning {pair}: {e}")
                continue

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    def run_continuous(
        self,
        interval_seconds: int = 60,
        callback: object | None = None,
    ) -> None:
        """Run continuous signal scanning."""
        print(f"\n{Fore.CYAN}Starting Signal Generator (Multi-Timeframe)...{Style.RESET_ALL}")
        print(f"Pairs: {', '.join(self.config.trading.trading_pairs)}")
        print(f"Timeframes: 5m, 15m, 1h, 4h (combined analysis)")
        print(f"Scan interval: {interval_seconds}s")
        print(f"Min confidence: {self.config.strategy.min_confidence_score}")
        print("-" * 60)

        while True:
            try:
                signals = self.scan_all_pairs(multi_timeframe=True)
                actionable = [
                    s for s in signals
                    if s.action != "HOLD"
                    and s.confidence >= self.config.strategy.min_confidence_score
                ]

                if actionable:
                    print(f"\n{Fore.YELLOW}[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Actionable Signals:{Style.RESET_ALL}")
                    for sig in actionable:
                        self._print_signal(sig)

                    if callback and callable(callback):
                        callback(actionable)
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No actionable signals. "
                          f"Best: {signals[0].pair} ({signals[0].confidence}/100) "
                          if signals else "No data")

                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                print(f"\n{Fore.RED}Signal generator stopped.{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)

    def _print_signal(self, signal: TradeSignal) -> None:
        color = Fore.GREEN if signal.action == "BUY" else Fore.RED
        arrow = "▲" if signal.action == "BUY" else "▼"

        print(f"\n{color}{'=' * 50}")
        print(f"{arrow} {signal.action} Signal — {signal.pair}")
        print(f"{'=' * 50}{Style.RESET_ALL}")
        print(f"  Confidence: {signal.confidence}/100")
        print(f"  Entry Price: {signal.entry_price:,.2f}")
        print(f"  Stop-Loss: {signal.stop_loss:,.2f}")
        print(f"  Target 1: {signal.take_profit_1:,.2f}")
        print(f"  Target 2: {signal.take_profit_2:,.2f}")
        print(f"  Risk/Reward: 1:{signal.risk_reward_ratio}")
        print(f"  Buy Signals: {signal.buy_count} | Sell Signals: {signal.sell_count}")

        if signal.pattern_signals:
            print(f"  Patterns: {', '.join(p.name for p in signal.pattern_signals)}")

        print()

    @staticmethod
    def format_telegram_message(signal: TradeSignal) -> str:
        emoji = "🟢" if signal.action == "BUY" else "🔴" if signal.action == "SELL" else "⚪"

        msg_lines = [
            f"{emoji} {signal.action} Signal — {signal.pair}",
            "",
            f"Confidence: {signal.confidence}/100",
            f"Entry Price: {signal.entry_price:,.2f}",
            f"Stop-Loss: {signal.stop_loss:,.2f}",
            f"Target 1: {signal.take_profit_1:,.2f}",
            f"Target 2: {signal.take_profit_2:,.2f}",
            f"Risk/Reward: 1:{signal.risk_reward_ratio}",
            "",
            f"Indicators: {signal.buy_count} Buy | {signal.sell_count} Sell",
        ]

        if signal.pattern_signals:
            msg_lines.append(
                f"Patterns: {', '.join(p.name for p in signal.pattern_signals)}"
            )

        key_indicators = [
            i for i in signal.indicator_signals
            if i.signal.value in ("STRONG_BUY", "STRONG_SELL")
        ]
        if key_indicators:
            msg_lines.append("\nKey Signals:")
            for ind in key_indicators[:5]:
                msg_lines.append(f"  {ind.name}: {ind.details}")

        msg_lines.append(f"\nTimeframe: {signal.timeframe}")
        msg_lines.append(f"Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(msg_lines)
