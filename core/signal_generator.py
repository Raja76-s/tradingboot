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

        signals.sort(key=lambda s: s.quality_score, reverse=True)
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
                    best = signals[0] if signals else None
                    if best:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] No actionable signals. "
                              f"Best: {best.pair} "
                              f"conf={best.confidence}/100 "
                              f"RR=1:{best.risk_reward_ratio} "
                              f"SL={best.stop_loss_pct}% "
                              f"grade={best.grade}")
                    else:
                        print("No data")

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
        grade_color = Fore.GREEN if signal.grade == "A" else Fore.YELLOW if signal.grade == "B" else Fore.WHITE

        print(f"\n{color}{'=' * 55}")
        print(f"{arrow} {signal.action}  {signal.pair}  "
              f"{grade_color}[Grade: {signal.grade}  Quality: {signal.quality_score}/100]{color}")
        print(f"{'=' * 55}{Style.RESET_ALL}")
        print(f"  Confidence : {signal.confidence}/100")
        print(f"  Entry      : {signal.entry_price:,.4f}")
        print(f"  Stop-Loss  : {signal.stop_loss:,.4f}  (-{signal.stop_loss_pct}%)  ← MAX LOSS")
        print(f"  Target 1   : {signal.take_profit_1:,.4f}")
        print(f"  Target 2   : {signal.take_profit_2:,.4f}")
        print(f"  Risk/Reward: 1:{signal.risk_reward_ratio}  "
              f"({'GOOD' if signal.risk_reward_ratio >= 2 else 'OK' if signal.risk_reward_ratio >= 1.5 else 'WEAK'})")
        print(f"  Buy/Sell   : {signal.buy_count} buy | {signal.sell_count} sell | {signal.neutral_count} neutral")
        if signal.pattern_signals:
            print(f"  Patterns   : {', '.join(p.name for p in signal.pattern_signals)}")
        print()

    @staticmethod
    def format_telegram_message(signal: TradeSignal) -> str:
        emoji = "🟢" if signal.action == "BUY" else "🔴"
        grade = (
            "⭐⭐⭐ STRONG" if signal.confidence >= 85
            else "⭐⭐ GOOD" if signal.confidence >= 75
            else "⭐ MODERATE"
        )

        # Risk % from entry to stop
        if signal.entry_price > 0 and signal.stop_loss > 0:
            risk_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
            reward_pct = abs(signal.take_profit_1 - signal.entry_price) / signal.entry_price * 100
        else:
            risk_pct = reward_pct = 0

        core_signals = [
            i for i in signal.indicator_signals
            if i.name in (
                "Golden/Death Cross (50/200)", "MACD", "RSI",
                "Supertrend", "Ichimoku Cloud"
            )
        ]
        core_lines = "\n".join(
            f"  • {i.name}: {i.signal.value} — {i.details}"
            for i in core_signals
        )

        pattern_line = ""
        if signal.pattern_signals:
            names = ", ".join(p.name for p in signal.pattern_signals)
            pattern_line = f"\n📊 <b>Patterns:</b> {names}"

        lines = [
            f"{emoji} <b>{signal.action} — {signal.pair}</b>  {grade}",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"🎯 <b>Confidence:</b> {signal.confidence}/100",
            f"⏱ <b>Timeframe:</b> {signal.timeframe}",
            f"",
            f"💰 <b>Entry:</b>  {signal.entry_price:,.4f}",
            f"🛑 <b>Stop-Loss:</b>  {signal.stop_loss:,.4f}  (-{risk_pct:.1f}%)",
            f"🎯 <b>Target 1:</b>  {signal.take_profit_1:,.4f}  (+{reward_pct:.1f}%)",
            f"🚀 <b>Target 2:</b>  {signal.take_profit_2:,.4f}  (+{reward_pct*2:.1f}%)",
            f"⚖️ <b>Risk/Reward:</b>  1:{signal.risk_reward_ratio}",
            f"",
            f"📈 <b>Indicators:</b> {signal.buy_count} Buy | {signal.sell_count} Sell | {signal.neutral_count} Neutral",
            f"<b>Core Signals:</b>",
            core_lines,
        ]
        if pattern_line:
            lines.append(pattern_line)
        lines += [
            f"",
            f"⚠️ Always use stop-loss. Risk max 1-2% of capital.",
            f"🕐 {signal.timestamp.strftime('%d %b %Y  %H:%M:%S')}",
        ]
        return "\n".join(lines)
