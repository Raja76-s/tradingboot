"""
Backtesting Engine.

Tests trading strategies on historical data to evaluate performance
before risking real money.
"""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from config.settings import AppConfig
from core.risk_manager import RiskManager, TradeRecord
from core.scoring_engine import ScoringEngine


@dataclass
class BacktestConfig:
    pair: str = "BTCUSDT"
    timeframe: str = "15m"
    initial_balance: float = 20000.0
    min_confidence: int = 65
    use_trailing_stop: bool = True


@dataclass
class BacktestResult:
    config: BacktestConfig
    stats: dict[str, float | int]
    trades: list[TradeRecord]
    equity_curve: list[float]
    timestamps: list[datetime]
    signals_generated: int = 0
    trades_taken: int = 0
    trades_skipped: int = 0


class Backtester:
    """Runs strategies against historical data."""

    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config
        self.scoring_engine = ScoringEngine(app_config.strategy)

    def run(
        self,
        df: pd.DataFrame,
        config: BacktestConfig | None = None,
        step: int = 1,
    ) -> BacktestResult:
        if config is None:
            config = BacktestConfig()

        risk_manager = RiskManager(self.app_config.risk, config.initial_balance)
        equity_curve: list[float] = [config.initial_balance]
        timestamps: list[datetime] = []
        signals_generated = 0
        trades_taken = 0
        trades_skipped = 0

        lookback = min(max(60, self.app_config.strategy.lookback_periods), len(df) - 10)
        if lookback < 60 or len(df) <= lookback:
            return BacktestResult(
                config=config,
                stats=risk_manager.get_stats(),
                trades=[],
                equity_curve=[config.initial_balance],
                timestamps=[],
                signals_generated=0,
                trades_taken=0,
                trades_skipped=0,
            )

        for i in range(lookback, len(df), step):
            window = df.iloc[i - lookback : i + 1]
            current_price = df["close"].iloc[i]
            current_time = df.index[i]

            # Update existing positions
            prices = {config.pair: current_price}
            risk_manager.update_positions(prices)

            # Generate signal
            signal = self.scoring_engine.analyze(window, config.pair, config.timeframe)
            signals_generated += 1

            # Check if we should open a new position
            if signal.action in ("BUY", "SELL") and signal.confidence >= config.min_confidence:
                can_open, reason = risk_manager.can_open_position()

                if can_open:
                    # Check for conflicting positions
                    has_conflicting = any(
                        p.pair == config.pair and p.side != signal.action
                        for p in risk_manager.open_positions
                    )

                    if has_conflicting:
                        for p in list(risk_manager.open_positions):
                            if p.pair == config.pair and p.side != signal.action:
                                risk_manager.close_position(
                                    p, current_price, "SIGNAL_REVERSAL"
                                )

                    # Check we don't already have same-side position
                    has_same = any(
                        p.pair == config.pair and p.side == signal.action
                        for p in risk_manager.open_positions
                    )

                    if not has_same:
                        position = risk_manager.open_position(
                            pair=config.pair,
                            side=signal.action,
                            entry_price=current_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit_1,
                        )
                        if position:
                            trades_taken += 1
                        else:
                            trades_skipped += 1
                else:
                    trades_skipped += 1

            equity_curve.append(risk_manager.current_balance)
            if isinstance(current_time, pd.Timestamp):
                timestamps.append(current_time.to_pydatetime())
            else:
                timestamps.append(current_time)

        # Close any remaining positions at the end
        last_price = df["close"].iloc[-1]
        for position in list(risk_manager.open_positions):
            risk_manager.close_position(position, last_price, "BACKTEST_END")

        stats = risk_manager.get_stats()

        return BacktestResult(
            config=config,
            stats=stats,
            trades=risk_manager.trade_history,
            equity_curve=equity_curve,
            timestamps=timestamps,
            signals_generated=signals_generated,
            trades_taken=trades_taken,
            trades_skipped=trades_skipped,
        )

    def run_optimization(
        self,
        df: pd.DataFrame,
        pair: str = "BTCUSDT",
    ) -> list[BacktestResult]:
        """Run multiple backtests with different confidence thresholds."""
        results: list[BacktestResult] = []
        confidence_levels = [55, 60, 65, 70, 75, 80]

        for conf in confidence_levels:
            config = BacktestConfig(
                pair=pair,
                min_confidence=conf,
            )
            result = self.run(df, config, step=5)
            results.append(result)

        results.sort(key=lambda r: r.stats.get("total_pnl", 0), reverse=True)
        return results

    def print_results(self, result: BacktestResult) -> str:
        stats = result.stats
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Pair: {result.config.pair}",
            f"Timeframe: {result.config.timeframe}",
            f"Initial Balance: {result.config.initial_balance:,.2f}",
            f"Min Confidence: {result.config.min_confidence}",
            "-" * 60,
            f"Signals Generated: {result.signals_generated}",
            f"Trades Taken: {result.trades_taken}",
            f"Trades Skipped: {result.trades_skipped}",
            "-" * 60,
            f"Total Trades: {stats['total_trades']}",
            f"Winning Trades: {stats['winning_trades']}",
            f"Losing Trades: {stats['losing_trades']}",
            f"Win Rate: {stats['win_rate']:.1f}%",
            "-" * 60,
            f"Total P&L: {stats['total_pnl']:,.2f}",
            f"Total Return: {stats['total_pnl_pct']:.2f}%",
            f"Final Balance: {stats['current_balance']:,.2f}",
            "-" * 60,
            f"Avg Win: {stats['avg_win']:,.2f}",
            f"Avg Loss: {stats['avg_loss']:,.2f}",
            f"Max Win: {stats['max_win']:,.2f}",
            f"Max Loss: {stats['max_loss']:,.2f}",
            f"Profit Factor: {stats['profit_factor']:.2f}",
            f"Max Drawdown: {stats['drawdown_pct']:.2f}%",
            "=" * 60,
        ]

        return "\n".join(lines)
