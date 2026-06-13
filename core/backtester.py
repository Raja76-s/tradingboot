"""
Backtesting Engine — realistic trade simulation.

Improvements:
- Only enter on candle CLOSE (not mid-candle)
- Trailing stop using ATR
- No re-entry until previous trade closed
- Minimum bars between trades (avoid overtrading)
- Proper slippage simulation
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
    min_bars_between_trades: int = 3   # avoid overtrading
    slippage_pct: float = 0.05         # 0.05% slippage per trade


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

    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config
        self.scoring_engine = ScoringEngine(app_config.strategy)

    def run(self, df: pd.DataFrame, config: BacktestConfig | None = None, step: int = 1) -> BacktestResult:
        if config is None:
            config = BacktestConfig()

        risk_manager = RiskManager(self.app_config.risk, config.initial_balance)
        equity_curve: list[float] = [config.initial_balance]
        timestamps: list[datetime] = []
        signals_generated = 0
        trades_taken = 0
        trades_skipped = 0
        last_trade_bar = -999  # track when last trade was opened

        lookback = min(max(60, self.app_config.strategy.lookback_periods), len(df) - 10)
        if lookback < 60 or len(df) <= lookback:
            return BacktestResult(
                config=config, stats=risk_manager.get_stats(),
                trades=[], equity_curve=[config.initial_balance],
                timestamps=[], signals_generated=0, trades_taken=0, trades_skipped=0,
            )

        # Pre-calculate ATR for trailing stop
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_series = tr.ewm(alpha=1.0 / 14, adjust=False).mean()

        for i in range(lookback, len(df), step):
            window = df.iloc[i - lookback: i + 1]
            current_price = float(df["close"].iloc[i])
            current_time = df.index[i]
            atr_now = float(atr_series.iloc[i])

            # Trailing stop update for open positions
            if config.use_trailing_stop:
                for pos in list(risk_manager.open_positions):
                    if pos.side == "BUY":
                        new_sl = current_price - atr_now * 2.0
                        if new_sl > pos.stop_loss:
                            pos.stop_loss = new_sl
                    else:
                        new_sl = current_price + atr_now * 2.0
                        if new_sl < pos.stop_loss:
                            pos.stop_loss = new_sl

            # Update positions with current price
            prices = {config.pair: current_price}
            risk_manager.update_positions(prices)

            # Generate signal
            signal = self.scoring_engine.analyze(window, config.pair, config.timeframe)
            signals_generated += 1

            # Only trade if:
            # 1. Signal is actionable
            # 2. Confidence meets threshold
            # 3. Minimum bars since last trade
            # 4. Grade is not SKIP
            bars_since_last = i - last_trade_bar
            if (signal.action in ("BUY", "SELL")
                    and signal.confidence >= config.min_confidence
                    and signal.grade != "SKIP"
                    and bars_since_last >= config.min_bars_between_trades):

                can_open, _ = risk_manager.can_open_position()

                if can_open:
                    # Close conflicting positions
                    for p in list(risk_manager.open_positions):
                        if p.pair == config.pair and p.side != signal.action:
                            risk_manager.close_position(p, current_price, "SIGNAL_REVERSAL")

                    # No same-side duplicate
                    has_same = any(
                        p.pair == config.pair and p.side == signal.action
                        for p in risk_manager.open_positions
                    )

                    if not has_same:
                        # Apply slippage
                        if signal.action == "BUY":
                            entry = current_price * (1 + config.slippage_pct / 100)
                        else:
                            entry = current_price * (1 - config.slippage_pct / 100)

                        position = risk_manager.open_position(
                            pair=config.pair,
                            side=signal.action,
                            entry_price=entry,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit_1,
                        )
                        if position:
                            trades_taken += 1
                            last_trade_bar = i
                        else:
                            trades_skipped += 1
                else:
                    trades_skipped += 1

            equity_curve.append(risk_manager.current_balance)
            if isinstance(current_time, pd.Timestamp):
                timestamps.append(current_time.to_pydatetime())
            else:
                timestamps.append(current_time)

        # Close remaining positions
        last_price = float(df["close"].iloc[-1])
        for position in list(risk_manager.open_positions):
            risk_manager.close_position(position, last_price, "BACKTEST_END")

        return BacktestResult(
            config=config,
            stats=risk_manager.get_stats(),
            trades=risk_manager.trade_history,
            equity_curve=equity_curve,
            timestamps=timestamps,
            signals_generated=signals_generated,
            trades_taken=trades_taken,
            trades_skipped=trades_skipped,
        )

    def run_optimization(self, df: pd.DataFrame, pair: str = "BTCUSDT") -> list[BacktestResult]:
        results: list[BacktestResult] = []
        for conf in [55, 60, 65, 70, 75, 80]:
            config = BacktestConfig(pair=pair, min_confidence=conf)
            result = self.run(df, config, step=5)
            results.append(result)
        results.sort(key=lambda r: r.stats.get("total_pnl", 0), reverse=True)
        return results

    def print_results(self, result: BacktestResult) -> str:
        s = result.stats
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Pair:            {result.config.pair}",
            f"Timeframe:       {result.config.timeframe}",
            f"Initial Balance: {result.config.initial_balance:,.2f}",
            f"Min Confidence:  {result.config.min_confidence}",
            f"Trailing Stop:   {'ON' if result.config.use_trailing_stop else 'OFF'}",
            f"Slippage:        {result.config.slippage_pct}%",
            "-" * 60,
            f"Signals Generated: {result.signals_generated}",
            f"Trades Taken:      {result.trades_taken}",
            f"Trades Skipped:    {result.trades_skipped}",
            "-" * 60,
            f"Total Trades:    {s['total_trades']}",
            f"Winning Trades:  {s['winning_trades']}",
            f"Losing Trades:   {s['losing_trades']}",
            f"Win Rate:        {s['win_rate']:.1f}%",
            "-" * 60,
            f"Total P&L:       {s['total_pnl']:,.2f}",
            f"Total Return:    {s['total_pnl_pct']:.2f}%",
            f"Final Balance:   {s['current_balance']:,.2f}",
            "-" * 60,
            f"Avg Win:         {s['avg_win']:,.2f}",
            f"Avg Loss:        {s['avg_loss']:,.2f}",
            f"Profit Factor:   {s['profit_factor']:.2f}",
            f"Max Drawdown:    {s['drawdown_pct']:.2f}%",
            "=" * 60,
        ]
        return "\n".join(lines)
