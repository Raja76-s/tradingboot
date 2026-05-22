"""
Multi-Indicator Confidence Scoring Engine.

Aggregates signals from all indicators and patterns to produce
a single confidence score (0-100) with BUY/SELL/HOLD recommendation.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from config.settings import StrategyConfig
from core.indicators import IndicatorEngine, IndicatorResult, Signal
from core.patterns import CandlestickPatterns, PatternResult


@dataclass
class TradeSignal:
    pair: str
    timestamp: datetime
    action: str  # "BUY", "SELL", "HOLD"
    confidence: int  # 0-100
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: float
    indicator_signals: list[IndicatorResult] = field(default_factory=list)
    pattern_signals: list[PatternResult] = field(default_factory=list)
    timeframe: str = "15m"
    summary: str = ""
    buy_count: int = 0
    sell_count: int = 0
    neutral_count: int = 0
    strong_buy_count: int = 0
    strong_sell_count: int = 0
    quality_score: float = 0.0   # final rank: confidence + RR + volume
    stop_loss_pct: float = 0.0   # how far stop is from entry (%)
    grade: str = ""              # A / B / C / SKIP


SIGNAL_SCORES = {
    Signal.STRONG_BUY: 2.0,
    Signal.BUY: 1.0,
    Signal.NEUTRAL: 0.0,
    Signal.SELL: -1.0,
    Signal.STRONG_SELL: -2.0,
}

# High-weight indicators that must agree for a CONFIRMED signal
# If these disagree, confidence is penalised heavily
CORE_INDICATORS = {
    "Golden/Death Cross (50/200)",
    "MACD",
    "RSI",
    "Supertrend",
    "Ichimoku Cloud",
}


class ScoringEngine:
    """Aggregates all indicator and pattern signals into a confidence score."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.indicator_engine = IndicatorEngine(config)
        self.pattern_detector = CandlestickPatterns()

    def analyze(
        self,
        df: pd.DataFrame,
        pair: str,
        timeframe: str = "15m",
    ) -> TradeSignal:
        indicator_results = self.indicator_engine.compute_all(df)
        pattern_results = self.pattern_detector.detect_all(df)

        weighted_score = 0.0
        total_weight = 0.0

        buy_count = 0
        sell_count = 0
        neutral_count = 0
        strong_buy_count = 0
        strong_sell_count = 0

        core_buy = 0
        core_sell = 0
        core_seen = 0

        for result in indicator_results:
            score = SIGNAL_SCORES[result.signal]
            weighted_score += score * result.weight
            total_weight += result.weight

            if result.signal == Signal.STRONG_BUY:
                strong_buy_count += 1
            elif result.signal == Signal.BUY:
                buy_count += 1
            elif result.signal == Signal.SELL:
                sell_count += 1
            elif result.signal == Signal.STRONG_SELL:
                strong_sell_count += 1
            else:
                neutral_count += 1

            if result.name in CORE_INDICATORS:
                core_seen += 1
                if result.signal in (Signal.BUY, Signal.STRONG_BUY):
                    core_buy += 1
                elif result.signal in (Signal.SELL, Signal.STRONG_SELL):
                    core_sell += 1

        for pattern in pattern_results:
            p_score = SIGNAL_SCORES[pattern.signal] * pattern.confidence
            weight = 1.5
            weighted_score += p_score * weight
            total_weight += weight

            if pattern.signal == Signal.STRONG_BUY:
                strong_buy_count += 1
            elif pattern.signal == Signal.BUY:
                buy_count += 1
            elif pattern.signal == Signal.SELL:
                sell_count += 1
            elif pattern.signal == Signal.STRONG_SELL:
                strong_sell_count += 1
            else:
                neutral_count += 1

        if total_weight == 0:
            normalized = 0.0
        else:
            normalized = weighted_score / total_weight

        # Convert -2..+2 range to 0..100 confidence
        confidence = int(min(100, max(0, (normalized + 2) * 25)))

        # --- Core Confluence Filter ---
        # If fewer than 3 of the 5 core indicators agree, cap confidence at 55
        # This prevents weak signals from reaching the BUY/SELL threshold
        if core_seen >= 3:
            dominant_core = max(core_buy, core_sell)
            if dominant_core < 3:
                confidence = min(confidence, 55)  # not enough agreement → HOLD
            elif dominant_core == core_seen:  # all core agree → bonus
                confidence = min(100, confidence + 8)

        # Volume confirmation: if volume spike > 1.5x average, boost by 5
        vol_spike = next(
            (r for r in indicator_results if r.name == "Volume Spike"), None
        )
        if vol_spike and vol_spike.value >= 1.5:
            confidence = min(100, confidence + 5)

        current_price = df["close"].iloc[-1]
        atr_val = self._calculate_atr(df)

        if confidence >= 60:
            action = "BUY"
            # Stop = recent swing low (last 20 candles) or 1.5x ATR, whichever is tighter
            swing_low  = df["low"].rolling(20).min().iloc[-1]
            stop_loss  = max(swing_low, current_price - atr_val * 1.5)
            # Target = next resistance (recent swing high) or 3x ATR minimum
            swing_high = df["high"].rolling(20).max().iloc[-1]
            tp1 = max(current_price + atr_val * 2.0, swing_high * 0.995)
            tp2 = current_price + (current_price - stop_loss) * 3.0  # 1:3 RR
        elif confidence <= 40:
            action = "SELL"
            swing_high = df["high"].rolling(20).max().iloc[-1]
            stop_loss  = min(swing_high, current_price + atr_val * 1.5)
            swing_low  = df["low"].rolling(20).min().iloc[-1]
            tp1 = min(current_price - atr_val * 2.0, swing_low * 1.005)
            tp2 = current_price - (stop_loss - current_price) * 3.0
        else:
            action = "HOLD"
            stop_loss = current_price - atr_val * 1.5
            tp1 = current_price + atr_val * 2.0
            tp2 = current_price + atr_val * 4.0

        risk   = abs(current_price - stop_loss)
        reward = abs(tp1 - current_price)
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

        sl_pct = round((risk / current_price * 100), 2) if current_price > 0 else 0.0

        # Quality score — based on ACTUAL computed values
        # RR bonus: real reward vs real risk
        if rr_ratio >= 3.0:    rr_bonus = 30.0
        elif rr_ratio >= 2.0:  rr_bonus = 20.0
        elif rr_ratio >= 1.5:  rr_bonus = 10.0
        else:                  rr_bonus = 0.0

        # Volume bonus
        vol_bonus = 0.0
        if vol_spike and vol_spike.value >= 2.0:   vol_bonus = 10.0
        elif vol_spike and vol_spike.value >= 1.5: vol_bonus = 5.0

        # Pattern bonus
        pattern_bonus = min(10.0, len(pattern_results) * 5.0)

        # Stop loss penalty — if SL > 3% away, trade is too risky
        sl_penalty = max(0.0, (sl_pct - 3.0) * 3.0)

        quality = round(
            max(0.0, min(100.0,
                (confidence * 0.5) + rr_bonus + vol_bonus + pattern_bonus - sl_penalty
            )), 1
        )

        if quality >= 70 and confidence >= 65:    grade = "A"
        elif quality >= 55 and confidence >= 60:  grade = "B"
        elif quality >= 40 and confidence >= 55:  grade = "C"
        else:                                      grade = "SKIP"

        total_signals = (
            buy_count + sell_count + neutral_count
            + strong_buy_count + strong_sell_count
        )
        summary = self._build_summary(
            action, confidence, buy_count, sell_count, neutral_count,
            strong_buy_count, strong_sell_count, total_signals,
            indicator_results, pattern_results,
        )

        return TradeSignal(
            pair=pair,
            timestamp=datetime.now(),
            action=action,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=round(stop_loss, 2),
            take_profit_1=round(tp1, 2),
            take_profit_2=round(tp2, 2),
            risk_reward_ratio=round(rr_ratio, 2),
            indicator_signals=indicator_results,
            pattern_signals=pattern_results,
            timeframe=timeframe,
            summary=summary,
            buy_count=buy_count + strong_buy_count,
            sell_count=sell_count + strong_sell_count,
            neutral_count=neutral_count,
            strong_buy_count=strong_buy_count,
            strong_sell_count=strong_sell_count,
            quality_score=quality,
            stop_loss_pct=round(sl_pct, 2),
            grade=grade,
        )

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        return float(atr) if not pd.isna(atr) else float(df["close"].iloc[-1] * 0.02)

    def _build_summary(
        self,
        action: str,
        confidence: int,
        buy_count: int,
        sell_count: int,
        neutral_count: int,
        strong_buy_count: int,
        strong_sell_count: int,
        total_signals: int,
        indicators: list[IndicatorResult],
        patterns: list[PatternResult],
    ) -> str:
        lines = [
            f"Action: {action} | Confidence: {confidence}/100",
            f"Signals: {strong_buy_count} Strong Buy, {buy_count} Buy, "
            f"{neutral_count} Neutral, {sell_count} Sell, {strong_sell_count} Strong Sell",
            "",
        ]

        strong_indicators = [
            i for i in indicators
            if i.signal in (Signal.STRONG_BUY, Signal.STRONG_SELL)
        ]
        if strong_indicators:
            lines.append("Key Signals:")
            for ind in strong_indicators:
                lines.append(f"  {ind.name}: {ind.signal.value} - {ind.details}")

        if patterns:
            lines.append("Patterns Detected:")
            for p in patterns:
                lines.append(f"  {p.name}: {p.signal.value} ({p.confidence:.0%}) - {p.description}")

        return "\n".join(lines)


class MultiTimeframeScoringEngine:
    """Analyzes across multiple timeframes for higher accuracy."""

    TIMEFRAME_WEIGHTS = {
        "1m": 0.3,
        "5m": 0.5,
        "15m": 1.0,
        "30m": 1.2,
        "1h": 1.5,
        "4h": 2.0,
        "1d": 2.5,
    }

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.scoring_engine = ScoringEngine(config)

    def analyze_multi_tf(
        self,
        data: dict[str, pd.DataFrame],
        pair: str,
    ) -> TradeSignal:
        """Analyze across multiple timeframes and combine results."""
        signals: list[tuple[TradeSignal, float]] = []

        for tf, df in data.items():
            if len(df) < 60:
                continue
            signal = self.scoring_engine.analyze(df, pair, tf)
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 1.0)
            signals.append((signal, weight))

        if not signals:
            # Return neutral if no data
            return TradeSignal(
                pair=pair,
                timestamp=datetime.now(),
                action="HOLD",
                confidence=50,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit_1=0.0,
                take_profit_2=0.0,
                risk_reward_ratio=0.0,
                summary="Insufficient data for analysis",
            )

        total_weight = sum(w for _, w in signals)
        weighted_confidence = sum(s.confidence * w for s, w in signals) / total_weight

        buy_agreement = sum(1 for s, _ in signals if s.action == "BUY")
        sell_agreement = sum(1 for s, _ in signals if s.action == "SELL")

        # Boost confidence if multiple timeframes agree
        agreement_bonus = 0
        if buy_agreement == len(signals) or sell_agreement == len(signals):
            agreement_bonus = 10
        elif buy_agreement >= len(signals) * 0.7 or sell_agreement >= len(signals) * 0.7:
            agreement_bonus = 5

        final_confidence = min(100, int(weighted_confidence + agreement_bonus))

        # Use the primary timeframe signal as base
        primary_tf = self.config.default_timeframe
        primary_signal = next(
            (s for s, _ in signals if s.timeframe == primary_tf),
            signals[0][0],
        )

        primary_signal.confidence = final_confidence
        if final_confidence >= 60:
            primary_signal.action = "BUY"
        elif final_confidence <= 40:
            primary_signal.action = "SELL"
        else:
            primary_signal.action = "HOLD"

        tf_summary = "\n".join(
            f"  {s.timeframe}: {s.action} ({s.confidence}/100)"
            for s, _ in signals
        )
        primary_signal.summary = (
            f"Multi-Timeframe Analysis:\n{tf_summary}\n\n"
            f"Combined Confidence: {final_confidence}/100\n"
            f"Timeframe Agreement: {max(buy_agreement, sell_agreement)}/{len(signals)}\n\n"
            + primary_signal.summary
        )

        return primary_signal
