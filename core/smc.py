"""
Smart Money Concepts (SMC) — Institutional footprint detection.

Detects:
1. Break of Structure (BOS)     — trend confirmation
2. Change of Character (CHoCH)  — early reversal signal
3. Order Blocks (OB)            — institutional entry zones
4. Fair Value Gaps (FVG)        — imbalance zones price returns to
5. Liquidity Sweeps             — stop hunt detection
6. Equal Highs/Lows             — liquidity pools
"""

from dataclasses import dataclass

import pandas as pd
import numpy as np

from core.indicators import Signal, IndicatorResult


@dataclass
class SMCResult:
    name: str
    signal: Signal
    value: float
    weight: float
    details: str
    level: float = 0.0  # price level (for OB, FVG etc)


class SmartMoneyConcepts:

    def analyze(self, df: pd.DataFrame) -> list[IndicatorResult]:
        if len(df) < 50:
            return []

        results: list[IndicatorResult] = []
        results.extend(self._break_of_structure(df))
        results.extend(self._order_blocks(df))
        results.extend(self._fair_value_gap(df))
        results.extend(self._liquidity_sweep(df))
        results.extend(self._equal_highs_lows(df))
        return results

    # ── 1. Break of Structure + Change of Character ───────────────────────

    def _break_of_structure(self, df: pd.DataFrame) -> list[IndicatorResult]:
        """
        BOS  = price breaks previous swing high/low in SAME direction → trend continues
        CHoCH = price breaks previous swing high/low in OPPOSITE direction → reversal
        """
        results = []
        highs = df["high"]
        lows = df["low"]
        close = df["close"]

        # Find last 3 swing highs and lows
        swing_highs = []
        swing_lows = []
        for i in range(2, len(df) - 2):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] \
               and highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
                swing_highs.append((i, highs.iloc[i]))
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] \
               and lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]:
                swing_lows.append((i, lows.iloc[i]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return results

        last_high_idx, last_high = swing_highs[-1]
        prev_high_idx, prev_high = swing_highs[-2]
        last_low_idx, last_low = swing_lows[-1]
        prev_low_idx, prev_low = swing_lows[-2]

        current_close = close.iloc[-1]

        # Bullish BOS — current close breaks above previous swing high
        # AND previous structure was making higher lows (uptrend)
        if current_close > prev_high and last_low > prev_low:
            results.append(IndicatorResult(
                name="SMC: Bullish BOS",
                signal=Signal.STRONG_BUY,
                value=current_close - prev_high,
                weight=2.5,
                details=f"Broke above {prev_high:.2f} — uptrend confirmed, higher highs+lows",
            ))

        # Bearish BOS — current close breaks below previous swing low
        # AND previous structure was making lower highs (downtrend)
        elif current_close < prev_low and last_high < prev_high:
            results.append(IndicatorResult(
                name="SMC: Bearish BOS",
                signal=Signal.STRONG_SELL,
                value=current_close - prev_low,
                weight=2.5,
                details=f"Broke below {prev_low:.2f} — downtrend confirmed, lower highs+lows",
            ))

        # Bullish CHoCH — was in downtrend but now broke above last swing high
        elif current_close > last_high and last_high < prev_high:
            results.append(IndicatorResult(
                name="SMC: Bullish CHoCH",
                signal=Signal.BUY,
                value=current_close - last_high,
                weight=2.0,
                details=f"Character changed — broke {last_high:.2f}, possible reversal UP",
            ))

        # Bearish CHoCH — was in uptrend but now broke below last swing low
        elif current_close < last_low and last_low > prev_low:
            results.append(IndicatorResult(
                name="SMC: Bearish CHoCH",
                signal=Signal.SELL,
                value=current_close - last_low,
                weight=2.0,
                details=f"Character changed — broke {last_low:.2f}, possible reversal DOWN",
            ))

        return results

    # ── 2. Order Blocks ───────────────────────────────────────────────────

    def _order_blocks(self, df: pd.DataFrame) -> list[IndicatorResult]:
        """
        Bullish OB  = last bearish candle before a strong bullish move
        Bearish OB  = last bullish candle before a strong bearish move
        Price returning to OB = high probability entry zone
        """
        results = []
        close = df["close"].iloc[-1]

        # Look back 30 candles for order blocks
        lookback = min(30, len(df) - 5)

        for i in range(len(df) - lookback, len(df) - 3):
            candle = df.iloc[i]
            body = abs(candle["close"] - candle["open"])
            if body == 0:
                continue

            # Check if followed by strong move (3 candles)
            next_3 = df.iloc[i+1:i+4]
            move = next_3["close"].iloc[-1] - candle["close"]
            avg_body = df["close"].diff().abs().rolling(20).mean().iloc[i]

            if avg_body == 0:
                continue

            # Bullish OB — bearish candle followed by strong up move
            if (candle["close"] < candle["open"]  # bearish candle
                    and move > avg_body * 2        # strong up move after
                    and candle["low"] <= close <= candle["high"]):  # price back in OB
                results.append(IndicatorResult(
                    name="SMC: Bullish Order Block",
                    signal=Signal.STRONG_BUY,
                    value=close - candle["low"],
                    weight=2.5,
                    details=f"Price in Bullish OB zone {candle['low']:.2f}-{candle['high']:.2f} — institutional buy zone",
                ))
                break

            # Bearish OB — bullish candle followed by strong down move
            if (candle["close"] > candle["open"]  # bullish candle
                    and move < -avg_body * 2       # strong down move after
                    and candle["low"] <= close <= candle["high"]):  # price back in OB
                results.append(IndicatorResult(
                    name="SMC: Bearish Order Block",
                    signal=Signal.STRONG_SELL,
                    value=close - candle["high"],
                    weight=2.5,
                    details=f"Price in Bearish OB zone {candle['low']:.2f}-{candle['high']:.2f} — institutional sell zone",
                ))
                break

        return results

    # ── 3. Fair Value Gap ─────────────────────────────────────────────────

    def _fair_value_gap(self, df: pd.DataFrame) -> list[IndicatorResult]:
        """
        FVG (Imbalance) = 3-candle pattern where candle 1 high < candle 3 low (bullish)
                          or candle 1 low > candle 3 high (bearish)
        Price tends to return to fill the gap.
        """
        results = []
        close = df["close"].iloc[-1]

        lookback = min(20, len(df) - 3)

        for i in range(len(df) - lookback, len(df) - 2):
            c1 = df.iloc[i]
            c3 = df.iloc[i + 2]

            # Bullish FVG — gap between c1 high and c3 low
            if c1["high"] < c3["low"]:
                gap_low = c1["high"]
                gap_high = c3["low"]
                # Price returning to fill bullish FVG = buy opportunity
                if gap_low <= close <= gap_high:
                    results.append(IndicatorResult(
                        name="SMC: Bullish FVG",
                        signal=Signal.BUY,
                        value=close - gap_low,
                        weight=1.8,
                        details=f"Price filling Bullish FVG {gap_low:.2f}-{gap_high:.2f} — expect bounce up",
                    ))
                    break

            # Bearish FVG — gap between c1 low and c3 high
            if c1["low"] > c3["high"]:
                gap_high = c1["low"]
                gap_low = c3["high"]
                # Price returning to fill bearish FVG = sell opportunity
                if gap_low <= close <= gap_high:
                    results.append(IndicatorResult(
                        name="SMC: Bearish FVG",
                        signal=Signal.SELL,
                        value=close - gap_high,
                        weight=1.8,
                        details=f"Price filling Bearish FVG {gap_low:.2f}-{gap_high:.2f} — expect drop",
                    ))
                    break

        return results

    # ── 4. Liquidity Sweep ────────────────────────────────────────────────

    def _liquidity_sweep(self, df: pd.DataFrame) -> list[IndicatorResult]:
        """
        Liquidity Sweep = price briefly goes below recent lows (stop hunt)
        then quickly reverses — smart money grabbed retail stop losses.
        This is a HIGH probability reversal signal.
        """
        results = []

        if len(df) < 10:
            return results

        recent_low = df["low"].iloc[-20:-2].min()
        recent_high = df["high"].iloc[-20:-2].max()

        prev_candle = df.iloc[-2]
        curr_candle = df.iloc[-1]

        # Bullish sweep — prev candle wick went below recent low but closed above
        # Current candle confirms with bullish close
        if (prev_candle["low"] < recent_low
                and prev_candle["close"] > recent_low
                and curr_candle["close"] > prev_candle["close"]):
            results.append(IndicatorResult(
                name="SMC: Bullish Liquidity Sweep",
                signal=Signal.STRONG_BUY,
                value=recent_low - prev_candle["low"],
                weight=3.0,  # highest weight — very reliable
                details=f"Stop hunt below {recent_low:.2f} — smart money accumulated, reversal UP likely",
            ))

        # Bearish sweep — prev candle wick went above recent high but closed below
        elif (prev_candle["high"] > recent_high
                and prev_candle["close"] < recent_high
                and curr_candle["close"] < prev_candle["close"]):
            results.append(IndicatorResult(
                name="SMC: Bearish Liquidity Sweep",
                signal=Signal.STRONG_SELL,
                value=prev_candle["high"] - recent_high,
                weight=3.0,
                details=f"Stop hunt above {recent_high:.2f} — smart money distributed, reversal DOWN likely",
            ))

        return results

    # ── 5. Equal Highs / Equal Lows (Liquidity Pools) ────────────────────

    def _equal_highs_lows(self, df: pd.DataFrame) -> list[IndicatorResult]:
        """
        Equal Highs/Lows = price tested same level 2+ times without breaking
        = liquidity pool sitting there (retail stop losses clustered)
        = smart money will sweep it before reversing
        """
        results = []
        close = df["close"].iloc[-1]
        tolerance = close * 0.002  # 0.2% tolerance

        highs = df["high"].iloc[-30:]
        lows = df["low"].iloc[-30:]

        # Equal highs — resistance with liquidity above
        high_counts = {}
        for h in highs:
            key = round(h / tolerance) * tolerance
            high_counts[key] = high_counts.get(key, 0) + 1

        for level, count in high_counts.items():
            if count >= 2 and abs(close - level) / close < 0.005:
                results.append(IndicatorResult(
                    name="SMC: Equal Highs (Liquidity)",
                    signal=Signal.SELL,
                    value=level - close,
                    weight=1.5,
                    details=f"Liquidity pool at {level:.2f} — smart money may sweep then reverse DOWN",
                ))
                break

        # Equal lows — support with liquidity below
        low_counts = {}
        for l in lows:
            key = round(l / tolerance) * tolerance
            low_counts[key] = low_counts.get(key, 0) + 1

        for level, count in low_counts.items():
            if count >= 2 and abs(close - level) / close < 0.005:
                results.append(IndicatorResult(
                    name="SMC: Equal Lows (Liquidity)",
                    signal=Signal.BUY,
                    value=close - level,
                    weight=1.5,
                    details=f"Liquidity pool at {level:.2f} — smart money may sweep then reverse UP",
                ))
                break

        return results
