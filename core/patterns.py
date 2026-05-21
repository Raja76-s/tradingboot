"""
Candlestick Pattern Recognition Module.

Detects 20+ candlestick patterns:
- Single candle patterns (Doji, Hammer, Shooting Star, etc.)
- Double candle patterns (Engulfing, Harami, etc.)
- Triple candle patterns (Morning/Evening Star, Three Soldiers/Crows, etc.)
"""

from dataclasses import dataclass

import pandas as pd

from core.indicators import Signal


@dataclass
class PatternResult:
    name: str
    signal: Signal
    confidence: float  # 0.0 to 1.0
    description: str


class CandlestickPatterns:
    """Detects candlestick patterns on OHLCV data."""

    @staticmethod
    def _body(row: pd.Series) -> float:
        return abs(row["close"] - row["open"])

    @staticmethod
    def _upper_shadow(row: pd.Series) -> float:
        return row["high"] - max(row["close"], row["open"])

    @staticmethod
    def _lower_shadow(row: pd.Series) -> float:
        return min(row["close"], row["open"]) - row["low"]

    @staticmethod
    def _is_bullish(row: pd.Series) -> bool:
        return row["close"] > row["open"]

    @staticmethod
    def _is_bearish(row: pd.Series) -> bool:
        return row["close"] < row["open"]

    @staticmethod
    def _candle_range(row: pd.Series) -> float:
        return row["high"] - row["low"]

    def detect_all(self, df: pd.DataFrame) -> list[PatternResult]:
        if len(df) < 5:
            return []

        patterns: list[PatternResult] = []

        # Single candle patterns
        patterns.extend(self._detect_doji(df))
        patterns.extend(self._detect_hammer(df))
        patterns.extend(self._detect_inverted_hammer(df))
        patterns.extend(self._detect_shooting_star(df))
        patterns.extend(self._detect_hanging_man(df))
        patterns.extend(self._detect_spinning_top(df))
        patterns.extend(self._detect_marubozu(df))
        patterns.extend(self._detect_dragonfly_doji(df))
        patterns.extend(self._detect_gravestone_doji(df))

        # Double candle patterns
        patterns.extend(self._detect_engulfing(df))
        patterns.extend(self._detect_harami(df))
        patterns.extend(self._detect_piercing_line(df))
        patterns.extend(self._detect_dark_cloud_cover(df))
        patterns.extend(self._detect_tweezer_top_bottom(df))

        # Triple candle patterns
        patterns.extend(self._detect_morning_star(df))
        patterns.extend(self._detect_evening_star(df))
        patterns.extend(self._detect_three_white_soldiers(df))
        patterns.extend(self._detect_three_black_crows(df))
        patterns.extend(self._detect_three_inside(df))
        patterns.extend(self._detect_three_outside(df))

        # Chart patterns
        patterns.extend(self._detect_double_top(df))
        patterns.extend(self._detect_double_bottom(df))
        patterns.extend(self._detect_head_and_shoulders(df))

        return patterns

    def _detect_doji(self, df: pd.DataFrame) -> list[PatternResult]:
        row = df.iloc[-1]
        body = self._body(row)
        cr = self._candle_range(row)

        if cr == 0:
            return []

        if body / cr < 0.1:
            return [PatternResult(
                name="Doji",
                signal=Signal.NEUTRAL,
                confidence=0.6,
                description="Doji - indecision, potential reversal",
            )]
        return []

    def _detect_dragonfly_doji(self, df: pd.DataFrame) -> list[PatternResult]:
        row = df.iloc[-1]
        body = self._body(row)
        cr = self._candle_range(row)
        lower = self._lower_shadow(row)
        upper = self._upper_shadow(row)

        if cr == 0:
            return []

        if body / cr < 0.1 and lower / cr > 0.6 and upper / cr < 0.1:
            return [PatternResult(
                name="Dragonfly Doji",
                signal=Signal.STRONG_BUY,
                confidence=0.7,
                description="Dragonfly Doji - strong bullish reversal signal",
            )]
        return []

    def _detect_gravestone_doji(self, df: pd.DataFrame) -> list[PatternResult]:
        row = df.iloc[-1]
        body = self._body(row)
        cr = self._candle_range(row)
        lower = self._lower_shadow(row)
        upper = self._upper_shadow(row)

        if cr == 0:
            return []

        if body / cr < 0.1 and upper / cr > 0.6 and lower / cr < 0.1:
            return [PatternResult(
                name="Gravestone Doji",
                signal=Signal.STRONG_SELL,
                confidence=0.7,
                description="Gravestone Doji - strong bearish reversal signal",
            )]
        return []

    def _detect_hammer(self, df: pd.DataFrame) -> list[PatternResult]:
        row = df.iloc[-1]
        df.iloc[-2]
        body = self._body(row)
        lower = self._lower_shadow(row)
        upper = self._upper_shadow(row)

        if body == 0:
            return []

        is_downtrend = df["close"].iloc[-5:].iloc[0] > df["close"].iloc[-5:].iloc[-2]

        if lower > 2 * body and upper < body * 0.3 and is_downtrend:
            return [PatternResult(
                name="Hammer",
                signal=Signal.STRONG_BUY,
                confidence=0.75,
                description="Hammer - bullish reversal after downtrend",
            )]
        return []

    def _detect_inverted_hammer(self, df: pd.DataFrame) -> list[PatternResult]:
        row = df.iloc[-1]
        body = self._body(row)
        lower = self._lower_shadow(row)
        upper = self._upper_shadow(row)

        if body == 0:
            return []

        is_downtrend = df["close"].iloc[-5:].iloc[0] > df["close"].iloc[-5:].iloc[-2]

        if upper > 2 * body and lower < body * 0.3 and is_downtrend:
            return [PatternResult(
                name="Inverted Hammer",
                signal=Signal.BUY,
                confidence=0.65,
                description="Inverted Hammer - potential bullish reversal",
            )]
        return []

    def _detect_shooting_star(self, df: pd.DataFrame) -> list[PatternResult]:
        row = df.iloc[-1]
        body = self._body(row)
        lower = self._lower_shadow(row)
        upper = self._upper_shadow(row)

        if body == 0:
            return []

        is_uptrend = df["close"].iloc[-5:].iloc[0] < df["close"].iloc[-5:].iloc[-2]

        if upper > 2 * body and lower < body * 0.3 and is_uptrend:
            return [PatternResult(
                name="Shooting Star",
                signal=Signal.STRONG_SELL,
                confidence=0.75,
                description="Shooting Star - bearish reversal after uptrend",
            )]
        return []

    def _detect_hanging_man(self, df: pd.DataFrame) -> list[PatternResult]:
        row = df.iloc[-1]
        body = self._body(row)
        lower = self._lower_shadow(row)
        upper = self._upper_shadow(row)

        if body == 0:
            return []

        is_uptrend = df["close"].iloc[-5:].iloc[0] < df["close"].iloc[-5:].iloc[-2]

        if lower > 2 * body and upper < body * 0.3 and is_uptrend:
            return [PatternResult(
                name="Hanging Man",
                signal=Signal.SELL,
                confidence=0.65,
                description="Hanging Man - bearish reversal warning",
            )]
        return []

    def _detect_spinning_top(self, df: pd.DataFrame) -> list[PatternResult]:
        row = df.iloc[-1]
        body = self._body(row)
        cr = self._candle_range(row)
        lower = self._lower_shadow(row)
        upper = self._upper_shadow(row)

        if cr == 0 or body == 0:
            return []

        if body / cr < 0.3 and upper > body and lower > body:
            return [PatternResult(
                name="Spinning Top",
                signal=Signal.NEUTRAL,
                confidence=0.5,
                description="Spinning Top - market indecision",
            )]
        return []

    def _detect_marubozu(self, df: pd.DataFrame) -> list[PatternResult]:
        row = df.iloc[-1]
        body = self._body(row)
        cr = self._candle_range(row)

        if cr == 0:
            return []

        if body / cr > 0.95:
            if self._is_bullish(row):
                return [PatternResult(
                    name="Bullish Marubozu",
                    signal=Signal.STRONG_BUY,
                    confidence=0.8,
                    description="Bullish Marubozu - strong buying pressure",
                )]
            else:
                return [PatternResult(
                    name="Bearish Marubozu",
                    signal=Signal.STRONG_SELL,
                    confidence=0.8,
                    description="Bearish Marubozu - strong selling pressure",
                )]
        return []

    def _detect_engulfing(self, df: pd.DataFrame) -> list[PatternResult]:
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        if self._is_bearish(prev) and self._is_bullish(curr):
            if curr["open"] <= prev["close"] and curr["close"] >= prev["open"]:
                return [PatternResult(
                    name="Bullish Engulfing",
                    signal=Signal.STRONG_BUY,
                    confidence=0.8,
                    description="Bullish Engulfing - strong reversal signal",
                )]

        if self._is_bullish(prev) and self._is_bearish(curr):
            if curr["open"] >= prev["close"] and curr["close"] <= prev["open"]:
                return [PatternResult(
                    name="Bearish Engulfing",
                    signal=Signal.STRONG_SELL,
                    confidence=0.8,
                    description="Bearish Engulfing - strong reversal signal",
                )]
        return []

    def _detect_harami(self, df: pd.DataFrame) -> list[PatternResult]:
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        if self._is_bearish(prev) and self._is_bullish(curr):
            if curr["open"] >= prev["close"] and curr["close"] <= prev["open"]:
                if self._body(curr) < self._body(prev) * 0.5:
                    return [PatternResult(
                        name="Bullish Harami",
                        signal=Signal.BUY,
                        confidence=0.65,
                        description="Bullish Harami - potential bullish reversal",
                    )]

        if self._is_bullish(prev) and self._is_bearish(curr):
            if curr["open"] <= prev["close"] and curr["close"] >= prev["open"]:
                if self._body(curr) < self._body(prev) * 0.5:
                    return [PatternResult(
                        name="Bearish Harami",
                        signal=Signal.SELL,
                        confidence=0.65,
                        description="Bearish Harami - potential bearish reversal",
                    )]
        return []

    def _detect_piercing_line(self, df: pd.DataFrame) -> list[PatternResult]:
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        if self._is_bearish(prev) and self._is_bullish(curr):
            midpoint = (prev["open"] + prev["close"]) / 2
            if curr["open"] < prev["close"] and curr["close"] > midpoint:
                return [PatternResult(
                    name="Piercing Line",
                    signal=Signal.BUY,
                    confidence=0.7,
                    description="Piercing Line - bullish reversal",
                )]
        return []

    def _detect_dark_cloud_cover(self, df: pd.DataFrame) -> list[PatternResult]:
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        if self._is_bullish(prev) and self._is_bearish(curr):
            midpoint = (prev["open"] + prev["close"]) / 2
            if curr["open"] > prev["close"] and curr["close"] < midpoint:
                return [PatternResult(
                    name="Dark Cloud Cover",
                    signal=Signal.SELL,
                    confidence=0.7,
                    description="Dark Cloud Cover - bearish reversal",
                )]
        return []

    def _detect_tweezer_top_bottom(self, df: pd.DataFrame) -> list[PatternResult]:
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        results: list[PatternResult] = []

        tolerance = self._candle_range(curr) * 0.05 if self._candle_range(curr) > 0 else 0.01

        # Tweezer Top
        if abs(curr["high"] - prev["high"]) < tolerance:
            if self._is_bullish(prev) and self._is_bearish(curr):
                results.append(PatternResult(
                    name="Tweezer Top",
                    signal=Signal.SELL,
                    confidence=0.65,
                    description="Tweezer Top - bearish reversal",
                ))

        # Tweezer Bottom
        if abs(curr["low"] - prev["low"]) < tolerance:
            if self._is_bearish(prev) and self._is_bullish(curr):
                results.append(PatternResult(
                    name="Tweezer Bottom",
                    signal=Signal.BUY,
                    confidence=0.65,
                    description="Tweezer Bottom - bullish reversal",
                ))

        return results

    def _detect_morning_star(self, df: pd.DataFrame) -> list[PatternResult]:
        if len(df) < 3:
            return []

        first = df.iloc[-3]
        second = df.iloc[-2]
        third = df.iloc[-1]

        if (
            self._is_bearish(first)
            and self._body(second) < self._body(first) * 0.3
            and self._is_bullish(third)
            and third["close"] > (first["open"] + first["close"]) / 2
        ):
            return [PatternResult(
                name="Morning Star",
                signal=Signal.STRONG_BUY,
                confidence=0.85,
                description="Morning Star - strong bullish reversal (3-candle pattern)",
            )]
        return []

    def _detect_evening_star(self, df: pd.DataFrame) -> list[PatternResult]:
        if len(df) < 3:
            return []

        first = df.iloc[-3]
        second = df.iloc[-2]
        third = df.iloc[-1]

        if (
            self._is_bullish(first)
            and self._body(second) < self._body(first) * 0.3
            and self._is_bearish(third)
            and third["close"] < (first["open"] + first["close"]) / 2
        ):
            return [PatternResult(
                name="Evening Star",
                signal=Signal.STRONG_SELL,
                confidence=0.85,
                description="Evening Star - strong bearish reversal (3-candle pattern)",
            )]
        return []

    def _detect_three_white_soldiers(self, df: pd.DataFrame) -> list[PatternResult]:
        if len(df) < 3:
            return []

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        if (
            self._is_bullish(c1) and self._is_bullish(c2) and self._is_bullish(c3)
            and c2["close"] > c1["close"]
            and c3["close"] > c2["close"]
            and c2["open"] > c1["open"]
            and c3["open"] > c2["open"]
        ):
            cr1 = self._candle_range(c1)
            cr2 = self._candle_range(c2)
            cr3 = self._candle_range(c3)
            if cr1 > 0 and cr2 > 0 and cr3 > 0:
                if (
                    self._body(c1) / cr1 > 0.5
                    and self._body(c2) / cr2 > 0.5
                    and self._body(c3) / cr3 > 0.5
                ):
                    return [PatternResult(
                        name="Three White Soldiers",
                        signal=Signal.STRONG_BUY,
                        confidence=0.85,
                        description="Three White Soldiers - very strong bullish signal",
                    )]
        return []

    def _detect_three_black_crows(self, df: pd.DataFrame) -> list[PatternResult]:
        if len(df) < 3:
            return []

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        if (
            self._is_bearish(c1) and self._is_bearish(c2) and self._is_bearish(c3)
            and c2["close"] < c1["close"]
            and c3["close"] < c2["close"]
            and c2["open"] < c1["open"]
            and c3["open"] < c2["open"]
        ):
            cr1 = self._candle_range(c1)
            cr2 = self._candle_range(c2)
            cr3 = self._candle_range(c3)
            if cr1 > 0 and cr2 > 0 and cr3 > 0:
                if (
                    self._body(c1) / cr1 > 0.5
                    and self._body(c2) / cr2 > 0.5
                    and self._body(c3) / cr3 > 0.5
                ):
                    return [PatternResult(
                        name="Three Black Crows",
                        signal=Signal.STRONG_SELL,
                        confidence=0.85,
                        description="Three Black Crows - very strong bearish signal",
                    )]
        return []

    def _detect_three_inside(self, df: pd.DataFrame) -> list[PatternResult]:
        if len(df) < 3:
            return []

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        # Three Inside Up
        if (
            self._is_bearish(c1)
            and self._is_bullish(c2)
            and c2["close"] < c1["open"] and c2["open"] > c1["close"]
            and self._is_bullish(c3)
            and c3["close"] > c1["open"]
        ):
            return [PatternResult(
                name="Three Inside Up",
                signal=Signal.STRONG_BUY,
                confidence=0.75,
                description="Three Inside Up - bullish reversal confirmation",
            )]

        # Three Inside Down
        if (
            self._is_bullish(c1)
            and self._is_bearish(c2)
            and c2["close"] > c1["open"] and c2["open"] < c1["close"]
            and self._is_bearish(c3)
            and c3["close"] < c1["open"]
        ):
            return [PatternResult(
                name="Three Inside Down",
                signal=Signal.STRONG_SELL,
                confidence=0.75,
                description="Three Inside Down - bearish reversal confirmation",
            )]
        return []

    def _detect_three_outside(self, df: pd.DataFrame) -> list[PatternResult]:
        if len(df) < 3:
            return []

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        # Three Outside Up
        if (
            self._is_bearish(c1)
            and self._is_bullish(c2)
            and c2["open"] <= c1["close"] and c2["close"] >= c1["open"]
            and self._is_bullish(c3)
            and c3["close"] > c2["close"]
        ):
            return [PatternResult(
                name="Three Outside Up",
                signal=Signal.STRONG_BUY,
                confidence=0.75,
                description="Three Outside Up - bullish reversal confirmation",
            )]

        # Three Outside Down
        if (
            self._is_bullish(c1)
            and self._is_bearish(c2)
            and c2["open"] >= c1["close"] and c2["close"] <= c1["open"]
            and self._is_bearish(c3)
            and c3["close"] < c2["close"]
        ):
            return [PatternResult(
                name="Three Outside Down",
                signal=Signal.STRONG_SELL,
                confidence=0.75,
                description="Three Outside Down - bearish reversal confirmation",
            )]
        return []

    def _detect_double_top(self, df: pd.DataFrame) -> list[PatternResult]:
        if len(df) < 30:
            return []

        highs = df["high"].iloc[-30:]
        peaks = []
        for i in range(2, len(highs) - 2):
            if highs.iloc[i] > highs.iloc[i - 1] and highs.iloc[i] > highs.iloc[i - 2]:
                if highs.iloc[i] > highs.iloc[i + 1] and highs.iloc[i] > highs.iloc[i + 2]:
                    peaks.append((i, highs.iloc[i]))

        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            tolerance = p1[1] * 0.02
            if abs(p1[1] - p2[1]) < tolerance and p2[0] - p1[0] >= 5:
                return [PatternResult(
                    name="Double Top",
                    signal=Signal.STRONG_SELL,
                    confidence=0.8,
                    description="Double Top - major bearish reversal pattern",
                )]
        return []

    def _detect_double_bottom(self, df: pd.DataFrame) -> list[PatternResult]:
        if len(df) < 30:
            return []

        lows = df["low"].iloc[-30:]
        troughs = []
        for i in range(2, len(lows) - 2):
            if lows.iloc[i] < lows.iloc[i - 1] and lows.iloc[i] < lows.iloc[i - 2]:
                if lows.iloc[i] < lows.iloc[i + 1] and lows.iloc[i] < lows.iloc[i + 2]:
                    troughs.append((i, lows.iloc[i]))

        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            tolerance = t1[1] * 0.02
            if abs(t1[1] - t2[1]) < tolerance and t2[0] - t1[0] >= 5:
                return [PatternResult(
                    name="Double Bottom",
                    signal=Signal.STRONG_BUY,
                    confidence=0.8,
                    description="Double Bottom - major bullish reversal pattern",
                )]
        return []

    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> list[PatternResult]:
        if len(df) < 40:
            return []

        highs = df["high"].iloc[-40:]
        peaks = []
        for i in range(2, len(highs) - 2):
            if highs.iloc[i] > highs.iloc[i - 1] and highs.iloc[i] > highs.iloc[i - 2]:
                if highs.iloc[i] > highs.iloc[i + 1] and highs.iloc[i] > highs.iloc[i + 2]:
                    peaks.append((i, highs.iloc[i]))

        if len(peaks) >= 3:
            left, head, right = peaks[-3], peaks[-2], peaks[-1]
            tolerance = head[1] * 0.03

            if (
                head[1] > left[1]
                and head[1] > right[1]
                and abs(left[1] - right[1]) < tolerance
            ):
                return [PatternResult(
                    name="Head and Shoulders",
                    signal=Signal.STRONG_SELL,
                    confidence=0.85,
                    description="Head and Shoulders - major bearish reversal pattern",
                )]

        # Inverse Head and Shoulders
        lows = df["low"].iloc[-40:]
        troughs = []
        for i in range(2, len(lows) - 2):
            if lows.iloc[i] < lows.iloc[i - 1] and lows.iloc[i] < lows.iloc[i - 2]:
                if lows.iloc[i] < lows.iloc[i + 1] and lows.iloc[i] < lows.iloc[i + 2]:
                    troughs.append((i, lows.iloc[i]))

        if len(troughs) >= 3:
            left, head, right = troughs[-3], troughs[-2], troughs[-1]
            tolerance = head[1] * 0.03

            if (
                head[1] < left[1]
                and head[1] < right[1]
                and abs(left[1] - right[1]) < tolerance
            ):
                return [PatternResult(
                    name="Inverse Head and Shoulders",
                    signal=Signal.STRONG_BUY,
                    confidence=0.85,
                    description="Inverse Head & Shoulders - major bullish reversal pattern",
                )]

        return []
