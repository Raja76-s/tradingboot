"""
Candlestick Pattern Recognition — correct math, strict conditions.
Only fires when pattern is clearly visible on chart.
"""

from dataclasses import dataclass

import pandas as pd

from core.indicators import Signal


@dataclass
class PatternResult:
    name: str
    signal: Signal
    confidence: float
    description: str


class CandlestickPatterns:

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _body(r: pd.Series) -> float:
        return abs(r["close"] - r["open"])

    @staticmethod
    def _upper_wick(r: pd.Series) -> float:
        return r["high"] - max(r["close"], r["open"])

    @staticmethod
    def _lower_wick(r: pd.Series) -> float:
        return min(r["close"], r["open"]) - r["low"]

    @staticmethod
    def _range(r: pd.Series) -> float:
        return r["high"] - r["low"]

    @staticmethod
    def _bull(r: pd.Series) -> bool:
        return r["close"] > r["open"]

    @staticmethod
    def _bear(r: pd.Series) -> bool:
        return r["close"] < r["open"]

    def _trend(self, df: pd.DataFrame, n: int = 5) -> str:
        """Returns 'up', 'down', or 'flat' based on last n closes."""
        closes = df["close"].iloc[-n-1:-1]
        if closes.iloc[-1] > closes.iloc[0] * 1.002:
            return "up"
        if closes.iloc[-1] < closes.iloc[0] * 0.998:
            return "down"
        return "flat"

    # ── main entry ───────────────────────────────────────────────────────────

    def detect_all(self, df: pd.DataFrame) -> list[PatternResult]:
        if len(df) < 10:
            return []
        results: list[PatternResult] = []
        results.extend(self._single(df))
        results.extend(self._double(df))
        results.extend(self._triple(df))
        results.extend(self._chart(df))
        return results

    # ── single candle ────────────────────────────────────────────────────────

    def _single(self, df: pd.DataFrame) -> list[PatternResult]:
        results = []
        r = df.iloc[-1]
        rng = self._range(r)
        if rng == 0:
            return []
        body = self._body(r)
        uw = self._upper_wick(r)
        lw = self._lower_wick(r)
        trend = self._trend(df)

        # Doji — body < 5% of range
        if body / rng < 0.05:
            results.append(PatternResult("Doji", Signal.NEUTRAL, 0.6,
                "Doji — indecision, watch for breakout direction"))

        # Dragonfly Doji — tiny body at top, long lower wick
        elif body / rng < 0.1 and lw / rng > 0.65 and uw / rng < 0.1 and trend == "down":
            results.append(PatternResult("Dragonfly Doji", Signal.STRONG_BUY, 0.75,
                "Dragonfly Doji — bulls rejected lower prices, bullish reversal"))

        # Gravestone Doji — tiny body at bottom, long upper wick
        elif body / rng < 0.1 and uw / rng > 0.65 and lw / rng < 0.1 and trend == "up":
            results.append(PatternResult("Gravestone Doji", Signal.STRONG_SELL, 0.75,
                "Gravestone Doji — bears rejected higher prices, bearish reversal"))

        # Hammer — small body at top, lower wick >= 2x body, tiny upper wick, in downtrend
        if body > 0 and lw >= 2 * body and uw <= 0.3 * body and trend == "down":
            results.append(PatternResult("Hammer", Signal.STRONG_BUY, 0.78,
                "Hammer — strong bullish reversal after downtrend"))

        # Inverted Hammer — small body at bottom, upper wick >= 2x body, in downtrend
        if body > 0 and uw >= 2 * body and lw <= 0.3 * body and trend == "down":
            results.append(PatternResult("Inverted Hammer", Signal.BUY, 0.65,
                "Inverted Hammer — potential bullish reversal, needs confirmation"))

        # Shooting Star — small body at bottom, upper wick >= 2x body, in uptrend
        if body > 0 and uw >= 2 * body and lw <= 0.3 * body and trend == "up":
            results.append(PatternResult("Shooting Star", Signal.STRONG_SELL, 0.78,
                "Shooting Star — bearish reversal after uptrend"))

        # Hanging Man — small body at top, lower wick >= 2x body, in uptrend
        if body > 0 and lw >= 2 * body and uw <= 0.3 * body and trend == "up":
            results.append(PatternResult("Hanging Man", Signal.SELL, 0.65,
                "Hanging Man — bearish warning after uptrend"))

        # Marubozu — body >= 95% of range (no wicks)
        if body / rng > 0.95:
            if self._bull(r):
                results.append(PatternResult("Bullish Marubozu", Signal.STRONG_BUY, 0.82,
                    "Bullish Marubozu — pure buying pressure, no rejection"))
            else:
                results.append(PatternResult("Bearish Marubozu", Signal.STRONG_SELL, 0.82,
                    "Bearish Marubozu — pure selling pressure, no rejection"))

        # Spinning Top — small body, both wicks larger than body
        if 0.05 < body / rng < 0.3 and uw > body and lw > body:
            results.append(PatternResult("Spinning Top", Signal.NEUTRAL, 0.5,
                "Spinning Top — indecision, neither bulls nor bears in control"))

        return results

    # ── double candle ────────────────────────────────────────────────────────

    def _double(self, df: pd.DataFrame) -> list[PatternResult]:
        results = []
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # ── Bullish Engulfing ──
        # prev = RED candle, curr = GREEN candle
        # curr opens BELOW prev close AND closes ABOVE prev open
        # curr body must be LARGER than prev body
        if (self._bear(prev) and self._bull(curr)
                and curr["open"] < prev["close"]
                and curr["close"] > prev["open"]
                and self._body(curr) > self._body(prev)):
            results.append(PatternResult("Bullish Engulfing", Signal.STRONG_BUY, 0.82,
                "Bullish Engulfing — green candle fully covers red, strong reversal"))

        # ── Bearish Engulfing ──
        # prev = GREEN candle, curr = RED candle
        # curr opens ABOVE prev close AND closes BELOW prev open
        elif (self._bull(prev) and self._bear(curr)
                and curr["open"] > prev["close"]
                and curr["close"] < prev["open"]
                and self._body(curr) > self._body(prev)):
            results.append(PatternResult("Bearish Engulfing", Signal.STRONG_SELL, 0.82,
                "Bearish Engulfing — red candle fully covers green, strong reversal"))

        # ── Bullish Harami ──
        # prev = large RED, curr = small GREEN inside prev body
        if (self._bear(prev) and self._bull(curr)
                and curr["open"] > prev["close"]
                and curr["close"] < prev["open"]
                and self._body(curr) < self._body(prev) * 0.5):
            results.append(PatternResult("Bullish Harami", Signal.BUY, 0.65,
                "Bullish Harami — small green inside large red, potential reversal"))

        # ── Bearish Harami ──
        if (self._bull(prev) and self._bear(curr)
                and curr["open"] < prev["close"]
                and curr["close"] > prev["open"]
                and self._body(curr) < self._body(prev) * 0.5):
            results.append(PatternResult("Bearish Harami", Signal.SELL, 0.65,
                "Bearish Harami — small red inside large green, potential reversal"))

        # ── Piercing Line ──
        # prev = RED, curr = GREEN opens below prev low, closes above prev midpoint
        if (self._bear(prev) and self._bull(curr)):
            midpoint = (prev["open"] + prev["close"]) / 2
            if curr["open"] < prev["close"] and curr["close"] > midpoint:
                results.append(PatternResult("Piercing Line", Signal.BUY, 0.70,
                    "Piercing Line — bulls pushed above midpoint of bearish candle"))

        # ── Dark Cloud Cover ──
        if (self._bull(prev) and self._bear(curr)):
            midpoint = (prev["open"] + prev["close"]) / 2
            if curr["open"] > prev["close"] and curr["close"] < midpoint:
                results.append(PatternResult("Dark Cloud Cover", Signal.SELL, 0.70,
                    "Dark Cloud Cover — bears pushed below midpoint of bullish candle"))

        # ── Tweezer Top ──
        tol = self._range(curr) * 0.003 if self._range(curr) > 0 else 0.5
        if (abs(curr["high"] - prev["high"]) < tol
                and self._bull(prev) and self._bear(curr)):
            results.append(PatternResult("Tweezer Top", Signal.SELL, 0.65,
                "Tweezer Top — equal highs, resistance confirmed"))

        # ── Tweezer Bottom ──
        if (abs(curr["low"] - prev["low"]) < tol
                and self._bear(prev) and self._bull(curr)):
            results.append(PatternResult("Tweezer Bottom", Signal.BUY, 0.65,
                "Tweezer Bottom — equal lows, support confirmed"))

        return results

    # ── triple candle ────────────────────────────────────────────────────────

    def _triple(self, df: pd.DataFrame) -> list[PatternResult]:
        if len(df) < 3:
            return []
        results = []
        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        # Morning Star
        if (self._bear(c1) and self._body(c2) < self._body(c1) * 0.3
                and self._bull(c3)
                and c3["close"] > (c1["open"] + c1["close"]) / 2):
            results.append(PatternResult("Morning Star", Signal.STRONG_BUY, 0.85,
                "Morning Star — 3-candle bullish reversal, very reliable"))

        # Evening Star
        if (self._bull(c1) and self._body(c2) < self._body(c1) * 0.3
                and self._bear(c3)
                and c3["close"] < (c1["open"] + c1["close"]) / 2):
            results.append(PatternResult("Evening Star", Signal.STRONG_SELL, 0.85,
                "Evening Star — 3-candle bearish reversal, very reliable"))

        # Three White Soldiers — 3 consecutive bullish candles, each closing higher
        if (self._bull(c1) and self._bull(c2) and self._bull(c3)
                and c2["close"] > c1["close"] and c3["close"] > c2["close"]
                and c2["open"] > c1["open"] and c3["open"] > c2["open"]
                and self._body(c1) / self._range(c1) > 0.6 if self._range(c1) > 0 else False
                and self._body(c2) / self._range(c2) > 0.6 if self._range(c2) > 0 else False
                and self._body(c3) / self._range(c3) > 0.6 if self._range(c3) > 0 else False):
            results.append(PatternResult("Three White Soldiers", Signal.STRONG_BUY, 0.88,
                "Three White Soldiers — 3 strong green candles, powerful uptrend"))

        # Three Black Crows — 3 consecutive bearish candles, each closing lower
        if (self._bear(c1) and self._bear(c2) and self._bear(c3)
                and c2["close"] < c1["close"] and c3["close"] < c2["close"]
                and c2["open"] < c1["open"] and c3["open"] < c2["open"]
                and self._body(c1) / self._range(c1) > 0.6 if self._range(c1) > 0 else False
                and self._body(c2) / self._range(c2) > 0.6 if self._range(c2) > 0 else False
                and self._body(c3) / self._range(c3) > 0.6 if self._range(c3) > 0 else False):
            results.append(PatternResult("Three Black Crows", Signal.STRONG_SELL, 0.88,
                "Three Black Crows — 3 strong red candles, powerful downtrend"))

        # Three Inside Up
        if (self._bear(c1) and self._bull(c2)
                and c2["open"] > c1["close"] and c2["close"] < c1["open"]
                and self._bull(c3) and c3["close"] > c1["open"]):
            results.append(PatternResult("Three Inside Up", Signal.STRONG_BUY, 0.75,
                "Three Inside Up — bullish reversal confirmed by 3rd candle"))

        # Three Inside Down
        if (self._bull(c1) and self._bear(c2)
                and c2["open"] < c1["close"] and c2["close"] > c1["open"]
                and self._bear(c3) and c3["close"] < c1["open"]):
            results.append(PatternResult("Three Inside Down", Signal.STRONG_SELL, 0.75,
                "Three Inside Down — bearish reversal confirmed by 3rd candle"))

        return results

    # ── chart patterns ───────────────────────────────────────────────────────

    def _chart(self, df: pd.DataFrame) -> list[PatternResult]:
        results = []
        if len(df) < 30:
            return results

        highs = df["high"].iloc[-40:]
        lows = df["low"].iloc[-40:]

        peaks = [i for i in range(2, len(highs) - 2)
                 if highs.iloc[i] > highs.iloc[i-1]
                 and highs.iloc[i] > highs.iloc[i-2]
                 and highs.iloc[i] > highs.iloc[i+1]
                 and highs.iloc[i] > highs.iloc[i+2]]

        troughs = [i for i in range(2, len(lows) - 2)
                   if lows.iloc[i] < lows.iloc[i-1]
                   and lows.iloc[i] < lows.iloc[i-2]
                   and lows.iloc[i] < lows.iloc[i+1]
                   and lows.iloc[i] < lows.iloc[i+2]]

        double_top = False
        double_bottom = False

        # Double Top — only if last peak is RECENT (within last 10 candles)
        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            if (abs(highs.iloc[p1] - highs.iloc[p2]) < highs.iloc[p1] * 0.015
                    and p2 - p1 >= 5
                    and p2 >= len(highs) - 10):  # recent peak
                double_top = True
                results.append(PatternResult("Double Top", Signal.STRONG_SELL, 0.80,
                    "Double Top — price failed twice at same resistance, bearish"))

        # Double Bottom — only if last trough is RECENT (within last 10 candles)
        # AND no double top detected (they cannot coexist)
        if len(troughs) >= 2 and not double_top:
            t1, t2 = troughs[-2], troughs[-1]
            if (abs(lows.iloc[t1] - lows.iloc[t2]) < lows.iloc[t1] * 0.015
                    and t2 - t1 >= 5
                    and t2 >= len(lows) - 10):  # recent trough
                double_bottom = True
                results.append(PatternResult("Double Bottom", Signal.STRONG_BUY, 0.80,
                    "Double Bottom — price bounced twice from same support, bullish"))

        # Head and Shoulders — only if no double bottom
        if len(peaks) >= 3 and not double_bottom:
            left, head, right = peaks[-3], peaks[-2], peaks[-1]
            if (highs.iloc[head] > highs.iloc[left]
                    and highs.iloc[head] > highs.iloc[right]
                    and abs(highs.iloc[left] - highs.iloc[right]) < highs.iloc[head] * 0.02
                    and right >= len(highs) - 10):
                results.append(PatternResult("Head and Shoulders", Signal.STRONG_SELL, 0.85,
                    "Head & Shoulders — classic bearish reversal, neckline break = sell"))

        # Inverse Head and Shoulders — only if no double top
        if len(troughs) >= 3 and not double_top:
            left, head, right = troughs[-3], troughs[-2], troughs[-1]
            if (lows.iloc[head] < lows.iloc[left]
                    and lows.iloc[head] < lows.iloc[right]
                    and abs(lows.iloc[left] - lows.iloc[right]) < lows.iloc[head] * 0.02
                    and right >= len(lows) - 10):
                results.append(PatternResult("Inverse Head and Shoulders", Signal.STRONG_BUY, 0.85,
                    "Inverse H&S — classic bullish reversal, neckline break = buy"))

        return results
