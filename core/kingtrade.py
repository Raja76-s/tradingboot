"""
KingTrade Indicator — Single composite signal from all indicators.

Combines:
- Trend (EMA, Supertrend, Ichimoku, ADX, PSAR)
- Momentum (RSI, MACD, Stoch, Williams, CCI, MFI, TSI)
- Volatility (BB, Keltner, ATR)
- Volume (OBV, CMF, Force Index)
- Smart Money (BOS, CHoCH, Order Blocks, FVG, Liquidity)
- Price Action (Candlestick patterns)

Output: -100 to +100
  +80 to +100 = STRONG BUY
  +50 to +79  = BUY
  +20 to +49  = WEAK BUY
  -19 to +19  = NEUTRAL
  -20 to -49  = WEAK SELL
  -50 to -79  = SELL
  -80 to -100 = STRONG SELL
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from core.indicators import (
    TrendIndicators, MomentumIndicators,
    VolatilityIndicators, VolumeIndicators, Signal
)
from core.patterns import CandlestickPatterns
from core.smc import SmartMoneyConcepts


@dataclass
class KingTradeResult:
    score: float          # -100 to +100
    signal: str           # STRONG_BUY / BUY / WEAK_BUY / NEUTRAL / WEAK_SELL / SELL / STRONG_SELL
    trend_score: float    # trend component
    momentum_score: float # momentum component
    volume_score: float   # volume component
    smc_score: float      # smart money component
    pattern_score: float  # pattern component
    details: str          # human readable summary


class KingTradeIndicator:
    """
    Single composite indicator combining everything.
    Score = weighted average of 5 categories.
    """

    # Category weights — trend is most important
    WEIGHTS = {
        "trend":    0.30,  # 30% — direction
        "momentum": 0.25,  # 25% — strength
        "volume":   0.15,  # 15% — confirmation
        "smc":      0.20,  # 20% — smart money
        "pattern":  0.10,  # 10% — price action
    }

    def __init__(self) -> None:
        self.smc = SmartMoneyConcepts()
        self.patterns = CandlestickPatterns()

    def compute(self, df: pd.DataFrame) -> KingTradeResult:
        if len(df) < 60:
            return KingTradeResult(0, "NEUTRAL", 0, 0, 0, 0, 0, "Insufficient data")

        trend_score    = self._trend_score(df)
        momentum_score = self._momentum_score(df)
        volume_score   = self._volume_score(df)
        smc_score      = self._smc_score(df)
        pattern_score  = self._pattern_score(df)

        # Weighted composite
        score = (
            trend_score    * self.WEIGHTS["trend"] +
            momentum_score * self.WEIGHTS["momentum"] +
            volume_score   * self.WEIGHTS["volume"] +
            smc_score      * self.WEIGHTS["smc"] +
            pattern_score  * self.WEIGHTS["pattern"]
        )
        score = round(max(-100, min(100, score)), 1)

        if score >= 80:    signal = "STRONG_BUY"
        elif score >= 50:  signal = "BUY"
        elif score >= 20:  signal = "WEAK_BUY"
        elif score >= -19: signal = "NEUTRAL"
        elif score >= -49: signal = "WEAK_SELL"
        elif score >= -79: signal = "SELL"
        else:              signal = "STRONG_SELL"

        details = (
            f"KingTrade: {score:+.1f} | "
            f"Trend:{trend_score:+.0f} "
            f"Mom:{momentum_score:+.0f} "
            f"Vol:{volume_score:+.0f} "
            f"SMC:{smc_score:+.0f} "
            f"Pat:{pattern_score:+.0f}"
        )

        return KingTradeResult(
            score=score,
            signal=signal,
            trend_score=round(trend_score, 1),
            momentum_score=round(momentum_score, 1),
            volume_score=round(volume_score, 1),
            smc_score=round(smc_score, 1),
            pattern_score=round(pattern_score, 1),
            details=details,
        )

    # ── Trend Score (-100 to +100) ────────────────────────────────────────

    def _trend_score(self, df: pd.DataFrame) -> float:
        scores = []
        close = df["close"].iloc[-1]

        # EMA 9/21 crossover
        ema9  = TrendIndicators.ema(df, 9).iloc[-1]
        ema21 = TrendIndicators.ema(df, 21).iloc[-1]
        scores.append(100 if ema9 > ema21 else -100)

        # EMA 50/200 Golden/Death Cross — highest weight
        ema50  = TrendIndicators.ema(df, 50).iloc[-1]
        ema200 = TrendIndicators.ema(df, 200).iloc[-1]
        cross_score = 100 if ema50 > ema200 else -100
        scores.extend([cross_score, cross_score])  # double weight

        # Price vs EMA200
        scores.append(100 if close > ema200 else -100)

        # Supertrend
        st = TrendIndicators.supertrend(df)
        scores.append(100 if st["direction"].iloc[-1] == 1 else -100)

        # Ichimoku
        ich = TrendIndicators.ichimoku(df)
        span_a = ich["senkou_span_a"].iloc[-1]
        span_b = ich["senkou_span_b"].iloc[-1]
        if not (pd.isna(span_a) or pd.isna(span_b)):
            cloud_top = max(span_a, span_b)
            cloud_bot = min(span_a, span_b)
            if close > cloud_top:   scores.append(100)
            elif close < cloud_bot: scores.append(-100)
            else:                   scores.append(0)

        # Parabolic SAR
        psar = TrendIndicators.parabolic_sar(df).iloc[-1]
        if not pd.isna(psar):
            scores.append(100 if close > psar else -100)

        # ADX direction
        adx = TrendIndicators.adx(df)
        adx_val = adx["adx"].iloc[-1]
        if not pd.isna(adx_val) and adx_val > 20:
            scores.append(100 if adx["plus_di"].iloc[-1] > adx["minus_di"].iloc[-1] else -100)

        # VWAP
        vwap = TrendIndicators.vwap(df).iloc[-1]
        scores.append(100 if close > vwap else -100)

        return float(np.mean(scores)) if scores else 0.0

    # ── Momentum Score (-100 to +100) ─────────────────────────────────────

    def _momentum_score(self, df: pd.DataFrame) -> float:
        scores = []

        # RSI — normalize 0-100 to -100 to +100
        rsi = MomentumIndicators.rsi(df).iloc[-1]
        if not pd.isna(rsi):
            scores.append((rsi - 50) * 2)  # 50→0, 30→-40, 70→+40

        # MACD histogram direction
        macd = MomentumIndicators.macd(df)
        hist = macd["histogram"].iloc[-1]
        prev_hist = macd["histogram"].iloc[-2]
        if not pd.isna(hist):
            if hist > 0 and hist > prev_hist:   scores.append(75)
            elif hist > 0:                       scores.append(40)
            elif hist < 0 and hist < prev_hist:  scores.append(-75)
            elif hist < 0:                       scores.append(-40)
            else:                                scores.append(0)

        # Stochastic RSI
        srsi = MomentumIndicators.stochastic_rsi(df)
        k = srsi["stoch_rsi_k"].iloc[-1]
        if not pd.isna(k):
            scores.append((k - 50) * 2)

        # Williams %R — normalize -100..0 to -100..+100
        wr = MomentumIndicators.williams_r(df).iloc[-1]
        if not pd.isna(wr):
            scores.append((wr + 50) * 2)  # -100→-100, -50→0, 0→+100

        # CCI — normalize
        cci = MomentumIndicators.cci(df).iloc[-1]
        if not pd.isna(cci):
            scores.append(max(-100, min(100, cci / 2)))

        # MFI
        mfi = MomentumIndicators.mfi(df).iloc[-1]
        if not pd.isna(mfi):
            scores.append((mfi - 50) * 2)

        # TSI
        tsi = MomentumIndicators.tsi(df).iloc[-1]
        if not pd.isna(tsi):
            scores.append(max(-100, min(100, tsi * 2)))

        # ROC
        roc = MomentumIndicators.roc(df).iloc[-1]
        if not pd.isna(roc):
            scores.append(max(-100, min(100, roc * 10)))

        return float(np.mean(scores)) if scores else 0.0

    # ── Volume Score (-100 to +100) ───────────────────────────────────────

    def _volume_score(self, df: pd.DataFrame) -> float:
        scores = []

        # OBV trend
        obv = VolumeIndicators.obv(df)
        obv_sma = obv.rolling(20).mean().iloc[-1]
        if not pd.isna(obv_sma):
            scores.append(100 if obv.iloc[-1] > obv_sma else -100)

        # CMF
        cmf = VolumeIndicators.cmf(df).iloc[-1]
        if not pd.isna(cmf):
            scores.append(max(-100, min(100, cmf * 500)))

        # Bollinger Bands %B
        bb = VolatilityIndicators.bollinger_bands(df)
        bb_pct = bb["bb_pct"].iloc[-1]
        if not pd.isna(bb_pct):
            scores.append((bb_pct - 0.5) * 200)

        # Keltner Channels
        kc = VolatilityIndicators.keltner_channels(df)
        close = df["close"].iloc[-1]
        kc_mid = kc["kc_middle"].iloc[-1]
        kc_upper = kc["kc_upper"].iloc[-1]
        kc_lower = kc["kc_lower"].iloc[-1]
        if not pd.isna(kc_mid):
            kc_range = kc_upper - kc_lower
            if kc_range > 0:
                kc_pct = (close - kc_lower) / kc_range
                scores.append((kc_pct - 0.5) * 200)

        # Force Index
        fi = VolumeIndicators.force_index(df).iloc[-1]
        if not pd.isna(fi):
            scores.append(100 if fi > 0 else -100)

        return float(np.mean(scores)) if scores else 0.0

    # ── SMC Score (-100 to +100) ──────────────────────────────────────────

    def _smc_score(self, df: pd.DataFrame) -> float:
        smc_results = self.smc.analyze(df)
        if not smc_results:
            return 0.0

        signal_map = {
            "STRONG_BUY": 100, "BUY": 60,
            "NEUTRAL": 0,
            "SELL": -60, "STRONG_SELL": -100,
        }
        scores = [signal_map.get(r.signal.value, 0) * (r.weight / 3.0)
                  for r in smc_results]
        return float(np.mean(scores)) if scores else 0.0

    # ── Pattern Score (-100 to +100) ──────────────────────────────────────

    def _pattern_score(self, df: pd.DataFrame) -> float:
        pattern_results = self.patterns.detect_all(df)
        if not pattern_results:
            return 0.0

        signal_map = {
            "STRONG_BUY": 100, "BUY": 60,
            "NEUTRAL": 0,
            "SELL": -60, "STRONG_SELL": -100,
        }
        scores = [signal_map.get(p.signal.value, 0) * p.confidence
                  for p in pattern_results]
        return float(np.mean(scores)) if scores else 0.0
