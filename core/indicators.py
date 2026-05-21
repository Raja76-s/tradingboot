"""
Comprehensive Technical Indicators Module.

Contains 30+ indicators across all categories:
- Trend indicators
- Momentum indicators
- Volatility indicators
- Volume indicators

Each indicator returns a DataFrame/Series that can be used by the scoring engine.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from config.settings import StrategyConfig


class Signal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class IndicatorResult:
    name: str
    signal: Signal
    value: float
    weight: float = 1.0
    details: str = ""


class TrendIndicators:
    """Trend-following indicators."""

    @staticmethod
    def sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        return df["close"].rolling(window=period).mean()

    @staticmethod
    def ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
        return df["close"].ewm(span=period, adjust=False).mean()

    @staticmethod
    def wma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        weights = np.arange(1, period + 1, dtype=float)
        return df["close"].rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    @staticmethod
    def dema(df: pd.DataFrame, period: int = 20) -> pd.Series:
        ema1 = df["close"].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return 2 * ema1 - ema2

    @staticmethod
    def tema(df: pd.DataFrame, period: int = 20) -> pd.Series:
        ema1 = df["close"].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
        cumulative_vol = df["volume"].cumsum()
        return cumulative_tp_vol / cumulative_vol

    @staticmethod
    def supertrend(
        df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
    ) -> pd.DataFrame:
        hl2 = (df["high"] + df["low"]) / 2
        atr = TrendIndicators._atr_calc(df, period)

        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        supertrend = pd.Series(np.nan, index=df.index)
        direction = pd.Series(1, index=df.index)

        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]
                if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                    lower_band.iloc[i] = lower_band.iloc[i - 1]
                if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                    upper_band.iloc[i] = upper_band.iloc[i - 1]

            supertrend.iloc[i] = (
                lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]
            )

        result = pd.DataFrame(index=df.index)
        result["supertrend"] = supertrend
        result["direction"] = direction
        return result

    @staticmethod
    def _atr_calc(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def ichimoku(
        df: pd.DataFrame,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52,
    ) -> pd.DataFrame:
        tenkan_sen = (
            df["high"].rolling(window=tenkan).max()
            + df["low"].rolling(window=tenkan).min()
        ) / 2

        kijun_sen = (
            df["high"].rolling(window=kijun).max()
            + df["low"].rolling(window=kijun).min()
        ) / 2

        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)

        senkou_span_b_val = (
            (
                df["high"].rolling(window=senkou_b).max()
                + df["low"].rolling(window=senkou_b).min()
            )
            / 2
        ).shift(kijun)

        chikou_span = df["close"].shift(-kijun)

        result = pd.DataFrame(index=df.index)
        result["tenkan_sen"] = tenkan_sen
        result["kijun_sen"] = kijun_sen
        result["senkou_span_a"] = senkou_span_a
        result["senkou_span_b"] = senkou_span_b_val
        result["chikou_span"] = chikou_span
        return result

    @staticmethod
    def parabolic_sar(
        df: pd.DataFrame,
        initial_af: float = 0.02,
        max_af: float = 0.2,
        step_af: float = 0.02,
    ) -> pd.Series:
        length = len(df)
        sar = pd.Series(np.nan, index=df.index)
        trend = pd.Series(0, index=df.index)

        af = initial_af
        ep = df["low"].iloc[0]
        sar.iloc[0] = df["high"].iloc[0]
        trend.iloc[0] = -1

        for i in range(1, length):
            if trend.iloc[i - 1] == 1:
                sar.iloc[i] = sar.iloc[i - 1] + af * (ep - sar.iloc[i - 1])
                sar.iloc[i] = min(sar.iloc[i], df["low"].iloc[i - 1])
                if i >= 2:
                    sar.iloc[i] = min(sar.iloc[i], df["low"].iloc[i - 2])

                if df["low"].iloc[i] < sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep
                    ep = df["low"].iloc[i]
                    af = initial_af
                else:
                    trend.iloc[i] = 1
                    if df["high"].iloc[i] > ep:
                        ep = df["high"].iloc[i]
                        af = min(af + step_af, max_af)
            else:
                sar.iloc[i] = sar.iloc[i - 1] + af * (ep - sar.iloc[i - 1])
                sar.iloc[i] = max(sar.iloc[i], df["high"].iloc[i - 1])
                if i >= 2:
                    sar.iloc[i] = max(sar.iloc[i], df["high"].iloc[i - 2])

                if df["high"].iloc[i] > sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep
                    ep = df["high"].iloc[i]
                    af = initial_af
                else:
                    trend.iloc[i] = -1
                    if df["low"].iloc[i] < ep:
                        ep = df["low"].iloc[i]
                        af = min(af + step_af, max_af)

        return sar

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr = TrendIndicators._atr_calc(df, period)

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx_val = dx.rolling(window=period).mean()

        result = pd.DataFrame(index=df.index)
        result["adx"] = adx_val
        result["plus_di"] = plus_di
        result["minus_di"] = minus_di
        return result


class MomentumIndicators:
    """Momentum oscillators."""

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def stochastic_rsi(
        df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14
    ) -> pd.DataFrame:
        rsi_val = MomentumIndicators.rsi(df, rsi_period)
        min_rsi = rsi_val.rolling(window=stoch_period).min()
        max_rsi = rsi_val.rolling(window=stoch_period).max()
        stoch_rsi = (rsi_val - min_rsi) / (max_rsi - min_rsi)
        k = stoch_rsi.rolling(window=3).mean() * 100
        d = k.rolling(window=3).mean()

        result = pd.DataFrame(index=df.index)
        result["stoch_rsi_k"] = k
        result["stoch_rsi_d"] = d
        return result

    @staticmethod
    def macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
    ) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        result = pd.DataFrame(index=df.index)
        result["macd"] = macd_line
        result["signal"] = signal_line
        result["histogram"] = histogram
        return result

    @staticmethod
    def stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        smooth: int = 3,
    ) -> pd.DataFrame:
        lowest_low = df["low"].rolling(window=k_period).min()
        highest_high = df["high"].rolling(window=k_period).max()
        k = ((df["close"] - lowest_low) / (highest_high - lowest_low)) * 100
        k = k.rolling(window=smooth).mean()
        d = k.rolling(window=d_period).mean()

        result = pd.DataFrame(index=df.index)
        result["stoch_k"] = k
        result["stoch_d"] = d
        return result

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        highest_high = df["high"].rolling(window=period).max()
        lowest_low = df["low"].rolling(window=period).min()
        return -100 * (highest_high - df["close"]) / (highest_high - lowest_low)

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        return (typical_price - sma) / (0.015 * mad)

    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        raw_money_flow = typical_price * df["volume"]
        delta = typical_price.diff()

        positive_flow = raw_money_flow.where(delta > 0, 0.0)
        negative_flow = raw_money_flow.where(delta < 0, 0.0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        money_ratio = positive_mf / negative_mf
        return 100 - (100 / (1 + money_ratio))

    @staticmethod
    def roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
        return ((df["close"] - df["close"].shift(period)) / df["close"].shift(period)) * 100

    @staticmethod
    def awesome_oscillator(df: pd.DataFrame) -> pd.Series:
        midpoint = (df["high"] + df["low"]) / 2
        return midpoint.rolling(window=5).mean() - midpoint.rolling(window=34).mean()

    @staticmethod
    def ultimate_oscillator(
        df: pd.DataFrame,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
    ) -> pd.Series:
        close_prev = df["close"].shift(1)
        bp = df["close"] - pd.concat([df["low"], close_prev], axis=1).min(axis=1)
        tr = pd.concat(
            [df["high"] - df["low"], (df["high"] - close_prev).abs(), (df["low"] - close_prev).abs()],
            axis=1,
        ).max(axis=1)

        avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()

        return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

    @staticmethod
    def tsi(df: pd.DataFrame, long_period: int = 25, short_period: int = 13) -> pd.Series:
        diff = df["close"].diff()
        double_smoothed = diff.ewm(span=long_period).mean().ewm(span=short_period).mean()
        double_smoothed_abs = diff.abs().ewm(span=long_period).mean().ewm(span=short_period).mean()
        return 100 * double_smoothed / double_smoothed_abs


class VolatilityIndicators:
    """Volatility-based indicators."""

    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> pd.DataFrame:
        sma = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()

        result = pd.DataFrame(index=df.index)
        result["bb_upper"] = sma + std_dev * std
        result["bb_middle"] = sma
        result["bb_lower"] = sma - std_dev * std
        result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_middle"]
        result["bb_pct"] = (df["close"] - result["bb_lower"]) / (
            result["bb_upper"] - result["bb_lower"]
        )
        return result

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        return TrendIndicators._atr_calc(df, period)

    @staticmethod
    def keltner_channels(
        df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14, multiplier: float = 2.0
    ) -> pd.DataFrame:
        ema = df["close"].ewm(span=ema_period, adjust=False).mean()
        atr_val = TrendIndicators._atr_calc(df, atr_period)

        result = pd.DataFrame(index=df.index)
        result["kc_upper"] = ema + multiplier * atr_val
        result["kc_middle"] = ema
        result["kc_lower"] = ema - multiplier * atr_val
        return result

    @staticmethod
    def donchian_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        result["dc_upper"] = df["high"].rolling(window=period).max()
        result["dc_lower"] = df["low"].rolling(window=period).min()
        result["dc_middle"] = (result["dc_upper"] + result["dc_lower"]) / 2
        return result

    @staticmethod
    def historical_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
        log_returns = np.log(df["close"] / df["close"].shift(1))
        return log_returns.rolling(window=period).std() * np.sqrt(252)

    @staticmethod
    def chaikin_volatility(df: pd.DataFrame, period: int = 10) -> pd.Series:
        hl_ema = (df["high"] - df["low"]).ewm(span=period).mean()
        return (hl_ema - hl_ema.shift(period)) / hl_ema.shift(period) * 100


class VolumeIndicators:
    """Volume-based indicators."""

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        obv_vals = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv_vals.iloc[i] = obv_vals.iloc[i - 1] + df["volume"].iloc[i]
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv_vals.iloc[i] = obv_vals.iloc[i - 1] - df["volume"].iloc[i]
            else:
                obv_vals.iloc[i] = obv_vals.iloc[i - 1]
        return obv_vals

    @staticmethod
    def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"]
        )
        mfv = mfm * df["volume"]
        return mfv.rolling(window=period).sum() / df["volume"].rolling(window=period).sum()

    @staticmethod
    def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"]
        )
        ad = (mfm * df["volume"]).cumsum()
        return ad

    @staticmethod
    def volume_profile(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
        price_range = np.linspace(df["low"].min(), df["high"].max(), bins + 1)
        volume_at_price = pd.Series(0.0, index=range(bins))

        for i in range(len(df)):
            for j in range(bins):
                if price_range[j] <= df["close"].iloc[i] <= price_range[j + 1]:
                    volume_at_price.iloc[j] += df["volume"].iloc[i]
                    break

        result = pd.DataFrame({
            "price_low": price_range[:-1],
            "price_high": price_range[1:],
            "volume": volume_at_price.values,
        })
        return result

    @staticmethod
    def force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
        fi = df["close"].diff() * df["volume"]
        return fi.ewm(span=period).mean()

    @staticmethod
    def ease_of_movement(df: pd.DataFrame, period: int = 14) -> pd.Series:
        dm = ((df["high"] + df["low"]) / 2) - ((df["high"].shift(1) + df["low"].shift(1)) / 2)
        box_ratio = (df["volume"] / 1e6) / (df["high"] - df["low"])
        emv = dm / box_ratio
        return emv.rolling(window=period).mean()

    @staticmethod
    def volume_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df["volume"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


class IndicatorEngine:
    """Runs all indicators and produces signals."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def compute_all(self, df: pd.DataFrame) -> list[IndicatorResult]:
        if len(df) < 60:
            return []

        results: list[IndicatorResult] = []

        results.extend(self._evaluate_trend(df))
        results.extend(self._evaluate_momentum(df))
        results.extend(self._evaluate_volatility(df))
        results.extend(self._evaluate_volume(df))

        return results

    def _evaluate_trend(self, df: pd.DataFrame) -> list[IndicatorResult]:
        results: list[IndicatorResult] = []
        close = df["close"].iloc[-1]

        # EMA Crossover (9/21)
        ema_fast = TrendIndicators.ema(df, self.config.ema_fast).iloc[-1]
        ema_slow = TrendIndicators.ema(df, self.config.ema_medium).iloc[-1]
        ema_prev_fast = TrendIndicators.ema(df, self.config.ema_fast).iloc[-2]
        ema_prev_slow = TrendIndicators.ema(df, self.config.ema_medium).iloc[-2]

        if ema_prev_fast <= ema_prev_slow and ema_fast > ema_slow:
            signal = Signal.STRONG_BUY
        elif ema_prev_fast >= ema_prev_slow and ema_fast < ema_slow:
            signal = Signal.STRONG_SELL
        elif ema_fast > ema_slow:
            signal = Signal.BUY
        elif ema_fast < ema_slow:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        results.append(IndicatorResult(
            name="EMA Crossover (9/21)",
            signal=signal,
            value=ema_fast - ema_slow,
            weight=1.5,
            details=f"EMA9={ema_fast:.2f}, EMA21={ema_slow:.2f}",
        ))

        # EMA 50/200 (Golden/Death Cross)
        ema50 = TrendIndicators.ema(df, self.config.ema_slow).iloc[-1]
        ema200 = TrendIndicators.ema(df, self.config.ema_very_slow).iloc[-1]
        ema50_prev = TrendIndicators.ema(df, self.config.ema_slow).iloc[-2]
        ema200_prev = TrendIndicators.ema(df, self.config.ema_very_slow).iloc[-2]

        if ema50_prev <= ema200_prev and ema50 > ema200:
            signal = Signal.STRONG_BUY
        elif ema50_prev >= ema200_prev and ema50 < ema200:
            signal = Signal.STRONG_SELL
        elif ema50 > ema200:
            signal = Signal.BUY
        else:
            signal = Signal.SELL

        results.append(IndicatorResult(
            name="Golden/Death Cross (50/200)",
            signal=signal,
            value=ema50 - ema200,
            weight=2.0,
            details=f"EMA50={ema50:.2f}, EMA200={ema200:.2f}",
        ))

        # Price vs EMA200
        if close > ema200 * 1.02:
            signal = Signal.BUY
        elif close < ema200 * 0.98:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        results.append(IndicatorResult(
            name="Price vs EMA200",
            signal=signal,
            value=close - ema200,
            weight=1.0,
            details=f"Close={close:.2f}, EMA200={ema200:.2f}",
        ))

        # Supertrend
        st = TrendIndicators.supertrend(
            df, self.config.supertrend_period, self.config.supertrend_multiplier
        )
        st_dir = st["direction"].iloc[-1]
        st_prev_dir = st["direction"].iloc[-2]

        if st_prev_dir == -1 and st_dir == 1:
            signal = Signal.STRONG_BUY
        elif st_prev_dir == 1 and st_dir == -1:
            signal = Signal.STRONG_SELL
        elif st_dir == 1:
            signal = Signal.BUY
        else:
            signal = Signal.SELL

        results.append(IndicatorResult(
            name="Supertrend",
            signal=signal,
            value=st_dir,
            weight=1.5,
            details=f"Direction={'UP' if st_dir == 1 else 'DOWN'}",
        ))

        # Ichimoku Cloud
        ich = TrendIndicators.ichimoku(
            df, self.config.ichimoku_tenkan, self.config.ichimoku_kijun, self.config.ichimoku_senkou_b
        )
        tenkan = ich["tenkan_sen"].iloc[-1]
        kijun = ich["kijun_sen"].iloc[-1]
        span_a = ich["senkou_span_a"].iloc[-1]
        span_b = ich["senkou_span_b"].iloc[-1]

        cloud_top = max(span_a, span_b) if not (pd.isna(span_a) or pd.isna(span_b)) else close
        cloud_bottom = min(span_a, span_b) if not (pd.isna(span_a) or pd.isna(span_b)) else close

        if close > cloud_top and tenkan > kijun:
            signal = Signal.STRONG_BUY
        elif close > cloud_top:
            signal = Signal.BUY
        elif close < cloud_bottom and tenkan < kijun:
            signal = Signal.STRONG_SELL
        elif close < cloud_bottom:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        results.append(IndicatorResult(
            name="Ichimoku Cloud",
            signal=signal,
            value=close - cloud_top,
            weight=2.0,
            details=f"Price vs Cloud: {'Above' if close > cloud_top else 'Below' if close < cloud_bottom else 'Inside'}",
        ))

        # Parabolic SAR
        psar = TrendIndicators.parabolic_sar(df)
        psar_val = psar.iloc[-1]
        if not pd.isna(psar_val):
            if close > psar_val:
                signal = Signal.BUY
            else:
                signal = Signal.SELL
            results.append(IndicatorResult(
                name="Parabolic SAR",
                signal=signal,
                value=close - psar_val,
                weight=1.0,
                details=f"SAR={psar_val:.2f}, Price={'Above' if close > psar_val else 'Below'}",
            ))

        # ADX
        adx_data = TrendIndicators.adx(df, self.config.adx_period)
        adx_val = adx_data["adx"].iloc[-1]
        plus_di = adx_data["plus_di"].iloc[-1]
        minus_di = adx_data["minus_di"].iloc[-1]

        if not pd.isna(adx_val):
            if adx_val > self.config.adx_strong_trend:
                if plus_di > minus_di:
                    signal = Signal.BUY
                else:
                    signal = Signal.SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="ADX",
                signal=signal,
                value=adx_val,
                weight=1.0,
                details=f"ADX={adx_val:.1f}, +DI={plus_di:.1f}, -DI={minus_di:.1f}",
            ))

        # VWAP
        vwap_val = TrendIndicators.vwap(df).iloc[-1]
        if close > vwap_val * 1.005:
            signal = Signal.BUY
        elif close < vwap_val * 0.995:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        results.append(IndicatorResult(
            name="VWAP",
            signal=signal,
            value=close - vwap_val,
            weight=1.0,
            details=f"VWAP={vwap_val:.2f}",
        ))

        # DEMA
        dema_val = TrendIndicators.dema(df, 21).iloc[-1]
        if close > dema_val:
            signal = Signal.BUY
        else:
            signal = Signal.SELL

        results.append(IndicatorResult(
            name="DEMA (21)",
            signal=signal,
            value=close - dema_val,
            weight=0.8,
            details=f"DEMA={dema_val:.2f}",
        ))

        # TEMA
        tema_val = TrendIndicators.tema(df, 21).iloc[-1]
        if close > tema_val:
            signal = Signal.BUY
        else:
            signal = Signal.SELL

        results.append(IndicatorResult(
            name="TEMA (21)",
            signal=signal,
            value=close - tema_val,
            weight=0.8,
            details=f"TEMA={tema_val:.2f}",
        ))

        return results

    def _evaluate_momentum(self, df: pd.DataFrame) -> list[IndicatorResult]:
        results: list[IndicatorResult] = []

        # RSI
        rsi_val = MomentumIndicators.rsi(df, self.config.rsi_period).iloc[-1]
        MomentumIndicators.rsi(df, self.config.rsi_period).iloc[-2]

        if rsi_val < self.config.rsi_oversold:
            signal = Signal.STRONG_BUY
        elif rsi_val < 40:
            signal = Signal.BUY
        elif rsi_val > self.config.rsi_overbought:
            signal = Signal.STRONG_SELL
        elif rsi_val > 60:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        # RSI divergence detection
        price_higher = df["close"].iloc[-1] > df["close"].iloc[-5]
        rsi_higher = rsi_val > MomentumIndicators.rsi(df, self.config.rsi_period).iloc[-5]

        divergence = ""
        if price_higher and not rsi_higher:
            divergence = " (Bearish Divergence!)"
            if signal in (Signal.NEUTRAL, Signal.BUY):
                signal = Signal.SELL
        elif not price_higher and rsi_higher:
            divergence = " (Bullish Divergence!)"
            if signal in (Signal.NEUTRAL, Signal.SELL):
                signal = Signal.BUY

        results.append(IndicatorResult(
            name="RSI",
            signal=signal,
            value=rsi_val,
            weight=1.5,
            details=f"RSI={rsi_val:.1f}{divergence}",
        ))

        # Stochastic RSI
        stoch_rsi = MomentumIndicators.stochastic_rsi(df)
        srsi_k = stoch_rsi["stoch_rsi_k"].iloc[-1]
        srsi_d = stoch_rsi["stoch_rsi_d"].iloc[-1]

        if not (pd.isna(srsi_k) or pd.isna(srsi_d)):
            if srsi_k < 20 and srsi_k > srsi_d:
                signal = Signal.STRONG_BUY
            elif srsi_k < 30:
                signal = Signal.BUY
            elif srsi_k > 80 and srsi_k < srsi_d:
                signal = Signal.STRONG_SELL
            elif srsi_k > 70:
                signal = Signal.SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="Stochastic RSI",
                signal=signal,
                value=srsi_k,
                weight=1.2,
                details=f"StochRSI K={srsi_k:.1f}, D={srsi_d:.1f}",
            ))

        # MACD
        macd_data = MomentumIndicators.macd(
            df, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
        )
        macd_val = macd_data["macd"].iloc[-1]
        macd_signal = macd_data["signal"].iloc[-1]
        macd_hist = macd_data["histogram"].iloc[-1]
        macd_prev_hist = macd_data["histogram"].iloc[-2]

        if macd_prev_hist < 0 and macd_hist > 0:
            signal = Signal.STRONG_BUY
        elif macd_prev_hist > 0 and macd_hist < 0:
            signal = Signal.STRONG_SELL
        elif macd_hist > 0 and macd_hist > macd_prev_hist:
            signal = Signal.BUY
        elif macd_hist < 0 and macd_hist < macd_prev_hist:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        results.append(IndicatorResult(
            name="MACD",
            signal=signal,
            value=macd_hist,
            weight=1.5,
            details=f"MACD={macd_val:.4f}, Signal={macd_signal:.4f}, Hist={macd_hist:.4f}",
        ))

        # Stochastic Oscillator
        stoch = MomentumIndicators.stochastic(
            df, self.config.stoch_k, self.config.stoch_d, self.config.stoch_smooth
        )
        stoch_k_val = stoch["stoch_k"].iloc[-1]
        stoch_d_val = stoch["stoch_d"].iloc[-1]

        if not (pd.isna(stoch_k_val) or pd.isna(stoch_d_val)):
            if stoch_k_val < 20 and stoch_k_val > stoch_d_val:
                signal = Signal.STRONG_BUY
            elif stoch_k_val < 30:
                signal = Signal.BUY
            elif stoch_k_val > 80 and stoch_k_val < stoch_d_val:
                signal = Signal.STRONG_SELL
            elif stoch_k_val > 70:
                signal = Signal.SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="Stochastic Oscillator",
                signal=signal,
                value=stoch_k_val,
                weight=1.2,
                details=f"K={stoch_k_val:.1f}, D={stoch_d_val:.1f}",
            ))

        # Williams %R
        wr = MomentumIndicators.williams_r(df, self.config.williams_period).iloc[-1]
        if not pd.isna(wr):
            if wr < -80:
                signal = Signal.STRONG_BUY
            elif wr < -50:
                signal = Signal.BUY
            elif wr > -20:
                signal = Signal.STRONG_SELL
            elif wr > -50:
                signal = Signal.SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="Williams %R",
                signal=signal,
                value=wr,
                weight=1.0,
                details=f"Williams %R={wr:.1f}",
            ))

        # CCI
        cci_val = MomentumIndicators.cci(df, self.config.cci_period).iloc[-1]
        if not pd.isna(cci_val):
            if cci_val < self.config.cci_oversold:
                signal = Signal.STRONG_BUY
            elif cci_val < 0:
                signal = Signal.BUY
            elif cci_val > self.config.cci_overbought:
                signal = Signal.STRONG_SELL
            elif cci_val > 0:
                signal = Signal.SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="CCI",
                signal=signal,
                value=cci_val,
                weight=1.0,
                details=f"CCI={cci_val:.1f}",
            ))

        # MFI
        mfi_val = MomentumIndicators.mfi(df, self.config.mfi_period).iloc[-1]
        if not pd.isna(mfi_val):
            if mfi_val < self.config.mfi_oversold:
                signal = Signal.STRONG_BUY
            elif mfi_val < 40:
                signal = Signal.BUY
            elif mfi_val > self.config.mfi_overbought:
                signal = Signal.STRONG_SELL
            elif mfi_val > 60:
                signal = Signal.SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="MFI",
                signal=signal,
                value=mfi_val,
                weight=1.0,
                details=f"MFI={mfi_val:.1f}",
            ))

        # ROC
        roc_val = MomentumIndicators.roc(df, 12).iloc[-1]
        if not pd.isna(roc_val):
            if roc_val > 5:
                signal = Signal.STRONG_BUY
            elif roc_val > 0:
                signal = Signal.BUY
            elif roc_val < -5:
                signal = Signal.STRONG_SELL
            elif roc_val < 0:
                signal = Signal.SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="ROC",
                signal=signal,
                value=roc_val,
                weight=0.8,
                details=f"ROC={roc_val:.2f}%",
            ))

        # Awesome Oscillator
        ao = MomentumIndicators.awesome_oscillator(df).iloc[-1]
        ao_prev = MomentumIndicators.awesome_oscillator(df).iloc[-2]
        if not pd.isna(ao):
            if ao > 0 and ao > ao_prev:
                signal = Signal.BUY
            elif ao < 0 and ao < ao_prev:
                signal = Signal.SELL
            elif ao > 0 and ao_prev < 0:
                signal = Signal.STRONG_BUY
            elif ao < 0 and ao_prev > 0:
                signal = Signal.STRONG_SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="Awesome Oscillator",
                signal=signal,
                value=ao,
                weight=0.8,
                details=f"AO={ao:.4f}",
            ))

        # Ultimate Oscillator
        uo = MomentumIndicators.ultimate_oscillator(df).iloc[-1]
        if not pd.isna(uo):
            if uo < 30:
                signal = Signal.STRONG_BUY
            elif uo < 50:
                signal = Signal.BUY
            elif uo > 70:
                signal = Signal.STRONG_SELL
            elif uo > 50:
                signal = Signal.SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="Ultimate Oscillator",
                signal=signal,
                value=uo,
                weight=0.8,
                details=f"UO={uo:.1f}",
            ))

        # TSI
        tsi_val = MomentumIndicators.tsi(df).iloc[-1]
        if not pd.isna(tsi_val):
            if tsi_val > 25:
                signal = Signal.STRONG_BUY
            elif tsi_val > 0:
                signal = Signal.BUY
            elif tsi_val < -25:
                signal = Signal.STRONG_SELL
            elif tsi_val < 0:
                signal = Signal.SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="TSI",
                signal=signal,
                value=tsi_val,
                weight=0.8,
                details=f"TSI={tsi_val:.1f}",
            ))

        return results

    def _evaluate_volatility(self, df: pd.DataFrame) -> list[IndicatorResult]:
        results: list[IndicatorResult] = []
        close = df["close"].iloc[-1]

        # Bollinger Bands
        bb = VolatilityIndicators.bollinger_bands(
            df, self.config.bb_period, self.config.bb_std_dev
        )
        bb_pct = bb["bb_pct"].iloc[-1]
        bb_width = bb["bb_width"].iloc[-1]

        if not pd.isna(bb_pct):
            if bb_pct < 0:
                signal = Signal.STRONG_BUY
            elif bb_pct < 0.2:
                signal = Signal.BUY
            elif bb_pct > 1:
                signal = Signal.STRONG_SELL
            elif bb_pct > 0.8:
                signal = Signal.SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="Bollinger Bands",
                signal=signal,
                value=bb_pct,
                weight=1.5,
                details=f"BB%={bb_pct:.2f}, Width={bb_width:.4f}",
            ))

        # Keltner Channels
        kc = VolatilityIndicators.keltner_channels(df)
        kc_upper = kc["kc_upper"].iloc[-1]
        kc_lower = kc["kc_lower"].iloc[-1]

        if not (pd.isna(kc_upper) or pd.isna(kc_lower)):
            if close < kc_lower:
                signal = Signal.STRONG_BUY
            elif close > kc_upper:
                signal = Signal.STRONG_SELL
            elif close < kc["kc_middle"].iloc[-1]:
                signal = Signal.BUY
            elif close > kc["kc_middle"].iloc[-1]:
                signal = Signal.SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="Keltner Channels",
                signal=signal,
                value=close,
                weight=1.0,
                details=f"Upper={kc_upper:.2f}, Lower={kc_lower:.2f}",
            ))

        # Donchian Channels
        dc = VolatilityIndicators.donchian_channels(df)
        dc_upper = dc["dc_upper"].iloc[-1]
        dc_lower = dc["dc_lower"].iloc[-1]

        if not (pd.isna(dc_upper) or pd.isna(dc_lower)):
            dc_range = dc_upper - dc_lower
            if dc_range > 0:
                dc_pct = (close - dc_lower) / dc_range
                if dc_pct > 0.95:
                    signal = Signal.BUY  # breakout
                elif dc_pct < 0.05:
                    signal = Signal.SELL  # breakdown
                elif dc_pct > 0.7:
                    signal = Signal.BUY
                elif dc_pct < 0.3:
                    signal = Signal.SELL
                else:
                    signal = Signal.NEUTRAL

                results.append(IndicatorResult(
                    name="Donchian Channels",
                    signal=signal,
                    value=dc_pct,
                    weight=0.8,
                    details=f"DC%={dc_pct:.2f}",
                ))

        # BB Squeeze (Bollinger inside Keltner = low volatility, expecting breakout)
        if not (pd.isna(bb["bb_upper"].iloc[-1]) or pd.isna(kc_upper)):
            squeeze = bb["bb_upper"].iloc[-1] < kc_upper and bb["bb_lower"].iloc[-1] > kc_lower
            if squeeze:
                results.append(IndicatorResult(
                    name="BB Squeeze",
                    signal=Signal.NEUTRAL,
                    value=1.0,
                    weight=1.5,
                    details="SQUEEZE DETECTED - Breakout imminent!",
                ))

        return results

    def _evaluate_volume(self, df: pd.DataFrame) -> list[IndicatorResult]:
        results: list[IndicatorResult] = []

        # OBV trend
        obv_vals = VolumeIndicators.obv(df)
        obv_sma = obv_vals.rolling(20).mean()
        obv_current = obv_vals.iloc[-1]
        obv_sma_current = obv_sma.iloc[-1]

        if not pd.isna(obv_sma_current):
            if obv_current > obv_sma_current:
                signal = Signal.BUY
            else:
                signal = Signal.SELL

            results.append(IndicatorResult(
                name="OBV",
                signal=signal,
                value=obv_current,
                weight=1.0,
                details=f"OBV={'Rising' if obv_current > obv_sma_current else 'Falling'}",
            ))

        # CMF
        cmf_val = VolumeIndicators.cmf(df).iloc[-1]
        if not pd.isna(cmf_val):
            if cmf_val > 0.1:
                signal = Signal.STRONG_BUY
            elif cmf_val > 0:
                signal = Signal.BUY
            elif cmf_val < -0.1:
                signal = Signal.STRONG_SELL
            elif cmf_val < 0:
                signal = Signal.SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="CMF",
                signal=signal,
                value=cmf_val,
                weight=1.0,
                details=f"CMF={cmf_val:.4f}",
            ))

        # Volume vs Average
        vol_current = df["volume"].iloc[-1]
        vol_avg = df["volume"].rolling(20).mean().iloc[-1]
        if not pd.isna(vol_avg) and vol_avg > 0:
            vol_ratio = vol_current / vol_avg
            results.append(IndicatorResult(
                name="Volume Spike",
                signal=Signal.NEUTRAL,
                value=vol_ratio,
                weight=0.5,
                details=f"Volume={vol_ratio:.1f}x average" + (" HIGH VOLUME!" if vol_ratio > 2 else ""),
            ))

        # Force Index
        fi = VolumeIndicators.force_index(df).iloc[-1]
        if not pd.isna(fi):
            if fi > 0:
                signal = Signal.BUY
            else:
                signal = Signal.SELL

            results.append(IndicatorResult(
                name="Force Index",
                signal=signal,
                value=fi,
                weight=0.8,
                details=f"FI={'Positive' if fi > 0 else 'Negative'}",
            ))

        # Volume RSI
        vrsi = VolumeIndicators.volume_rsi(df).iloc[-1]
        if not pd.isna(vrsi):
            if vrsi > 70:
                signal = Signal.BUY  # high buying volume
            elif vrsi < 30:
                signal = Signal.SELL
            else:
                signal = Signal.NEUTRAL

            results.append(IndicatorResult(
                name="Volume RSI",
                signal=signal,
                value=vrsi,
                weight=0.5,
                details=f"VRSI={vrsi:.1f}",
            ))

        return results
