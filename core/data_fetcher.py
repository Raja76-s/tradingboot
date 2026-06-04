import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from config.settings import AppConfig

PROXIES = {"http": "http://proxy.server:3128", "https": "http://proxy.server:3128"}

KUCOIN_INTERVAL = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1hour", "2h": "2hour", "4h": "4hour", "1d": "1day",
}

OKX_INTERVAL = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1H", "2h": "2H", "4h": "4H", "1d": "1D",
}

# Global price store — updated every 2s by background thread
_live_prices: dict[str, float] = {}
_live_tickers: dict[str, dict] = {}


def _start_price_poller() -> None:
    """Poll CoinDCX REST every 2 seconds in background thread."""
    def run():
        while True:
            try:
                r = requests.get(
                    "https://api.coindcx.com/exchange/ticker",
                    proxies=PROXIES, timeout=5,
                )
                for t in r.json():
                    if "market" in t and "last_price" in t:
                        market = t["market"]
                        last_price = float(t["last_price"])
                        _live_prices[market] = last_price
                        _live_tickers[market] = {
                            **t,
                            "last_price": last_price,
                            "volume": float(t.get("volume", 0) or 0),
                        }
            except Exception:
                pass
            time.sleep(2)
    threading.Thread(target=run, daemon=True).start()
    time.sleep(3)  # wait for first fetch


class DataFetcher:

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.last_data_source = "unknown"
        self._coindcx_cache: dict = {}
        self._cache_time: float = 0
        _start_price_poller()

    def get_live_price(self, pair: str) -> float | None:
        return _live_prices.get(pair)

    def _get_coindcx_tickers(self) -> dict:
        return {k: {"last_price": v} for k, v in _live_prices.items()} if _live_prices else self._coindcx_cache

    def _patch_live_price(self, df: pd.DataFrame, pair: str) -> None:
        live = self.get_live_price(pair)
        if not live:
            return
        df.iloc[-1, df.columns.get_loc("close")] = live
        if live > df.iloc[-1]["high"]:
            df.iloc[-1, df.columns.get_loc("high")] = live
        if live < df.iloc[-1]["low"]:
            df.iloc[-1, df.columns.get_loc("low")] = live

    def _fetch_kucoin(self, pair: str, interval: str, limit: int) -> pd.DataFrame | None:
        try:
            symbol = pair[:-4] + "-" + pair[-4:]
            url = (
                f"https://api.kucoin.com/api/v1/market/candles"
                f"?type={KUCOIN_INTERVAL.get(interval, '15min')}"
                f"&symbol={symbol}&pageSize={min(limit, 1500)}"
            )
            r = requests.get(url, proxies=PROXIES, timeout=15)
            data = r.json()
            if data.get("code") != "200000" or not data.get("data"):
                return None
            rows = [
                {
                    "timestamp": pd.to_datetime(int(c[0]), unit="s"),
                    "open": float(c[1]), "close": float(c[2]),
                    "high": float(c[3]), "low": float(c[4]),
                    "volume": float(c[5]),
                }
                for c in reversed(data["data"])
            ]
            return pd.DataFrame(rows).set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        except Exception:
            return None

    def _fetch_okx(self, pair: str, interval: str, limit: int) -> pd.DataFrame | None:
        try:
            symbol = pair[:-4] + "-" + pair[-4:]
            bar = OKX_INTERVAL.get(interval, "15m")
            all_rows = []
            after = None  # pagination cursor
            remaining = limit

            while remaining > 0:
                fetch = min(remaining, 300)
                url = (
                    f"https://www.okx.com/api/v5/market/history-candles"
                    f"?instId={symbol}&bar={bar}&limit={fetch}"
                )
                if after:
                    url += f"&after={after}"
                r = requests.get(url, proxies=PROXIES, timeout=15)
                data = r.json()
                if data.get("code") != "0" or not data.get("data"):
                    break
                batch = data["data"]
                all_rows.extend(batch)
                remaining -= len(batch)
                if len(batch) < fetch:
                    break  # no more data
                after = batch[-1][0]  # oldest timestamp for next page

            if not all_rows:
                return None
            rows = [
                {
                    "timestamp": pd.to_datetime(int(c[0]), unit="ms"),
                    "open": float(c[1]), "high": float(c[2]),
                    "low": float(c[3]), "close": float(c[4]),
                    "volume": float(c[5]),
                }
                for c in all_rows
            ]
            df = pd.DataFrame(rows).set_index("timestamp")
            df = df.sort_index()
            return df[["open", "high", "low", "close", "volume"]]
        except Exception:
            return None

    def fetch_ohlcv(self, pair: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
        # For large historical requests (backtest), use OKX first (better history)
        if limit > 300:
            df = self._fetch_okx(pair, interval, limit)
            if df is not None and len(df) >= 50:
                self.last_data_source = "OKX+CoinDCX"
                self._patch_live_price(df, pair)
                return df

        # Normal: KuCoin first
        df = self._fetch_kucoin(pair, interval, limit)
        if df is not None and len(df) >= 50:
            self.last_data_source = "KuCoin+CoinDCX"
            self._patch_live_price(df, pair)
            return df

        df = self._fetch_okx(pair, interval, limit)
        if df is not None and len(df) >= 50:
            self.last_data_source = "OKX+CoinDCX"
            self._patch_live_price(df, pair)
            return df

        self.last_data_source = "SAMPLE (no live data!)"
        return self.generate_sample_data(limit)

    def fetch_multi_timeframe(self, pair: str, timeframes: list | None = None, limit: int = 200) -> dict:
        if timeframes is None:
            timeframes = ["5m", "15m", "1h", "4h"]
        result = {}
        for tf in timeframes:
            try:
                result[tf] = self.fetch_ohlcv(pair, tf, limit)
            except Exception:
                continue
        return result

    def get_all_tickers(self) -> dict:
        # Return the richer live payload when available.
        if _live_tickers:
            return dict(_live_tickers)
        # Fallback: fetch once
        try:
            r = requests.get(
                "https://api.coindcx.com/exchange/ticker",
                proxies=PROXIES, timeout=8,
            )
            tickers = {}
            for t in r.json():
                if "market" not in t or "last_price" not in t:
                    continue
                market = t["market"]
                tickers[market] = {
                    **t,
                    "last_price": float(t["last_price"]),
                    "volume": float(t.get("volume", 0) or 0),
                }
            return tickers
        except Exception:
            return {}

    def get_market_universe(
        self,
        quote_suffixes: tuple[str, ...] = ("USDT", "INR", "BTC"),
    ) -> list[dict]:
        """Return all known markets ordered by 24h volume, highest first."""
        tickers = self.get_all_tickers()
        universe: list[dict] = []

        for market, ticker in tickers.items():
            if quote_suffixes and not market.endswith(quote_suffixes):
                continue
            volume = float(ticker.get("volume", 0) or 0)
            universe.append({
                "market": market,
                "last_price": float(ticker.get("last_price", 0) or 0),
                "volume": volume,
            })

        universe.sort(key=lambda item: item["volume"], reverse=True)
        return universe

    def generate_sample_data(self, periods: int = 500, start_price: float = 50000.0) -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=periods, freq="15min")
        prices = [start_price]
        trend = 0.0
        for i in range(1, periods):
            if i % 50 == 0:
                trend = np.random.choice([-0.002, -0.001, 0, 0.001, 0.002])
            prices.append(prices[-1] * (1 + trend + np.random.normal(0, 0.008)))
        v = [abs(np.random.normal(0, 0.006)) for _ in prices]
        highs = [p * (1 + x) for p, x in zip(prices, v)]
        lows = [p * (1 - abs(np.random.normal(0, 0.006))) for p in prices]
        closes = [lo + (hi - lo) * np.random.beta(2, 2) for hi, lo in zip(highs, lows)]
        df = pd.DataFrame({
            "open": prices, "high": highs, "low": lows,
            "close": closes, "volume": np.random.lognormal(10, 1.5, periods),
        }, index=dates)
        df.index.name = "timestamp"
        return df
