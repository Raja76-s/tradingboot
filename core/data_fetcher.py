"""
Data fetcher for CoinDCX API and alternative data sources.
Fetches historical OHLCV data and real-time market data.
"""

import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

from config.settings import AppConfig


class CoinDCXClient:
    """Client for CoinDCX REST API."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.api_key = config.coindcx.api_key
        self.api_secret = config.coindcx.api_secret
        self.base_url = config.coindcx.base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _sign_request(self, body: dict[str, Any]) -> dict[str, str]:
        json_body = json.dumps(body, separators=(",", ":"))
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            json_body.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "X-AUTH-APIKEY": self.api_key,
            "X-AUTH-SIGNATURE": signature,
        }

    def get_markets(self) -> list[dict[str, Any]]:
        resp = self.session.get(f"{self.base_url}/exchange/v1/markets")
        resp.raise_for_status()
        return resp.json()

    def get_markets_details(self) -> list[dict[str, Any]]:
        resp = self.session.get(f"{self.base_url}/exchange/v1/markets_details")
        resp.raise_for_status()
        return resp.json()

    def get_ticker(self) -> list[dict[str, Any]]:
        resp = self.session.get(f"{self.base_url}/exchange/ticker")
        resp.raise_for_status()
        return resp.json()

    def get_order_book(self, pair: str) -> dict[str, Any]:
        resp = self.session.get(
            f"{self.base_url}/exchange/v1/books",
            params={"pair": pair},
        )
        resp.raise_for_status()
        return resp.json()

    def get_candles(
        self,
        pair: str,
        interval: str = "1m",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        resp = self.session.get(
            f"{self.base_url}/exchange/v1/candles",
            params={"pair": pair, "interval": interval, "limit": limit},
        )
        resp.raise_for_status()
        return resp.json()

    def get_balances(self) -> list[dict[str, Any]]:
        timestamp = int(time.time() * 1000)
        body = {"timestamp": timestamp}
        headers = self._sign_request(body)
        resp = self.session.post(
            f"{self.base_url}/exchange/v1/users/balances",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    def place_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        price: float | None = None,
        quantity: float = 0.0,
        target_price: float | None = None,
        stop_price: float | None = None,
    ) -> dict[str, Any]:
        timestamp = int(time.time() * 1000)
        body: dict[str, Any] = {
            "side": side,
            "order_type": order_type,
            "market": pair,
            "total_quantity": quantity,
            "timestamp": timestamp,
        }
        if price is not None:
            body["price_per_unit"] = price
        if target_price is not None:
            body["target_price"] = target_price
        if stop_price is not None:
            body["stop_price"] = stop_price

        headers = self._sign_request(body)
        resp = self.session.post(
            f"{self.base_url}/exchange/v1/orders/create",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        timestamp = int(time.time() * 1000)
        body = {"id": order_id, "timestamp": timestamp}
        headers = self._sign_request(body)
        resp = self.session.post(
            f"{self.base_url}/exchange/v1/orders/cancel",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    def get_active_orders(self) -> list[dict[str, Any]]:
        timestamp = int(time.time() * 1000)
        body = {"timestamp": timestamp}
        headers = self._sign_request(body)
        resp = self.session.post(
            f"{self.base_url}/exchange/v1/orders/active_orders",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()


class DataFetcher:
    """Fetches and processes market data from multiple sources."""

    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
        "1M": "1M",
    }

    # CoinDCX uses "B-BTC_USDT" format for spot pairs
    @staticmethod
    def _to_coindcx_pair(pair: str) -> str:
        """Convert BTCUSDT → B-BTC_USDT (CoinDCX spot format)."""
        pair = pair.upper().replace("-", "").replace("_", "")
        for quote in ("USDT", "BTC", "ETH", "INR", "USD"):
            if pair.endswith(quote):
                base = pair[: -len(quote)]
                return f"B-{base}_{quote}"
        return pair

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = CoinDCXClient(config)
        self.last_data_source: str = "unknown"  # track where data came from

    def fetch_ohlcv(
        self,
        pair: str,
        interval: str = "15m",
        limit: int = 500,
    ) -> pd.DataFrame:
        coindcx_pair = self._to_coindcx_pair(pair)
        try:
            raw = self.client.get_candles(
                pair=coindcx_pair,
                interval=self.INTERVAL_MAP.get(interval, interval),
                limit=limit,
            )
            if raw and len(raw) >= 10:
                self.last_data_source = "CoinDCX"
                return self._parse_candles(raw)
        except Exception:
            pass

        df = self._fetch_from_binance(pair, interval, limit)
        return df

    def _parse_candles(self, raw: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(raw)
        column_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "time": "timestamp",
        }
        df = df.rename(columns=column_map)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")
        df = df.sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    def _fetch_from_binance(
        self,
        pair: str,
        interval: str,
        limit: int,
    ) -> pd.DataFrame:
        """Fallback: fetch from public Binance API."""
        symbol = pair.replace("_", "").replace("/", "").upper()

        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                raise ValueError("empty")
            self.last_data_source = "Binance"
        except Exception:
            return self._fetch_from_cryptocompare(symbol, interval, limit)

        df = pd.DataFrame(
            data,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    def fetch_multi_timeframe(
        self,
        pair: str,
        timeframes: list[str] | None = None,
        limit: int = 200,
    ) -> dict[str, pd.DataFrame]:
        if timeframes is None:
            timeframes = self.config.strategy.timeframes
        result: dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            try:
                result[tf] = self.fetch_ohlcv(pair, tf, limit)
            except Exception:
                continue
        return result

    def get_all_tickers(self) -> dict[str, float]:
        try:
            tickers = self.client.get_ticker()
            return {t["market"]: float(t["last_price"]) for t in tickers if "last_price" in t}
        except Exception:
            return {}

    def _fetch_from_cryptocompare(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> pd.DataFrame:
        """Fallback: fetch from CryptoCompare API."""
        base = symbol.replace("USDT", "").replace("USD", "")
        quote = "USDT" if "USDT" in symbol else "USD"

        interval_map = {
            "1m": ("histominute", 1),
            "5m": ("histominute", 5),
            "15m": ("histominute", 15),
            "30m": ("histominute", 30),
            "1h": ("histohour", 1),
            "2h": ("histohour", 2),
            "4h": ("histohour", 4),
            "1d": ("histoday", 1),
        }
        endpoint, aggregate = interval_map.get(interval, ("histominute", 15))

        self.last_data_source = "CryptoCompare"
        url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
        params = {
            "fsym": base,
            "tsym": quote,
            "limit": limit,
            "aggregate": aggregate,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("Response") == "Error" or "Data" not in data.get("Data", {}):
            self.last_data_source = "SAMPLE (no live data!)"
            return self.generate_sample_data(limit)

        records = data["Data"]["Data"]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("timestamp")
        df = df.rename(columns={"volumefrom": "volume"})
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    def generate_sample_data(
        self,
        periods: int = 500,
        start_price: float = 50000.0,
    ) -> pd.DataFrame:
        """Generate realistic volatile sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=periods, freq="15min")

        # Create realistic price movements with trends and reversals
        prices = [start_price]
        trend = 0.0
        for i in range(1, periods):
            # Random walk with momentum and mean reversion
            if i % 50 == 0:
                trend = np.random.choice([-0.002, -0.001, 0, 0.001, 0.002])
            noise = np.random.normal(0, 0.008)
            change = trend + noise
            prices.append(prices[-1] * (1 + change))

        # Add realistic high/low/close variation
        volatility = [abs(np.random.normal(0, 0.006)) for _ in prices]
        highs = [p * (1 + v) for p, v in zip(prices, volatility)]
        lows = [p * (1 - abs(np.random.normal(0, 0.006))) for p in prices]
        closes = []
        for _i, (op, hi, lo) in enumerate(zip(prices, highs, lows)):
            ratio = np.random.beta(2, 2)
            closes.append(lo + (hi - lo) * ratio)

        volumes = np.random.lognormal(10, 1.5, periods)
        # Add volume spikes
        for i in range(0, periods, np.random.randint(20, 50)):
            volumes[i] *= np.random.uniform(3, 8)

        df = pd.DataFrame({
            "open": prices,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }, index=dates)
        df.index.name = "timestamp"
        return df
