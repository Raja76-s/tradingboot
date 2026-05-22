"""
Configuration settings for CryptoTrader Pro.
All settings can be overridden via environment variables or .env file.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class CoinDCXConfig:
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://api.coindcx.com"
    ws_url: str = "wss://stream.coindcx.com"

    def __post_init__(self) -> None:
        self.api_key = os.getenv("COINDCX_API_KEY", self.api_key)
        self.api_secret = os.getenv("COINDCX_API_SECRET", self.api_secret)


@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = False

    def __post_init__(self) -> None:
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", self.bot_token)
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", self.chat_id)
        self.enabled = bool(self.bot_token and self.chat_id)


@dataclass
class RiskConfig:
    max_risk_per_trade_pct: float = 1.5
    max_daily_loss_pct: float = 5.0
    max_open_positions: int = 3
    default_stop_loss_pct: float = 2.0
    default_take_profit_pct: float = 4.0
    trailing_stop_pct: float = 1.5
    min_risk_reward_ratio: float = 1.5

    def __post_init__(self) -> None:
        self.max_risk_per_trade_pct = float(
            os.getenv("MAX_RISK_PER_TRADE_PCT", self.max_risk_per_trade_pct)
        )
        self.max_daily_loss_pct = float(
            os.getenv("MAX_DAILY_LOSS_PCT", self.max_daily_loss_pct)
        )


@dataclass
class StrategyConfig:
    min_confidence_score: int = 62  # Balanced — fires on good setups without too much noise
    timeframes: list[str] = field(
        default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"]
    )
    default_timeframe: str = "15m"
    lookback_periods: int = 200

    # EMA settings
    ema_fast: int = 9
    ema_medium: int = 21
    ema_slow: int = 50
    ema_very_slow: int = 200

    # RSI settings
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70

    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std_dev: float = 2.0

    # Stochastic
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3

    # ADX
    adx_period: int = 14
    adx_strong_trend: int = 25

    # ATR
    atr_period: int = 14

    # Supertrend
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0

    # Ichimoku
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou_b: int = 52

    # CCI
    cci_period: int = 20
    cci_overbought: int = 100
    cci_oversold: int = -100

    # Williams %R
    williams_period: int = 14

    # MFI
    mfi_period: int = 14
    mfi_overbought: int = 80
    mfi_oversold: int = 20


@dataclass
class TradingConfig:
    mode: str = "paper"  # "paper" or "live"
    trading_pairs: list[str] = field(
        default_factory=lambda: [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOTUSDT", "MATICUSDT", "BNBUSDT",
            "XAUUSDT", "XAGUSDT",
        ]
    )
    initial_balance: float = 20000.0
    currency: str = "INR"

    def __post_init__(self) -> None:
        self.mode = os.getenv("TRADING_MODE", self.mode)
        self.initial_balance = float(
            os.getenv("INITIAL_BALANCE", self.initial_balance)
        )


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False


@dataclass
class AppConfig:
    coindcx: CoinDCXConfig = field(default_factory=CoinDCXConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


def get_config() -> AppConfig:
    return AppConfig()
