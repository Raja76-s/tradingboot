"""
Per-Coin Strategy Recommendations.

Different coins have different volatility profiles, trading volumes,
and behavior patterns. This module recommends the best strategy
parameters for each coin/asset.
"""

from dataclasses import dataclass

from config.settings import StrategyConfig


@dataclass
class CoinProfile:
    symbol: str
    name: str
    category: str  # "large_cap", "mid_cap", "small_cap", "stablecoin", "commodity"
    volatility: str  # "low", "medium", "high", "very_high"
    recommended_timeframe: str
    recommended_confidence: int
    description: str
    best_indicators: list[str]
    risk_per_trade_pct: float
    stop_loss_pct: float
    take_profit_pct: float


COIN_PROFILES: dict[str, CoinProfile] = {
    "BTCUSDT": CoinProfile(
        symbol="BTCUSDT",
        name="Bitcoin",
        category="large_cap",
        volatility="medium",
        recommended_timeframe="1h",
        recommended_confidence=65,
        description="Most stable crypto. Best for trend-following strategies.",
        best_indicators=[
            "EMA Crossover (50/200)", "MACD", "RSI", "Ichimoku Cloud",
            "Supertrend", "Volume Profile",
        ],
        risk_per_trade_pct=1.5,
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
    ),
    "ETHUSDT": CoinProfile(
        symbol="ETHUSDT",
        name="Ethereum",
        category="large_cap",
        volatility="medium",
        recommended_timeframe="1h",
        recommended_confidence=65,
        description="Second largest. Follows BTC but with higher beta.",
        best_indicators=[
            "EMA Crossover (9/21)", "MACD", "Bollinger Bands", "RSI",
            "Stochastic RSI", "OBV",
        ],
        risk_per_trade_pct=1.5,
        stop_loss_pct=2.5,
        take_profit_pct=5.0,
    ),
    "SOLUSDT": CoinProfile(
        symbol="SOLUSDT",
        name="Solana",
        category="large_cap",
        volatility="high",
        recommended_timeframe="15m",
        recommended_confidence=70,
        description="High volatility, strong momentum plays.",
        best_indicators=[
            "RSI", "MACD", "Bollinger Bands", "Volume Spike",
            "Stochastic Oscillator", "ATR",
        ],
        risk_per_trade_pct=1.0,
        stop_loss_pct=3.0,
        take_profit_pct=6.0,
    ),
    "XRPUSDT": CoinProfile(
        symbol="XRPUSDT",
        name="XRP",
        category="large_cap",
        volatility="high",
        recommended_timeframe="15m",
        recommended_confidence=70,
        description="News-driven. Momentum and breakout strategies work best.",
        best_indicators=[
            "Bollinger Bands", "RSI", "MACD", "Donchian Channels",
            "Volume Spike", "CCI",
        ],
        risk_per_trade_pct=1.0,
        stop_loss_pct=3.0,
        take_profit_pct=6.0,
    ),
    "ADAUSDT": CoinProfile(
        symbol="ADAUSDT",
        name="Cardano",
        category="mid_cap",
        volatility="high",
        recommended_timeframe="15m",
        recommended_confidence=70,
        description="Range-bound often. Mean reversion works well.",
        best_indicators=[
            "Bollinger Bands", "RSI", "Stochastic Oscillator",
            "Keltner Channels", "CCI", "MFI",
        ],
        risk_per_trade_pct=1.0,
        stop_loss_pct=3.5,
        take_profit_pct=7.0,
    ),
    "DOTUSDT": CoinProfile(
        symbol="DOTUSDT",
        name="Polkadot",
        category="mid_cap",
        volatility="high",
        recommended_timeframe="15m",
        recommended_confidence=70,
        description="Follows major market trends with lag.",
        best_indicators=[
            "EMA Crossover", "Supertrend", "RSI", "MACD",
            "Parabolic SAR", "ADX",
        ],
        risk_per_trade_pct=1.0,
        stop_loss_pct=3.0,
        take_profit_pct=6.0,
    ),
    "MATICUSDT": CoinProfile(
        symbol="MATICUSDT",
        name="Polygon",
        category="mid_cap",
        volatility="very_high",
        recommended_timeframe="5m",
        recommended_confidence=75,
        description="Very volatile. Quick scalping strategies work.",
        best_indicators=[
            "RSI", "Stochastic RSI", "Bollinger Bands", "MACD",
            "Volume Spike", "Williams %R",
        ],
        risk_per_trade_pct=0.8,
        stop_loss_pct=4.0,
        take_profit_pct=8.0,
    ),
    "BNBUSDT": CoinProfile(
        symbol="BNBUSDT",
        name="BNB",
        category="large_cap",
        volatility="medium",
        recommended_timeframe="1h",
        recommended_confidence=65,
        description="Exchange token. Relatively stable, trend-following.",
        best_indicators=[
            "EMA Crossover (50/200)", "MACD", "RSI", "Ichimoku Cloud",
            "Supertrend", "OBV",
        ],
        risk_per_trade_pct=1.5,
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
    ),
    "DOGEUSDT": CoinProfile(
        symbol="DOGEUSDT",
        name="Dogecoin",
        category="mid_cap",
        volatility="very_high",
        recommended_timeframe="5m",
        recommended_confidence=75,
        description="Meme coin. Very volatile, news/social media driven.",
        best_indicators=[
            "RSI", "Volume Spike", "Bollinger Bands", "Stochastic RSI",
            "MACD", "Force Index",
        ],
        risk_per_trade_pct=0.5,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
    ),
    "AVAXUSDT": CoinProfile(
        symbol="AVAXUSDT",
        name="Avalanche",
        category="mid_cap",
        volatility="high",
        recommended_timeframe="15m",
        recommended_confidence=70,
        description="L1 blockchain. Momentum and breakout trading.",
        best_indicators=[
            "MACD", "RSI", "Supertrend", "Bollinger Bands",
            "ADX", "Volume Spike",
        ],
        risk_per_trade_pct=1.0,
        stop_loss_pct=3.0,
        take_profit_pct=6.0,
    ),
    # Gold (via tokenized gold or gold pairs)
    "XAUUSDT": CoinProfile(
        symbol="XAUUSDT",
        name="Gold",
        category="commodity",
        volatility="low",
        recommended_timeframe="4h",
        recommended_confidence=60,
        description="Safe haven. Slow trends, mean reversion at extremes.",
        best_indicators=[
            "EMA Crossover (50/200)", "Ichimoku Cloud", "RSI",
            "Bollinger Bands", "Parabolic SAR", "ADX",
        ],
        risk_per_trade_pct=1.5,
        stop_loss_pct=1.0,
        take_profit_pct=2.0,
    ),
    # Silver
    "XAGUSDT": CoinProfile(
        symbol="XAGUSDT",
        name="Silver",
        category="commodity",
        volatility="medium",
        recommended_timeframe="4h",
        recommended_confidence=60,
        description="More volatile than gold. Good for swing trading.",
        best_indicators=[
            "EMA Crossover (50/200)", "Bollinger Bands", "RSI",
            "MACD", "Parabolic SAR", "Keltner Channels",
        ],
        risk_per_trade_pct=1.2,
        stop_loss_pct=1.5,
        take_profit_pct=3.0,
    ),
}


def get_coin_profile(symbol: str) -> CoinProfile | None:
    return COIN_PROFILES.get(symbol.upper())


def get_recommended_config(symbol: str, base_config: StrategyConfig) -> StrategyConfig:
    """Get strategy config optimized for a specific coin."""
    profile = get_coin_profile(symbol)
    if profile is None:
        return base_config

    import copy
    config = copy.deepcopy(base_config)
    config.default_timeframe = profile.recommended_timeframe
    config.min_confidence_score = profile.recommended_confidence
    return config


def print_coin_recommendations() -> str:
    lines = [
        "=" * 70,
        "RECOMMENDED STRATEGIES PER COIN",
        "=" * 70,
    ]

    for symbol, profile in COIN_PROFILES.items():
        lines.extend([
            f"\n{profile.name} ({symbol})",
            f"  Category: {profile.category} | Volatility: {profile.volatility}",
            f"  Best Timeframe: {profile.recommended_timeframe}",
            f"  Min Confidence: {profile.recommended_confidence}/100",
            f"  Risk/Trade: {profile.risk_per_trade_pct}%",
            f"  Stop-Loss: {profile.stop_loss_pct}% | Take-Profit: {profile.take_profit_pct}%",
            f"  Best Indicators: {', '.join(profile.best_indicators)}",
            f"  Notes: {profile.description}",
        ])

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
