# CryptoTrader Pro

A comprehensive crypto trading system with 30+ technical indicators, candlestick pattern recognition, backtesting, real-time signal generation, paper/live trading, and Telegram notifications.

> **Risk Warning:** Trading cryptocurrency involves significant risk. No trading system can guarantee profits. Only trade with money you can afford to lose. Always start with paper trading first.

## Features

### Technical Indicators (30+)

**Trend Indicators:**
- EMA (9, 21, 50, 200), SMA, WMA, DEMA, TEMA
- VWAP (Volume Weighted Average Price)
- Supertrend
- Ichimoku Cloud (Tenkan, Kijun, Senkou A/B, Chikou)
- Parabolic SAR
- ADX (Average Directional Index)

**Momentum Indicators:**
- RSI (with divergence detection)
- Stochastic RSI
- MACD (with histogram analysis)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- MFI (Money Flow Index)
- ROC (Rate of Change)
- Awesome Oscillator
- Ultimate Oscillator
- TSI (True Strength Index)

**Volatility Indicators:**
- Bollinger Bands (with %B and bandwidth)
- ATR (Average True Range)
- Keltner Channels
- Donchian Channels
- BB Squeeze detection (Bollinger inside Keltner)
- Historical Volatility
- Chaikin Volatility

**Volume Indicators:**
- OBV (On-Balance Volume)
- CMF (Chaikin Money Flow)
- Accumulation/Distribution
- Volume Profile
- Force Index
- Ease of Movement
- Volume RSI

### Candlestick Patterns (20+)

**Single Candle:** Doji, Dragonfly Doji, Gravestone Doji, Hammer, Inverted Hammer, Shooting Star, Hanging Man, Spinning Top, Marubozu

**Double Candle:** Bullish/Bearish Engulfing, Bullish/Bearish Harami, Piercing Line, Dark Cloud Cover, Tweezer Top/Bottom

**Triple Candle:** Morning Star, Evening Star, Three White Soldiers, Three Black Crows, Three Inside Up/Down, Three Outside Up/Down

**Chart Patterns:** Double Top/Bottom, Head and Shoulders, Inverse Head and Shoulders

### Confidence Scoring Engine
- Aggregates all indicators into a 0-100 confidence score
- Weighted scoring based on indicator reliability
- Multi-timeframe analysis (1m, 5m, 15m, 1h, 4h, 1d)
- Agreement bonus when multiple timeframes confirm

### Risk Management
- Max 1-2% risk per trade
- Automatic stop-loss on every trade
- Max daily loss limit (5%)
- Trailing stop-loss
- Position sizing based on account balance
- Risk/Reward ratio filtering

### Futures Trading
- Long and Short positions (profit from both price up and down)
- Configurable leverage (2x to 10x)
- Isolated margin mode
- Liquidation price calculation
- Margin management (max 50% margin usage)
- ROE (Return on Equity) tracking
- Paper trading mode for futures

### Per-Coin Strategy Recommendations
- Optimized parameters for BTC, ETH, SOL, XRP, ADA, DOT, MATIC, BNB, DOGE, AVAX, Gold
- Volatility-based timeframe recommendations
- Best indicators per coin

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/Raja7380/TradingBoot.git
cd TradingBoot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env with your API keys (optional for paper trading)
```

### Commands

```bash
# Scan all pairs for trading signals
python main.py scan

# Scan a specific pair
python main.py scan BTCUSDT

# Run continuous signal generator
python main.py signals

# Backtest strategies on historical data
python main.py backtest
python main.py backtest ETHUSDT

# Start paper trading - spot (simulated, no real money)
python main.py paper

# Start futures paper trading with leverage
python main.py futures           # Default 5x leverage
python main.py futures 10        # Custom 10x leverage

# Start live trading (requires CoinDCX API keys)
python main.py live

# Launch web dashboard
python main.py dashboard

# View per-coin recommendations
python main.py coins

# Test Telegram connection
python main.py telegram-test
```

### CoinDCX API Setup

1. Go to [CoinDCX](https://coindcx.com) → Settings → API Management
2. Create a new API key
3. Copy the API Key and API Secret
4. Add them to your `.env` file:
   ```
   COINDCX_API_KEY=your_key
   COINDCX_API_SECRET=your_secret
   ```

### Telegram Setup

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the instructions
3. Copy the bot token
4. Message [@userinfobot](https://t.me/userinfobot) to get your chat ID
5. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```

## Recommended Workflow

```
Step 1: python main.py coins          → See which strategy works best per coin
Step 2: python main.py backtest       → Test strategy on historical data
Step 3: python main.py signals        → Watch signals in real-time for a few days
Step 4: python main.py paper          → Paper trade for 1-2 weeks
Step 5: python main.py live           → Go live with small amounts (only after paper trading proves profitable)
```

## Project Structure

```
TradingBoot/
├── config/
│   └── settings.py              # All configuration (API keys, risk params, strategy params)
├── core/
│   ├── backtester.py            # Backtesting engine
│   ├── data_fetcher.py          # Market data from CoinDCX/Binance
│   ├── indicators.py            # 30+ technical indicators
│   ├── patterns.py              # 20+ candlestick patterns
│   ├── risk_manager.py          # Risk management & position sizing
│   ├── scoring_engine.py        # Multi-indicator confidence scoring
│   ├── signal_generator.py      # Real-time signal generation
│   ├── trader.py                # Paper & live spot trading bots
│   └── futures_trader.py        # Futures trading with leverage
├── strategies/
│   └── coin_strategies.py       # Per-coin strategy recommendations
├── notifications/
│   └── telegram_bot.py          # Telegram notifications
├── dashboard/
│   ├── app.py                   # Flask web dashboard
│   └── templates/
│       └── index.html           # Dashboard UI
├── utils/
│   └── helpers.py               # Utility functions
├── main.py                      # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

## Important Notes

- **Always start with paper trading** — never go live until you've tested for weeks
- **No system guarantees profits** — even the best strategies have losing streaks
- **Risk management is key** — small consistent gains beat large risky bets
- **Keep position sizes small** — with ₹20K, risk only ₹200-400 per trade
- The system falls back to Binance public API if CoinDCX data is unavailable
