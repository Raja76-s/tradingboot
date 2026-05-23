"""
Flask Web Dashboard for CryptoTrader Pro.

Displays:
- Current signals for all pairs
- Open positions
- Trade history
- Performance stats
- Equity curve chart
"""

import requests
from datetime import datetime

from flask import Flask, jsonify, render_template, request

from config.settings import get_config
from core.data_fetcher import DataFetcher
from core.scoring_engine import TradeSignal
from core.signal_generator import SignalGenerator
from notifications.telegram_bot import TelegramNotifier
from strategies.coin_strategies import COIN_PROFILES

app = Flask(__name__)

config = get_config()
signal_generator = SignalGenerator(config)
data_fetcher = DataFetcher(config)

# In-memory state
trade_history: list[dict] = []
paper_stats: dict = {}

TRACKED_PAIRS = config.trading.trading_pairs


@app.route("/api/live-prices")
def live_prices():
    """Fetch real-time prices from Binance ticker (no API key needed)."""
    symbols = [p.replace("_", "").upper() for p in TRACKED_PAIRS]
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            timeout=5,
        )
        resp.raise_for_status()
        all_prices = {item["symbol"]: float(item["price"]) for item in resp.json()}
        prices = {
            sym: all_prices[sym]
            for sym in symbols
            if sym in all_prices
        }
        return jsonify({"prices": prices, "source": "Binance", "ts": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"prices": {}, "source": "error", "error": str(e)})


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/scan", methods=["GET"])
def scan_markets():
    pair = request.args.get("pair")
    timeframe = request.args.get("timeframe", config.strategy.default_timeframe)

    try:
        if pair:
            signal = signal_generator.scan_pair(pair, timeframe)
            return jsonify({"signals": [_signal_to_dict(signal)]})
        else:
            signals = signal_generator.scan_all_pairs()
            return jsonify({
                "signals": [_signal_to_dict(s) for s in signals],
                "timestamp": datetime.now().isoformat(),
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/signal/<pair>")
def get_signal(pair: str):
    timeframe = request.args.get("timeframe", config.strategy.default_timeframe)
    try:
        signal = signal_generator.scan_pair(pair.upper(), timeframe)
        result = _signal_to_dict(signal)
        result["indicators"] = [
            {
                "name": i.name,
                "signal": i.signal.value,
                "value": round(float(i.value), 4) if i.value else 0,
                "weight": float(i.weight),
                "details": i.details,
            }
            for i in signal.indicator_signals
        ]
        result["patterns"] = [
            {
                "name": p.name,
                "signal": p.signal.value,
                "confidence": float(p.confidence),
                "description": p.description,
            }
            for p in signal.pattern_signals
        ]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/coins")
def get_coins():
    coins = []
    for symbol, profile in COIN_PROFILES.items():
        coins.append({
            "symbol": profile.symbol,
            "name": profile.name,
            "category": profile.category,
            "volatility": profile.volatility,
            "recommended_timeframe": profile.recommended_timeframe,
            "recommended_confidence": profile.recommended_confidence,
            "best_indicators": profile.best_indicators,
            "risk_per_trade": profile.risk_per_trade_pct,
            "stop_loss": profile.stop_loss_pct,
            "take_profit": profile.take_profit_pct,
            "description": profile.description,
        })
    return jsonify({"coins": coins})


@app.route("/api/data-source")
def data_source():
    """Show where market data is actually coming from."""
    src = data_fetcher.last_data_source
    ok = src in ("CoinDCX", "Binance")
    return jsonify({
        "source": src if src != "unknown" else "Not fetched yet",
        "is_live": ok,
        "warning": None if ok else "Using FAKE sample data! Signals are not real.",
    })


@app.route("/api/telegram/status")
def telegram_status():
    cfg = config.telegram
    return jsonify({
        "enabled": cfg.enabled,
        "bot_token_set": bool(cfg.bot_token and cfg.bot_token != "your_bot_token_here"),
        "chat_id_set": bool(cfg.chat_id and cfg.chat_id != "your_chat_id_here"),
        "bot_token_preview": (cfg.bot_token[:10] + "...") if cfg.bot_token else "",
        "chat_id": cfg.chat_id if cfg.chat_id else "",
    })


@app.route("/api/telegram/test", methods=["POST"])
def telegram_test():
    cfg = config.telegram
    if not cfg.bot_token or cfg.bot_token == "your_bot_token_here":
        return jsonify({"success": False, "error": "TELEGRAM_BOT_TOKEN not set in .env file"})
    if not cfg.chat_id or cfg.chat_id == "your_chat_id_here":
        return jsonify({"success": False, "error": "TELEGRAM_CHAT_ID not set in .env file"})

    notifier = TelegramNotifier(cfg)
    ok = notifier.send_message(
        "✅ <b>CryptoTrader Pro — Test Message</b>\n\n"
        "🎉 Telegram notifications are working!\n"
        "You will now receive trading signals here."
    )
    if ok:
        return jsonify({"success": True, "message": "Test message sent! Check your Telegram."})
    else:
        return jsonify({"success": False, "error": "Message failed. Check your bot token and chat ID."})


@app.route("/api/stats")
def get_stats():
    return jsonify(paper_stats)


@app.route("/api/history")
def get_history():
    return jsonify({"trades": trade_history})


def _signal_to_dict(signal: TradeSignal) -> dict:
    return {
        "pair": signal.pair,
        "action": signal.action,
        "confidence": signal.confidence,
        "quality_score": signal.quality_score,
        "grade": signal.grade,
        "stop_loss_pct": signal.stop_loss_pct,
        "entry_price": round(signal.entry_price, 4),
        "stop_loss": round(signal.stop_loss, 4),
        "take_profit_1": round(signal.take_profit_1, 4),
        "take_profit_2": round(signal.take_profit_2, 4),
        "risk_reward": signal.risk_reward_ratio,
        "buy_count": signal.buy_count,
        "sell_count": signal.sell_count,
        "neutral_count": signal.neutral_count,
        "timeframe": signal.timeframe,
        "timestamp": signal.timestamp.isoformat(),
        "summary": signal.summary,
        "smc_signals": [
            {"name": i.name, "signal": i.signal.value, "details": i.details}
            for i in signal.indicator_signals if i.name.startswith("SMC:")
        ],
    }


def run_dashboard(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    app.run(host=host, port=port, debug=debug)
