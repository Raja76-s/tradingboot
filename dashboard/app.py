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
    """Fetch real-time prices from CoinDCX via DataFetcher (proxy already configured)."""
    try:
        tickers = data_fetcher.get_all_tickers()
        symbols = [p.replace("_", "").upper() for p in TRACKED_PAIRS]
        prices = {
            sym: tickers[sym]["last_price"]
            for sym in symbols
            if sym in tickers and "last_price" in tickers[sym]
        }
        if not prices:
            raise ValueError("No prices found")
        return jsonify({"prices": prices, "source": "CoinDCX", "ts": datetime.now().isoformat()})
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
            market = data_fetcher.get_all_tickers().get(pair.upper(), {})
            signal = signal_generator.scan_pair(
                pair,
                timeframe,
                market_volume=market.get("volume"),
            )
            # Replace entry price with live CoinDCX price
            live = data_fetcher.get_live_price(pair)
            if live:
                signal.entry_price = live
            return jsonify({"signals": [_signal_to_dict(signal)]})
        else:
            signals = []
            for s in signal_generator.scan_all_pairs(timeframe=timeframe):
                try:
                    live = data_fetcher.get_live_price(s.pair)
                    if live and live > 0:
                        s.entry_price = live
                        # Recalculate SL/TP based on live price
                        risk = abs(live - s.stop_loss)
                        if s.action == "BUY":
                            s.stop_loss = round(live - risk, 4)
                            s.take_profit_1 = round(live + risk * 1.5, 4)
                            s.take_profit_2 = round(live + risk * 3.0, 4)
                        elif s.action == "SELL":
                            s.stop_loss = round(live + risk, 4)
                            s.take_profit_1 = round(live - risk * 1.5, 4)
                            s.take_profit_2 = round(live - risk * 3.0, 4)
                        if risk > 0:
                            s.risk_reward_ratio = round(abs(s.take_profit_1 - live) / risk, 2)
                            s.stop_loss_pct = round(risk / live * 100, 2)
                    signals.append(s)
                except Exception:
                    continue
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
        "king_trade_score": signal.king_trade_score,
        "king_trade_signal": signal.king_trade_signal,
        "king_trade_details": signal.king_trade_details,
        "market_volume": getattr(signal, "market_volume", 0.0),
        "volume_rank": getattr(signal, "volume_rank", None),
    }


def run_dashboard(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    app.run(host=host, port=port, debug=debug)
