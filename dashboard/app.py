"""
Flask Web Dashboard for CryptoTrader Pro.

Displays:
- Current signals for all pairs
- Open positions
- Trade history
- Performance stats
- Equity curve chart
"""

from datetime import datetime

from flask import Flask, jsonify, render_template, request

from config.settings import get_config
from core.data_fetcher import DataFetcher
from core.scoring_engine import TradeSignal
from core.signal_generator import SignalGenerator
from strategies.coin_strategies import COIN_PROFILES

app = Flask(__name__)

config = get_config()
signal_generator = SignalGenerator(config)
data_fetcher = DataFetcher(config)

# In-memory state
trade_history: list[dict] = []
paper_stats: dict = {}


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
        "entry_price": round(signal.entry_price, 2),
        "stop_loss": round(signal.stop_loss, 2),
        "take_profit_1": round(signal.take_profit_1, 2),
        "take_profit_2": round(signal.take_profit_2, 2),
        "risk_reward": signal.risk_reward_ratio,
        "buy_count": signal.buy_count,
        "sell_count": signal.sell_count,
        "neutral_count": signal.neutral_count,
        "timeframe": signal.timeframe,
        "timestamp": signal.timestamp.isoformat(),
        "summary": signal.summary,
    }


def run_dashboard(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    app.run(host=host, port=port, debug=debug)
