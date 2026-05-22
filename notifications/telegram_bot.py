"""
Telegram Notification Module.

Sends trading signals to a Telegram chat/channel.
Setup:
  1. Create a bot via @BotFather on Telegram
  2. Get the bot token
  3. Get your chat ID (message @userinfobot)
  4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars
"""

import os
import requests

from config.settings import TelegramConfig
from core.scoring_engine import TradeSignal
from core.signal_generator import SignalGenerator


class TelegramNotifier:
    """Sends trading signals via Telegram."""

    def __init__(self, config: TelegramConfig) -> None:
        self.config = config
        self.base_url = f"https://api.telegram.org/bot{config.bot_token}"

    def send_message(self, text: str) -> bool:
        if not self.config.enabled:
            return False

        # PythonAnywhere free plan requires proxy for external HTTPS
        proxies = {}
        pa_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
        if pa_proxy:
            proxies = {"http": pa_proxy, "https": pa_proxy}
        else:
            # PythonAnywhere default proxy
            proxies = {
                "http": "http://proxy.server:3128",
                "https": "http://proxy.server:3128",
            }

        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.config.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                },
                proxies=proxies,
                timeout=15,
            )
            if resp.status_code != 200:
                print(f"Telegram error {resp.status_code}: {resp.text}")
            return resp.status_code == 200
        except Exception as e:
            print(f"Telegram send failed: {e}")
            return False

    def send_signal(self, signal: TradeSignal) -> bool:
        message = SignalGenerator.format_telegram_message(signal)
        return self.send_message(message)

    def send_trade_opened(
        self,
        pair: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        confidence: int,
    ) -> bool:
        emoji = "🟢" if side == "BUY" else "🔴"
        msg = (
            f"{emoji} <b>Trade Opened — {pair}</b>\n\n"
            f"Side: {side}\n"
            f"Entry: {entry_price:,.2f}\n"
            f"Quantity: {quantity:.6f}\n"
            f"Stop-Loss: {stop_loss:,.2f}\n"
            f"Take-Profit: {take_profit:,.2f}\n"
            f"Confidence: {confidence}/100"
        )
        return self.send_message(msg)

    def send_trade_closed(
        self,
        pair: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ) -> bool:
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"{emoji} <b>Trade Closed — {pair}</b>\n\n"
            f"Side: {side}\n"
            f"Entry: {entry_price:,.2f} → Exit: {exit_price:,.2f}\n"
            f"P&L: {pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
            f"Reason: {reason}"
        )
        return self.send_message(msg)

    def send_daily_summary(
        self,
        stats: dict[str, float | int],
    ) -> bool:
        msg = (
            "📊 <b>Daily Trading Summary</b>\n\n"
            f"Total Trades: {stats['total_trades']}\n"
            f"Win Rate: {stats['win_rate']:.1f}%\n"
            f"Total P&L: {stats['total_pnl']:+,.2f}\n"
            f"Balance: {stats['current_balance']:,.2f}\n"
            f"Max Drawdown: {stats['drawdown_pct']:.2f}%"
        )
        return self.send_message(msg)

    def send_alert(self, title: str, message: str) -> bool:
        msg = f"⚠️ <b>{title}</b>\n\n{message}"
        return self.send_message(msg)

    def test_connection(self) -> bool:
        return self.send_message("🤖 CryptoTrader Pro bot connected successfully!")
