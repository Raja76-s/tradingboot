import requests

from config.settings import TelegramConfig
from core.scoring_engine import TradeSignal
from core.signal_generator import SignalGenerator

PROXIES = {"http": "http://proxy.server:3128", "https": "http://proxy.server:3128"}


class TelegramNotifier:

    def __init__(self, config: TelegramConfig) -> None:
        self.config = config
        self.base_url = f"https://api.telegram.org/bot{config.bot_token}"

    def send_message(self, text: str) -> bool:
        if not self.config.enabled:
            return False
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": self.config.chat_id, "text": text, "parse_mode": "HTML"},
                proxies=PROXIES,
                timeout=15,
            )
            if resp.status_code != 200:
                print(f"Telegram error {resp.status_code}: {resp.text}")
            return resp.status_code == 200
        except Exception as e:
            print(f"Telegram send failed: {e}")
            return False

    def send_signal(self, signal: TradeSignal) -> bool:
        return self.send_message(SignalGenerator.format_telegram_message(signal))

    def send_trade_opened(self, pair, side, entry_price, quantity, stop_loss, take_profit, confidence) -> bool:
        emoji = "🟢" if side == "BUY" else "🔴"
        return self.send_message(
            f"{emoji} <b>Trade Opened — {pair}</b>\n\n"
            f"Side: {side}\nEntry: {entry_price:,.4f}\n"
            f"Stop-Loss: {stop_loss:,.4f}\nTake-Profit: {take_profit:,.4f}\n"
            f"Confidence: {confidence}/100"
        )

    def send_trade_closed(self, pair, side, entry_price, exit_price, pnl, pnl_pct, reason) -> bool:
        emoji = "✅" if pnl > 0 else "❌"
        return self.send_message(
            f"{emoji} <b>Trade Closed — {pair}</b>\n\n"
            f"Side: {side}\nEntry: {entry_price:,.4f} → Exit: {exit_price:,.4f}\n"
            f"P&L: {pnl:+,.2f} ({pnl_pct:+.2f}%)\nReason: {reason}"
        )

    def send_daily_summary(self, stats: dict) -> bool:
        return self.send_message(
            f"📊 <b>Daily Summary</b>\n\n"
            f"Trades: {stats['total_trades']}\nWin Rate: {stats['win_rate']:.1f}%\n"
            f"P&L: {stats['total_pnl']:+,.2f}\nBalance: {stats['current_balance']:,.2f}"
        )

    def send_alert(self, title: str, message: str) -> bool:
        return self.send_message(f"⚠️ <b>{title}</b>\n\n{message}")

    def test_connection(self) -> bool:
        return self.send_message(
            "✅ <b>CryptoTrader Pro — Connected!</b>\n\n"
            "🎉 Telegram notifications working!\n"
            "Ab trading signals yahan aayenge."
        )
