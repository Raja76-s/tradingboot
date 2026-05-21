#!/usr/bin/env python3
"""
CryptoTrader Pro - Main Entry Point.

Usage:
    python main.py scan              # Scan all pairs once
    python main.py scan BTCUSDT      # Scan specific pair
    python main.py signals           # Run continuous signal generator
    python main.py backtest          # Run backtest on historical data
    python main.py paper             # Start paper trading (spot)
    python main.py futures           # Start futures paper trading (leverage)
    python main.py live              # Start live trading (requires API keys)
    python main.py dashboard         # Start web dashboard
    python main.py coins             # Show per-coin recommendations
    python main.py telegram-test     # Test Telegram connection
"""

import sys

from colorama import Fore, Style
from colorama import init as colorama_init

from config.settings import get_config
from core.backtester import BacktestConfig, Backtester
from core.data_fetcher import DataFetcher
from core.futures_trader import FuturesConfig, FuturesPaperTrader
from core.signal_generator import SignalGenerator
from core.trader import LiveTrader, PaperTrader
from dashboard.app import run_dashboard
from notifications.telegram_bot import TelegramNotifier
from strategies.coin_strategies import print_coin_recommendations

colorama_init()


def cmd_scan(pair: str | None = None) -> None:
    config = get_config()
    generator = SignalGenerator(config)

    if pair:
        print(f"\nScanning {pair}...")
        signal = generator.scan_pair(pair.upper())
        generator._print_signal(signal)
        print(f"\n{signal.summary}")
    else:
        print("\nScanning all pairs...")
        signals = generator.scan_all_pairs()
        for signal in signals:
            generator._print_signal(signal)
        print(f"\n{Fore.CYAN}Scanned {len(signals)} pairs.{Style.RESET_ALL}")


def cmd_signals() -> None:
    config = get_config()
    generator = SignalGenerator(config)

    notifier = None
    if config.telegram.enabled:
        notifier = TelegramNotifier(config.telegram)
        print(f"{Fore.GREEN}Telegram notifications enabled.{Style.RESET_ALL}")

    def on_signal(signals: list) -> None:
        if notifier:
            for signal in signals:
                notifier.send_signal(signal)

    generator.run_continuous(interval_seconds=60, callback=on_signal)


def cmd_backtest(pair: str = "BTCUSDT") -> None:
    config = get_config()
    fetcher = DataFetcher(config)
    backtester = Backtester(config)

    print(f"\nFetching historical data for {pair}...")
    try:
        df = fetcher.fetch_ohlcv(pair, "1h", 2000)
        if len(df) < 100:
            print("Insufficient live data. Using sample data for demonstration.")
            df = fetcher.generate_sample_data(500)
    except Exception:
        print("Could not fetch live data. Using sample data for demonstration.")
        df = fetcher.generate_sample_data(500)

    print(f"Data points: {len(df)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    bt_config = BacktestConfig(
        pair=pair,
        timeframe="15m",
        initial_balance=config.trading.initial_balance,
        min_confidence=65,
    )

    print("\nRunning backtest...")
    result = backtester.run(df, bt_config)
    print(backtester.print_results(result))

    # Run optimization
    print(f"\n{Fore.CYAN}Running optimization across confidence levels...{Style.RESET_ALL}")
    opt_results = backtester.run_optimization(df, pair)

    print(f"\n{'Confidence':<12} {'Trades':<8} {'Win Rate':<10} {'P&L':<12} {'Return':<10}")
    print("-" * 52)
    for r in opt_results:
        s = r.stats
        print(
            f"{r.config.min_confidence:<12} "
            f"{s['total_trades']:<8} "
            f"{s['win_rate']:<10.1f}% "
            f"{s['total_pnl']:<12,.2f} "
            f"{s['total_pnl_pct']:<10.2f}%"
        )


def cmd_paper() -> None:
    config = get_config()
    config.trading.mode = "paper"

    notifier = None
    if config.telegram.enabled:
        notifier = TelegramNotifier(config.telegram)

    trader = PaperTrader(config)

    def on_trade(trade) -> None:
        if notifier:
            notifier.send_trade_closed(
                pair=trade.pair,
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                pnl=trade.pnl,
                pnl_pct=trade.pnl_pct,
                reason=trade.exit_reason,
            )

    trader.run(interval_seconds=60, on_trade=on_trade)


def cmd_futures() -> None:
    config = get_config()
    futures_config = FuturesConfig(
        leverage=5,
        max_leverage=10,
        margin_type="isolated",
        max_margin_usage_pct=50.0,
    )

    print(f"\n{Fore.YELLOW}Futures Trading Configuration:{Style.RESET_ALL}")
    print(f"  Default Leverage: {futures_config.leverage}x")
    print(f"  Max Leverage: {futures_config.max_leverage}x")
    print(f"  Margin Type: {futures_config.margin_type}")

    if len(sys.argv) > 2:
        try:
            leverage = int(sys.argv[2])
            futures_config.leverage = min(leverage, futures_config.max_leverage)
            print(f"  Using leverage: {futures_config.leverage}x")
        except ValueError:
            pass

    trader = FuturesPaperTrader(config, futures_config)
    trader.run(interval_seconds=60)


def cmd_live() -> None:
    config = get_config()

    if not config.coindcx.api_key or not config.coindcx.api_secret:
        print(f"{Fore.RED}Error: CoinDCX API keys not set!{Style.RESET_ALL}")
        print("Set COINDCX_API_KEY and COINDCX_API_SECRET environment variables.")
        print("\nTo get API keys:")
        print("1. Go to CoinDCX → Settings → API Management")
        print("2. Create a new API key")
        print("3. Copy the key and secret")
        return

    print(f"\n{Fore.RED}WARNING: This will trade with REAL MONEY!{Style.RESET_ALL}")
    confirm = input("Type 'YES I UNDERSTAND' to continue: ")
    if confirm != "YES I UNDERSTAND":
        print("Aborted.")
        return

    trader = LiveTrader(config)
    trader.run()


def cmd_dashboard() -> None:
    config = get_config()
    print(f"\n{Fore.CYAN}Starting dashboard at http://localhost:{config.dashboard.port}{Style.RESET_ALL}")
    run_dashboard(
        host=config.dashboard.host,
        port=config.dashboard.port,
        debug=config.dashboard.debug,
    )


def cmd_coins() -> None:
    print(print_coin_recommendations())


def cmd_telegram_test() -> None:
    config = get_config()
    if not config.telegram.enabled:
        print(f"{Fore.RED}Telegram not configured.{Style.RESET_ALL}")
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")
        print("\nSetup steps:")
        print("1. Message @BotFather on Telegram to create a bot")
        print("2. Copy the bot token")
        print("3. Message @userinfobot to get your chat ID")
        return

    notifier = TelegramNotifier(config.telegram)
    if notifier.test_connection():
        print(f"{Fore.GREEN}Telegram connection successful!{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Telegram connection failed.{Style.RESET_ALL}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    commands = {
        "scan": lambda: cmd_scan(sys.argv[2] if len(sys.argv) > 2 else None),
        "signals": cmd_signals,
        "backtest": lambda: cmd_backtest(sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT"),
        "paper": cmd_paper,
        "futures": cmd_futures,
        "live": cmd_live,
        "dashboard": cmd_dashboard,
        "coins": cmd_coins,
        "telegram-test": cmd_telegram_test,
    }

    handler = commands.get(command)
    if handler:
        handler()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
