#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ta_signal_agent_live_binary.py

Binär TA-agent (0% / 100%) med CCXT-broker och dry-run-läge.
- Läser CSV (OHLCV), räknar signal (BUY/SELL/HOLD)
- Hämtar/simulerar portfölj
- Handlar till 0% eller 100% allokering
- Loggar affärer
- Kan köras i dry-run-läge (ingen riktig handel)
"""

import os
import json
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
import numpy as np
import ccxt
from pathlib import Path
import sys

# --------- nycklar --------------------
REQUIRED_ENV = ("CCXT_API_KEY", "CCXT_API_SECRET")

def load_secrets_if_missing(file_path: str = "secrets.json") -> None:
    """
    - Om secrets.json finns: läs in och fyll endast saknade env-variabler.
    - Om den inte finns: gör ingenting (miljön måste redan ha allt).
    - Skriver aldrig ut nyckelvärden.
    """
    p = Path(file_path).expanduser()
    if not p.exists():
        # ingen fil → bara lämna miljön orörd
        return

    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        # Fil finns men kan ej läsas → faila säkert
        sys.exit(f"Fel: kunde inte läsa {file_path}: {e}")

    # Sätt ENDAST saknade env-nycklar
    for k, v in (data or {}).items():
        if k in REQUIRED_ENV and (k not in os.environ or not os.environ[k]):
            os.environ[k] = str(v)

def require_env(keys=REQUIRED_ENV) -> None:
    """Avsluta med tydlig text om någon obligatorisk nyckel saknas (värden skrivs aldrig)."""
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        names = ", ".join(missing)
        sys.exit(
            "Saknar nödvändiga miljövariabler: "
            f"{names}. Sätt dem i miljön eller lägg en secrets.json."
        )



# ---------- TEKNISK ANALYS ----------
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi_wilder(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def decide_signal(df, close_col="close"):
    df["rsi_14"] = rsi_wilder(df[close_col])
    df["ema_12"] = ema(df[close_col], 12)
    df["ema_26"] = ema(df[close_col], 26)
    df["ema_200"] = ema(df[close_col], 200)
    macd_line, sig_line, hist = macd(df[close_col])
    df["macd"], df["macd_signal"], df["macd_hist"] = macd_line, sig_line, hist

    last = df.iloc[-1]
    score = 0
    reasons = []

    # RSI
    if last["rsi_14"] < 30:
        score += 1; reasons.append("RSI < 30 (bullish)")
    elif last["rsi_14"] > 70:
        score -= 1; reasons.append("RSI > 70 (bearish)")

    # EMA crossover
    if last["ema_12"] > last["ema_26"]:
        score += 1; reasons.append("EMA12 > EMA26 (bullish trend)")
    elif last["ema_12"] < last["ema_26"]:
        score -= 1; reasons.append("EMA12 < EMA26 (bearish trend)")

    # MACD
    if last["macd"] > last["macd_signal"]:
        score += 1; reasons.append("MACD > signal (bullish momentum)")
    else:
        score -= 1; reasons.append("MACD < signal (bearish momentum)")

    # Lång trend
    if last[close_col] > last["ema_200"]:
        score += 1; reasons.append("Pris > EMA200 (bullish lång trend)")
    else:
        score -= 1; reasons.append("Pris < EMA200 (bearish lång trend)")

    if score >= 2:
        return "BUY", score, reasons
    elif score <= -2:
        return "SELL", score, reasons
    return "HOLD", score, reasons


# ---------- CCXT BROKER ----------
class CCXTBroker:
    def __init__(self, exchange_id, api_key=None, api_secret=None, sandbox=False):
        if ccxt is None:
            raise RuntimeError("ccxt saknas. Installera med: pip install ccxt")
        klass = getattr(ccxt, exchange_id)
        self.exchange = klass({
            "apiKey": api_key or os.getenv("CCXT_API_KEY", ""),
            "secret": api_secret or os.getenv("CCXT_API_SECRET", ""),
            "enableRateLimit": True,
        })
        if sandbox and hasattr(self.exchange, "set_sandbox_mode"):
            self.exchange.set_sandbox_mode(True)

    def fetch_portfolio(self, symbol):
        base, quote = symbol.split("/")
        bal = self.exchange.fetch_balance()
        base_amt = float(bal["free"].get(base, 0))
        quote_amt = float(bal["free"].get(quote, 0))
        return base, quote, base_amt, quote_amt

    def fetch_price(self, symbol):
        t = self.exchange.fetch_ticker(symbol)
        return t.get("last") or t.get("close") or t.get("bid") or t.get("ask")

    def market_buy(self, symbol, qty):
        return self.exchange.create_order(symbol, "market", "buy", qty)

    def market_sell(self, symbol, qty):
        return self.exchange.create_order(symbol, "market", "sell", qty)


# ---------- MAIN ----------
def main():
    load_secrets_if_missing("secrets.json")

    ap = argparse.ArgumentParser(description="Binär TA-agent (0/100%) med CCXT och dry-run")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--portfolio", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--api-secret", default=None)
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--buy-alloc", type=float, default=1.0)
    ap.add_argument("--sell-alloc", type=float, default=0.0)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--min-trade", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true", help="Simulera utan att skicka riktiga ordrar")
    ap.add_argument("--log", default="trades_log.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
    signal, score, reasons = decide_signal(df)

    ts = df.iloc[-1][df.columns[0]]
    last_close = float(df.iloc[-1]["close"])

    print(f"Signal: {signal} (score {score}) vid {ts}")
    for r in reasons:
        print(" -", r)

    broker = None
    if not args.dry_run:
        broker = CCXTBroker(args.exchange, args.api_key, args.api_secret, args.sandbox)
        base, quote, units, cash = broker.fetch_portfolio(args.symbol)
    else:
        base, quote, units, cash = "BTC", "USDT", 0.0, 10000.0
        print("Dry run-läge: simulerar portfölj 10000 USDT cash.")

    price = last_close if args.dry_run else broker.fetch_price(args.symbol)
    equity = cash + units * price
    curr_alloc = (units * price) / equity if equity > 0 else 0

    target_alloc = args.buy-alloc if signal == "BUY" else args.sell_alloc if signal == "SELL" else curr_alloc
    diff = target_alloc - curr_alloc

    if abs(diff) < 0.01:
        print("Ingen rebalansering behövs.")
        return

    if signal == "BUY" and diff > 0:
        buy_value = equity * diff
        qty = (buy_value / price) * (1 - args.fee_bps / 10000)
        if args.dry_run:
            print(f"[DryRun] Skulle köpa {qty:.6f} {base} för ca {buy_value:.2f} {quote}")
        else:
            order = broker.market_buy(args.symbol, qty)
            print("Köpt:", order)
    elif signal == "SELL" and diff < 0:
        sell_value = equity * (-diff)
        qty = (sell_value / price) * (1 - args.fee_bps / 10000)
        if args.dry_run:
            print(f"[DryRun] Skulle sälja {qty:.6f} {base} för ca {sell_value:.2f} {quote}")
        else:
            order = broker.market_sell(args.symbol, qty)
            print("Sålt:", order)
    else:
        print("Håller position (HOLD).")

    now = datetime.now(ZoneInfo("Europe/Stockholm")).strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "time": now,
        "symbol": args.symbol,
        "decision": signal,
        "score": score,
        "price": price,
        "target_alloc": target_alloc,
        "current_alloc": curr_alloc,
        "dry_run": args.dry_run,
    }

    log_file = Path(args.log)
    exists = log_file.exists()
    import csv
    with log_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    portfolio = {"cash": cash, "positions": {base: units}, "quote": quote}
    Path(args.portfolio).write_text(json.dumps(portfolio, indent=2))
    print("Uppdaterade portfolio.json och trades_log.csv.")

if __name__ == "__main__":
    main()
