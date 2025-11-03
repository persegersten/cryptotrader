#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ta_signal_agent_live_binary.py (3-asset edition)

Tre-tillgångars TA-agent med CCXT-broker och dry-run.
- Läser tre CSV:er (OHLCV), kör TA per symbol (BUY/SELL/HOLD)
- Hämtar portfölj (tre bas-coin + USD/quote)
- Rebalanserar enligt rebalance_three: SELL=0%, övriga lika (1 aktör: 100%, 2: 50/50, 3: ~33/33/33) med ±10% band
- Lägger market-ordrar (eller simulerar i dry-run)
- Loggar beslut

Kräver: portfolio_rebalancer.py i samma katalog.
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import ccxt

from heroku_ip_proxy import getProxy

from portfolio_rebalancer import rebalance_three, RebalanceResult

# --------- nycklar --------------------
REQUIRED_ENV = ("CCXT_API_KEY", "CCXT_API_SECRET")

def load_secrets_if_missing(file_path: str = "secrets.json") -> None:
    p = Path(file_path).expanduser()
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception as e:
        sys.exit(f"Fel: kunde inte läsa {file_path}: {e}")
    for k, v in data.items():
        if k in REQUIRED_ENV and (k not in os.environ or not os.environ[k]):
            os.environ[k] = str(v)

def require_env(keys=REQUIRED_ENV) -> None:
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
    df = df.copy()
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
        score += 1; reasons.append("EMA12 > EMA26 (bullish)")
    elif last["ema_12"] < last["ema_26"]:
        score -= 1; reasons.append("EMA12 < EMA26 (bearish)")

    # MACD
    if last["macd"] > last["macd_signal"]:
        score += 1; reasons.append("MACD > signal (bullish)")
    else:
        score -= 1; reasons.append("MACD < signal (bearish)")

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

        proxies = getProxy()
        print(f"IP proxy: {proxies}")

        klass = getattr(ccxt, exchange_id)
        exchange_config = {
            "apiKey": api_key or os.getenv("CCXT_API_KEY", ""),
            "secret": api_secret or os.getenv("CCXT_API_SECRET", ""),
            "enableRateLimit": True,
        }
        if proxies:
            exchange_config["requests_kwargs"] = {"proxies": proxies}

        self.exchange = klass(exchange_config)
        if sandbox and hasattr(self.exchange, "set_sandbox_mode"):
            self.exchange.set_sandbox_mode(True)

    def fetch_balances(self):
        return self.exchange.fetch_balance()

    def fetch_price(self, symbol: str) -> float:
        t = self.exchange.fetch_ticker(symbol)
        return t.get("last") or t.get("close") or t.get("bid") or t.get("ask")

    def market_buy_quote(self, symbol: str, quote_amount: float):
        """
        Market-köp för en given quote-amount (t.ex. USDT).
        Skapar order i 'quote' om börsen stöder det, annars approximerar vi qty.
        """
        m = self.exchange.load_markets()
        market = m[symbol]
        if market.get("quote", "").upper() in ("USDT", "USD", "USDC") and market.get("spot", True):
            # De flesta stödjer 'createOrder' i bas-kvantitet, så beräkna qty från pris:
            px = self.fetch_price(symbol)
            qty = quote_amount / max(px, 1e-12)
            return self.exchange.create_order(symbol, "market", "buy", qty)
        else:
            # fallback: ändå qty
            px = self.fetch_price(symbol)
            qty = quote_amount / max(px, 1e-12)
            return self.exchange.create_order(symbol, "market", "buy", qty)

    def market_sell_base(self, symbol: str, base_qty: float):
        return self.exchange.create_order(symbol, "market", "sell", base_qty)

# ---------- Hjälp: portfölj & allokeringar ----------
def get_current_allocations_pct_three(
    balances: dict,
    symbols: Tuple[str, str, str],
    prices: Dict[str, float],
    quote_ccy: str
) -> Dict[str, float]:
    """
    Räknar procentallokeringar över tre bas-coin + USD (quote_ccy).
    balances: från ccxt.fetch_balance()
    """
    base_ccys = [s.split("/")[0] for s in symbols]
    vals: Dict[str, float] = {}
    total = 0.0

    # värde per bas
    for i, sym in enumerate(symbols):
        base = base_ccys[i]
        qty = float(balances["free"].get(base, 0.0) + balances["total"].get(base, 0.0) - balances["used"].get(base, 0.0)) \
              if isinstance(balances.get("free"), dict) else float(balances.get(base, 0.0))
        v = qty * prices[sym]
        vals[base] = v
        total += v

    # USD/quote
    quote_bal = float(balances["free"].get(quote_ccy, 0.0)) if isinstance(balances.get("free"), dict) else float(balances.get(quote_ccy, 0.0))
    vals["USD"] = quote_bal
    total += quote_bal

    # till procent
    if total <= 0:
        return {**{b: 0.0 for b in base_ccys}, "USD": 100.0}

    alloc = {b: (vals[b] * 100.0 / total) for b in base_ccys}
    alloc["USD"] = vals["USD"] * 100.0 / total
    return alloc

# ---------- MAIN ----------
def main():
    load_secrets_if_missing("secrets.json")

    ap = argparse.ArgumentParser(description="Tre-tillgångars TA-agent med CCXT och rebalansering")
    ap.add_argument("--csvA", required=True, help="CSV med OHLCV för symbol A")
    ap.add_argument("--csvB", required=True, help="CSV med OHLCV för symbol B")
    ap.add_argument("--csvC", required=True, help="CSV med OHLCV för symbol C")
    ap.add_argument("--symbols", required=True, help="Komma-separerade tre symbolpar, t.ex. BTC/USDT,ETH/USDT,SOL/USDT")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--api-secret", default=None)
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--fee-bps", type=float, default=10.0, help="avgifter i bps")
    ap.add_argument("--min-trade", type=float, default=10.0, help="minsta order i USD (quote)")
    ap.add_argument("--dry-run", action="store_true", help="Simulera utan riktiga ordrar")
    ap.add_argument("--log", default="trades_log.csv")
    ap.add_argument("--tolerance", type=float, default=0.10, help="±band runt mål (0.10 = ±10%)")
    ap.add_argument("--rebalance-even-when-inside-band", action="store_true", help="Tvinga rebalans även inom bandet")
    ap.add_argument("--portfolio", default="portfolio.json", help="Filen att spara täckt portfölj-snapshot till")
    args = ap.parse_args()

    # Läs CSV och skapa TA-signal per symbol
    csv_map = {"A": args.csvA, "B": args.csvB, "C": args.csvC}
    syms_list = [s.strip() for s in args.symbols.split(",")]
    if len(syms_list) != 3:
        sys.exit("Du måste ange exakt tre symboler i --symbols, t.ex. BTC/USDT,ETH/USDT,SOL/USDT")

    # Kontroll: alla tre ska ha samma quote
    quotes = [s.split("/")[1] for s in syms_list]
    if not (quotes[0] == quotes[1] == quotes[2]):
        sys.exit("Alla symboler måste ha samma quote-valuta (t.ex. USDT).")

    quote_ccy = quotes[0]
    base_ccys = [s.split("/")[0] for s in syms_list]

    def load_and_signal(csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
        sig, score, reasons = decide_signal(df)
        ts = df.iloc[-1][df.columns[0]]
        last_close = float(df.iloc[-1]["close"])
        return df, sig, score, reasons, ts, last_close

    # Per symbol
    dfs: Dict[str, pd.DataFrame] = {}
    signals: Dict[str, str] = {}
    meta: Dict[str, dict] = {}
    for name, sym in zip(("A","B","C"), syms_list):
        df, sig, score, reasons, ts, last_close = load_and_signal(csv_map[name])
        dfs[sym] = df
        signals[sym.split("/")[0]] = sig  # signal mappad på bas-ccy
        meta[sym] = {"score": score, "reasons": reasons, "ts": ts, "last_close": last_close}

    print("Signaler:")
    for sym in syms_list:
        b = sym.split("/")[0]
        m = meta[sym]
        print(f" - {sym}: {signals[b]} (score {m['score']}) @ {m['ts']}")
        for r in m["reasons"]:
            print("    ·", r)

    # Broker / portfölj
    if not args.dry_run:
        require_env()
        broker = CCXTBroker(args.exchange, args.api_key, args.api_secret, args.sandbox)
        balances = broker.fetch_balances()
        prices = {s: (meta[s]["last_close"] if args.dry_run else broker.fetch_price(s)) for s in syms_list}
    else:
        print("Dry run-läge: simulerar portfölj med 10 000 i kassa och 0 bas.")
        broker = None
        # Simulerade saldon
        balances = {
            "free": {quote_ccy: 10_000.0, base_ccys[0]: 0.0, base_ccys[1]: 0.0, base_ccys[2]: 0.0},
            "used": {},
            "total": {},
        }
        prices = {s: meta[s]["last_close"] for s in syms_list}

    # Allokeringar i %
    current_alloc = get_current_allocations_pct_three(balances, tuple(syms_list), prices, quote_ccy)
    print("Nuvarande allokering (%):", current_alloc)

    # Rebalansering
    rb: RebalanceResult = rebalance_three(
        [b for b in base_ccys],                 # symbols som bas-ccy
        {b: signals[b] for b in base_ccys},     # signal per bas
        current_alloc,                          # procent-alloc inkl USD
        tolerance=args.tolerance,
        only_if_outside_band=(not args.rebalance_even_when_inside_band),
    )

    print("Målvikt (%):", rb.target_allocations)
    print("Trades (p.p.):", rb.trades)
    print("Orsak:", rb.reason)

    # Portföljvärde i quote
    equity = 0.0
    # bas-värden
    for i, sym in enumerate(syms_list):
        base = base_ccys[i]
        # qty ur balances
        if isinstance(balances.get("free"), dict):
            qty = float(balances["free"].get(base, 0.0) + balances.get("total", {}).get(base, 0.0) - balances.get("used", {}).get(base, 0.0))
            if qty < 0: qty = float(balances["free"].get(base, 0.0))  # defensivt
        else:
            qty = float(balances.get(base, 0.0))
        equity += qty * prices[sym]
    # quote-kassa
    quote_bal = float(balances["free"].get(quote_ccy, 0.0)) if isinstance(balances.get("free"), dict) else float(balances.get(quote_ccy, 0.0))
    equity += quote_bal

    # Översätt trades → orders
    # rb.trades är i procentenheter av total portfölj. Positivt = köp, negativt = sälj.
    planned_orders: List[Tuple[str, str, float]] = []  # (side, symbol, amount); BUY i quote, SELL i bas-qty
    fee_mult = (1 - args.fee_bps / 10000.0)

    # Hjälp att hämta tillgängliga mängder i dry-run
    def get_free_base(b: str) -> float:
        if isinstance(balances.get("free"), dict):
            return float(balances["free"].get(b, 0.0))
        return float(balances.get(b, 0.0))

    def get_free_quote() -> float:
        if isinstance(balances.get("free"), dict):
            return float(balances["free"].get(quote_ccy, 0.0))
        return float(balances.get(quote_ccy, 0.0))

    # Köp/sälj per bas-ccy
    for i, base in enumerate(base_ccys):
        sym_pair = syms_list[i]
        delta_pp = float(rb.trades.get(base, 0.0))
        if abs(delta_pp) < 1e-6:
            continue
        usd_delta = (delta_pp / 100.0) * equity

        if usd_delta > 0:
            # KÖP bas för usd_delta (quote)
            usd_to_spend = usd_delta * fee_mult
            if usd_to_spend >= args.min_trade and get_free_quote() >= usd_to_spend:
                planned_orders.append(("BUY", sym_pair, usd_to_spend))
        else:
            # SÄLJ bas för |usd_delta| → qty
            px = prices[sym_pair]
            qty = (abs(usd_delta) / max(px, 1e-12)) * fee_mult
            qty = min(qty, get_free_base(base))
            if qty * px >= args.min_trade and qty > 0:
                planned_orders.append(("SELL", sym_pair, qty))

    # Kör ordrar
    executions = []
    if not planned_orders:
        print("Ingen rebalans behövs (inom band eller små poster).")
    else:
        for side, sym_pair, amount in planned_orders:
            if args.dry_run:
                if side == "BUY":
                    print(f"[DryRun] Skulle KÖPA {sym_pair} för ~{amount:.2f} {quote_ccy}")
                else:
                    print(f"[DryRun] Skulle SÄLJA {amount:.6f} av {sym_pair.split('/')[0]} (≈{amount*prices[sym_pair]:.2f} {quote_ccy})")
                executions.append({"side": side, "symbol": sym_pair, "amount": amount, "order_id": None})
            else:
                if side == "BUY":
                    order = broker.market_buy_quote(sym_pair, quote_amount=amount)
                else:
                    order = broker.market_sell_base(sym_pair, base_qty=amount)
                executions.append({"side": side, "symbol": sym_pair, "amount": amount, "order_id": order.get("id") if isinstance(order, dict) else str(order)})
                print(f"Order {side} {sym_pair}: {executions[-1]['order_id']}")

    # Loggning (en rad per körning)
    now = datetime.now(ZoneInfo("Europe/Stockholm")).strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "time": now,
        "symbols": args.symbols,
        "signals": json.dumps({b: signals[b] for b in base_ccys}),
        "prices": json.dumps({s: prices[s] for s in syms_list}),
        "current_alloc": json.dumps(current_alloc),
        "targets": json.dumps(rb.target_allocations),
        "trades_pp": json.dumps(rb.trades),
        "reason": rb.reason,
        "dry_run": args.dry_run,
    }
    log_file = Path(args.log)
    exists = log_file.exists()
    with log_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    # Spara portfölj-snapshot
    portfolio_snapshot = {
        "quote": quote_ccy,
        "prices": {s: prices[s] for s in syms_list},
        "executions": executions,
        "rebalance_reason": rb.reason,
        "signals": {b: signals[b] for b in base_ccys},
        "targets_pct": rb.target_allocations,
        "trades_pp": rb.trades,
        "current_alloc_pct": current_alloc,
    }
    Path(args.portfolio).write_text(json.dumps(portfolio_snapshot, indent=2, ensure_ascii=False))
    print("Uppdaterade portfolio.json och trades_log.csv.")

if __name__ == "__main__":
    main()
