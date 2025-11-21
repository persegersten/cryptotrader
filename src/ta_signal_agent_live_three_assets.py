#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ta_signal_agent_live_binary.py (3-asset edition)

Tre-tillgångars TA-agent med CCXT-broker och dry-run.
- Läser tre CSV:er (OHLCV), kör TA per symbol (BUY/SELL/HOLD)
- Hämtar portfölj (tre bas-coin + USD/quote)
- Rebalanserar enligt enkel strategi:
    * SELL => 0% (sälj allt)
    * HOLD => behåll nuvarande
    * BUY  => köp endast EN tillgång (den BUY med högst score) med all tillgänglig USD
- Lägg inte köp under minimal USD (default 20 USDC)
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

from heroku_ip_proxy import get_proxy

from portfolio_rebalancer import rebalance_three, RebalanceResult
from download_portfolio import load_secrets_if_missing

# --------- nycklar --------------------
REQUIRED_ENV = ("CCXT_API_KEY", "CCXT_API_SECRET")

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

        proxies = get_proxy()
        print(f"IP proxy: {proxies}")

        klass = getattr(ccxt, exchange_id)
        exchange_config = {
            "apiKey": api_key or os.getenv("CCXT_API_KEY", ""),
            "secret": api_secret or os.getenv("CCXT_API_SECRET", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
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


def run_agent(
    csvA: str,
    csvB: str,
    csvC: str,
    symbols: str,
    exchange: str = "binance",
    api_key: str | None = None,
    api_secret: str | None = None,
    sandbox: bool = False,
    fee_bps: float = 10.0,
    min_trade: float = 20.0,
    dry_run: bool = True,
    log: str = "trades_log.csv",
    portfolio: str = "portfolio.json",
) -> dict:
    """
    Programmatic entrypoint for the TA agent.
    Parameters mirror the --cli args from the original script.
    Returns a portfolio_snapshot dict (same structure that the CLI writes to portfolio.json).
    """

    # Use the same helpers already defined in the module: require_env, CCXTBroker, decide_signal, rebalance_three, etc.
    load_secrets_if_missing("secrets.json")

    # Validate symbols and build internal vars (copied from the original main logic)
    csv_map = {"A": csvA, "B": csvB, "C": csvC}
    syms_list = [s.strip() for s in symbols.split(",")]
    if len(syms_list) != 3:
        raise ValueError("You must provide exactly three comma-separated symbols, e.g. 'BTC/USDT,ETH/USDT,SOL/USDT'")

    # Ensure same quote
    quotes = [s.split("/")[1] for s in syms_list]
    if not (quotes[0] == quotes[1] == quotes[2]):
        raise ValueError("All three symbol pairs must share the same quote currency (e.g. USDT)")

    quote_ccy = quotes[0]
    base_ccys = [s.split("/")[0] for s in syms_list]

    # load csvs and compute signals
    def load_and_signal(csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
        sig, score, reasons = decide_signal(df)
        ts = df.iloc[-1][df.columns[0]]
        last_close = float(df.iloc[-1]["close"])
        return df, sig, score, reasons, ts, last_close

    dfs: Dict[str, pd.DataFrame] = {}
    signals: Dict[str, str] = {}
    meta: Dict[str, dict] = {}
    for name, sym in zip(("A", "B", "C"), syms_list):
        df, sig, score, reasons, ts, last_close = load_and_signal(csv_map[name])
        dfs[sym] = df
        signals[sym.split("/")[0]] = sig
        meta[sym] = {"score": score, "reasons": reasons, "ts": ts, "last_close": last_close}
        print(f"{sym} - {meta[sym]}") # TODO format output
    
    # Broker / portfolio: require_env guards that API keys exist in environment
    require_env()
    broker = CCXTBroker(exchange, api_key, api_secret, sandbox)

    balances = broker.fetch_balances()

    # Fetch prices
    prices: Dict[str, float] = {}
    for s in syms_list:
        try:
            prices[s] = broker.fetch_price(s)
        except Exception:
            raise RuntimeError("Failed to look up last close price")

    # current allocations (percent)
    current_alloc = get_current_allocations_pct_three(balances, tuple(syms_list), prices, quote_ccy)
    # TODO imprve forma on output current alloc
    print(f"Allocations {current_alloc}")

    # build scores_map and call rebalance_three
    scores_map = {}
    for i, full_sym in enumerate(syms_list):
        base = base_ccys[i]
        scores_map[base] = meta[full_sym]["score"]

    rb: RebalanceResult = rebalance_three(
        [b for b in base_ccys],
        {b: signals[b] for b in base_ccys},
        current_alloc,
        scores=scores_map,
    )

    # compute equity in quote
    equity = 0.0
    for i, sym in enumerate(syms_list):
        base = base_ccys[i]
        if isinstance(balances.get("free"), dict):
            qty = float(balances["free"].get(base, 0.0) + balances.get("total", {}).get(base, 0.0) - balances.get("used", {}).get(base, 0.0))
            if qty < 0:
                qty = float(balances["free"].get(base, 0.0))
        else:
            qty = float(balances.get(base, 0.0))
        equity += qty * prices[sym]
    quote_bal = float(balances["free"].get(quote_ccy, 0.0)) if isinstance(balances.get("free"), dict) else float(balances.get(quote_ccy, 0.0))
    equity += quote_bal

    # translate trades -> planned orders (same logic as original)
    planned_orders: List[Tuple[str, str, float]] = []
    fee_mult = (1 - fee_bps / 10000.0)

    def get_free_base(b: str) -> float:
        if isinstance(balances.get("free"), dict):
            return float(balances["free"].get(b, 0.0))
        return float(balances.get(b, 0.0))

    def get_free_quote() -> float:
        if isinstance(balances.get("free"), dict):
            return float(balances["free"].get(quote_ccy, 0.0))
        return float(balances.get(quote_ccy, 0.0))

    for i, base in enumerate(base_ccys):
        sym_pair = syms_list[i]
        delta_pp = float(rb.trades.get(base, 0.0))
        if abs(delta_pp) < 1e-6:
            continue
        usd_delta = (delta_pp / 100.0) * equity

        if usd_delta > 0:
            usd_to_spend = usd_delta * fee_mult
            if usd_to_spend >= min_trade and get_free_quote() >= usd_to_spend:
                planned_orders.append(("BUY", sym_pair, usd_to_spend))
        else:
            px = prices[sym_pair]
            qty = (abs(usd_delta) / max(px, 1e-12)) * fee_mult
            qty = min(qty, get_free_base(base))
            if qty * px >= min_trade and qty > 0:
                planned_orders.append(("SELL", sym_pair, qty))

    # execute planned orders (dry_run controls whether to place actual orders)
    executions = []
    if planned_orders:
        for side, sym_pair, amount in planned_orders:
            if dry_run:
                if side == "BUY":
                    executions.append({"side": side, "symbol": sym_pair, "amount": amount, "order_id": None})
                else:
                    px = prices[sym_pair]
                    executions.append({"side": side, "symbol": sym_pair, "amount": amount, "order_id": None})
            else:
                if side == "BUY":
                    order = broker.market_buy_quote(sym_pair, quote_amount=amount)
                else:
                    order = broker.market_sell_base(sym_pair, base_qty=amount)
                executions.append({"side": side, "symbol": sym_pair, "amount": amount, "order_id": order.get("id") if isinstance(order, dict) else str(order)})

    # build portfolio snapshot
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

    # write outputs as the CLI did
    Path(log).write_text("") if False else None  # noop to avoid lint errors; keep original write logic below
    # write CSV log row + portfolio json
    now = datetime.now(ZoneInfo("Europe/Stockholm")).strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "time": now,
        "symbols": symbols,
        "signals": json.dumps({b: signals[b] for b in base_ccys}),
        "prices": json.dumps({s: prices[s] for s in syms_list}),
        "current_alloc": json.dumps(current_alloc),
        "targets": json.dumps(rb.target_allocations),
        "trades_pp": json.dumps(rb.trades),
        "reason": rb.reason,
        "dry_run": dry_run,
    }
    #log_file = Path(log)
    #exists = log_file.exists()
    #with log_file.open("a", newline="", encoding="utf-8") as f:
    #    writer = csv.DictWriter(f, fieldnames=row.keys())
    #    if not exists:
    #        writer.writeheader()
    #    writer.writerow(row)
    print(f"{row}")

    Path(portfolio).write_text(json.dumps(portfolio_snapshot, indent=2, ensure_ascii=False))
    return portfolio_snapshot


# --- Change the existing main() at the end of the file to call run_agent(...) so CLI continues to work:
# Replace the current main() body (argparse + logic) with a thin wrapper:
def main():
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
    ap.add_argument("--min-trade", type=float, default=20.0, help="minsta order i USD (quote). Default 20 USDC per nya regler")
    ap.add_argument("--dry-run", action="store_true", help="Simulera utan riktiga ordrar (hämtar ändå verklig portfolio och priser)")
    ap.add_argument("--log", default="trades_log.csv")
    ap.add_argument("--portfolio", default="portfolio.json", help="Filen att spara täckt portfölj-snapshot till")
    args = ap.parse_args()

    snapshot = run_agent(
        csvA=args.csvA,
        csvB=args.csvB,
        csvC=args.csvC,
        symbols=args.symbols,
        exchange=args.exchange,
        api_key=args.api_key,
        api_secret=args.api_secret,
        sandbox=args.sandbox,
        fee_bps=args.fee_bps,
        min_trade=args.min_trade,
        dry_run=args.dry_run,
        log=args.log,
        portfolio=args.portfolio,
    )
    # main prints/writes as run_agent already wrote files
    if args.dry_run:
        print("Dry-run: used real balances/prices but did not place live orders.")
    else:
        print("Production: orders placed where planned.")