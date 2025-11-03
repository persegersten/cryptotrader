#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ta_signal_agent_live_binary_weighted.py

Tre-tillgångars TA-agent (RSI/MACD/EMA/ATR) med CCXT/dry-run och momentumviktad rebalansering.
- Läser tre CSV:er (OHLCV), kör TA per symbol → BUY/SELL/HOLD + momentumscore [0..1]
- Hämtar portfölj (tre bas-coin + USD/quote)
- Rebalanserar enligt "rebalance_three_weighted":
    · SELL = 0%
    · Aktiva (= ej SELL) delar på kapitalet proportionellt mot momentumscore (fallback: lika)
    · Adaptivt band via ATR (±band i procentenheter p.p.)
- Lägger market-ordrar (eller simulerar i dry-run)
- Loggar beslut (CSV) och skriver portfolio.json

Kräver: pandas, numpy. För live-ordrar: ccxt. Proxy-modul är valfri.
"""

import os
import sys
import json
import csv
import argparse
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None  # tillåter dry-run utan ccxt

# ---- Proxy (valfritt) ----
def _safe_get_proxy():
    try:
        from heroku_ip_proxy import getProxy  # type: ignore
        return getProxy()
    except Exception:
        return None

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
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, n: int = 14, high_col="high", low_col="low", close_col="close"):
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    return atr_

def _minmax_norm(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x):
        return 0.0
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, v)))

def momentum_score_row(row: pd.Series) -> float:
    """
    Score (0..1) av senaste rad, viktar:
      - RSI: 30->0, 70->1
      - MACD-hist normaliserad mot 0.5*ATR
      - EMA-spread: (ema12-ema26)/close, klipp [-2%, +2%] -> [0,1]
    Vikter: 0.4, 0.4, 0.2
    """
    rsi = float(row.get("rsi_14", np.nan))
    macd_hist = float(row.get("macd_hist", np.nan))
    atr_v = float(row.get("atr_14", np.nan))
    close = float(row.get("close", np.nan))
    ema12 = float(row.get("ema_12", np.nan))
    ema26 = float(row.get("ema_26", np.nan))

    rsi_norm = _minmax_norm(rsi, 30.0, 70.0)
    scale = max(atr_v, 1e-12) * 0.5
    macd_norm = _minmax_norm(macd_hist, -scale, +scale)
    spread = (ema12 - ema26) / max(close, 1e-12) if np.isfinite(close) else 0.0
    ema_norm = _minmax_norm(spread, -0.02, 0.02)
    score = 0.4 * rsi_norm + 0.4 * macd_norm + 0.2 * ema_norm
    return float(score)

# ---------- BESLUT ----------
def decide_signal(df: pd.DataFrame, close_col: str = "close"):
    """
    Returnerar: (decision, score_raw, reasons, mom_score)
    decision: BUY/SELL/HOLD med "kvalificerad exit/entry"
    """
    df = df.copy()
    # säkerställ float
    for col in ("open","high","low","close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["rsi_14"] = rsi_wilder(df[close_col])
    df["ema_12"] = ema(df[close_col], 12)
    df["ema_26"] = ema(df[close_col], 26)
    df["ema_200"] = ema(df[close_col], 200)
    macd_line, sig_line, hist = macd(df[close_col])
    df["macd"], df["macd_signal"], df["macd_hist"] = macd_line, sig_line, hist
    df["atr_14"] = atr(df)

    last = df.iloc[-1]
    reasons: List[str] = []

    ema200_slope = df["ema_200"].iloc[-1] - df["ema_200"].iloc[-5] if len(df) >= 5 else 0.0
    uptrend = (last[close_col] > last["ema_200"]) and (ema200_slope > 0)
    reasons.append(f"Regim: {'UP' if uptrend else 'DOWN/NEUTRAL'} (EMA200-slope={ema200_slope:.6f})")

    score_raw = 0
    if last["rsi_14"] < 30: score_raw += 1; reasons.append("RSI < 30 (bullish)")
    elif last["rsi_14"] > 70: score_raw -= 1; reasons.append("RSI > 70 (bearish)")
    if last["ema_12"] > last["ema_26"]: score_raw += 1; reasons.append("EMA12 > EMA26 (bullish)")
    else: score_raw -= 1; reasons.append("EMA12 < EMA26 (bearish)")
    if last["macd"] > last["macd_signal"]: score_raw += 1; reasons.append("MACD > signal (bullish)")
    else: score_raw -= 1; reasons.append("MACD < signal (bearish)")
    if last[close_col] > last["ema_200"]: score_raw += 1; reasons.append("Pris > EMA200 (bullish lång trend)")
    else: score_raw -= 1; reasons.append("Pris < EMA200 (bearish lång trend)")

    bullish = 0
    bearish = 0
    bullish += int(last["rsi_14"] < 40)
    bearish += int(last["rsi_14"] > 60)
    bullish += int(last["ema_12"] > last["ema_26"])
    bearish += int(last["ema_12"] < last["ema_26"])
    bullish += int(last["macd"] > last["macd_signal"])
    bearish += int(last["macd"] < last["macd_signal"])
    bullish += int(last[close_col] > last["ema_200"])
    bearish += int(last[close_col] < last["ema_200"])

    if uptrend and bullish >= 2:
        decision = "BUY"
    elif (not uptrend) and bearish >= 2 and (last[close_col] < last["ema_26"]) and (last["macd"] < last["macd_signal"]):
        decision = "SELL"
    else:
        decision = "HOLD"

    mom_score = momentum_score_row(last)
    reasons.append(f"MomentumScore={mom_score:.3f}")
    return decision, int(score_raw), reasons, mom_score

# ---------- ADAPTIV TOLERANS ----------
def adaptive_tolerance(df: pd.DataFrame, base_tol: float, low: float = 0.01, high: float = 0.05) -> float:
    """
    Skala toleransen (bandet i p.p.) baserat på senaste ATR/close.
    0.5% vol -> 'low', 3% vol -> 'high'. Blandas 50/50 med base_tol.
    """
    last = df.iloc[-1]
    atr_v = float(last.get("atr_14", np.nan))
    close = float(last.get("close", np.nan))
    if not np.isfinite(atr_v) or not np.isfinite(close) or close <= 0:
        return base_tol
    vol_q = atr_v / close
    v = _minmax_norm(vol_q, 0.005, 0.03)
    tol = low + v * (high - low)
    return 0.5 * base_tol + 0.5 * tol

# ---------- REBALANCE (weighted) ----------
@dataclass
class RebalanceResult:
    target_allocations: Dict[str, float]
    trades: Dict[str, float]          # p.p. (procentenheter)
    reason: str

def _within_band(current: float, target: float, band_pp: float) -> bool:
    # band i procentENHETER, ex: 0.10 = ±10 p.p.
    return abs(current - target) <= (band_pp * 100.0)

def rebalance_three_weighted(
    bases: List[str],
    signals: Dict[str, str],            # bas-> "BUY"/"SELL"/"HOLD"
    current_alloc: Dict[str, float],    # procent per bas + "USD"
    mom_scores: Dict[str, float],       # bas-> [0..1]
    tolerance: float = 0.10,            # 0.10 = ±10 p.p.
    only_if_outside_band: bool = True
) -> RebalanceResult:
    """
    SELL = 0%. Aktiva (= ej SELL) tar resterande kapital momentum-viktat (fallback: lika).
    Om ingen aktiv: USD=100%.
    Band i procentenheter, tillämpas på trades.
    """
    active = [b for b in bases if signals.get(b, "HOLD") != "SELL"]
    targets: Dict[str, float] = {b: 0.0 for b in bases}
    usd_target = 100.0

    if active:
        usd_target = 0.0
        scores = np.array([max(mom_scores.get(b, 0.0), 0.0) for b in active], dtype=float)
        if scores.sum() <= 1e-12:
            # lika
            w = np.ones_like(scores) / len(scores)
        else:
            w = scores / scores.sum()
        for b, wi in zip(active, w):
            targets[b] = float(100.0 * wi)
    else:
        # allt i USD
        pass

    targets["USD"] = usd_target

    # Trades p.p.
    keys = set(current_alloc.keys()) | set(targets.keys())
    trades = {k: float(targets.get(k, 0.0) - current_alloc.get(k, 0.0)) for k in keys}

    if only_if_outside_band:
        pruned = {}
        for k, t in trades.items():
            cur = current_alloc.get(k, 0.0)
            tgt = targets.get(k, 0.0)
            pruned[k] = 0.0 if _within_band(cur, tgt, tolerance) else t
        trades = pruned

        if all(abs(v) < 1e-9 for v in trades.values()):
            return RebalanceResult(targets, trades, reason=f"Inom band ±{int(tolerance*100)} p.p.")

    return RebalanceResult(targets, trades, reason="Momentum-viktad rebalans (SELL=0, aktiva viktas)")

# ---------- CCXT BROKER ----------
class CCXTBroker:
    def __init__(self, exchange_id, api_key=None, api_secret=None, sandbox=False):
        if ccxt is None:
            raise RuntimeError("ccxt saknas. Installera: pip install ccxt")

        proxies = _safe_get_proxy()
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
        m = self.exchange.load_markets()
        market = m[symbol]
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
    base_ccys = [s.split("/")[0] for s in symbols]
    vals: Dict[str, float] = {}
    total = 0.0

    for i, sym in enumerate(symbols):
        base = base_ccys[i]
        if isinstance(balances.get("free"), dict):
            qty = float(balances["free"].get(base, 0.0))
        else:
            qty = float(balances.get(base, 0.0))
        v = qty * prices[sym]
        vals[base] = v
        total += v

    quote_bal = float(balances["free"].get(quote_ccy, 0.0)) if isinstance(balances.get("free"), dict) else float(balances.get(quote_ccy, 0.0))
    vals["USD"] = quote_bal
    total += quote_bal

    if total <= 0:
        return {**{b: 0.0 for b in base_ccys}, "USD": 100.0}

    alloc = {b: (vals[b] * 100.0 / total) for b in base_ccys}
    alloc["USD"] = vals["USD"] * 100.0 / total
    return alloc

# ---------- MAIN ----------
def main():
    load_secrets_if_missing("secrets.json")

    ap = argparse.ArgumentParser(description="Tre-tillgångars TA-agent med CCXT och momentumviktad rebalansering")
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
    ap.add_argument("--tolerance", type=float, default=0.10, help="±band runt mål (0.10 = ±10 p.p.)")
    ap.add_argument("--rebalance-even-when-inside-band", action="store_true", help="Tvinga rebalans även inom bandet")
    ap.add_argument("--portfolio", default="portfolio.json", help="Filen att spara portfölj-snapshot till")
    args = ap.parse_args()

    # Läs CSV och skapa TA-signal per symbol
    csv_map = {"A": args.csvA, "B": args.csvB, "C": args.csvC}
    syms_list = [s.strip() for s in args.symbols.split(",")]
    if len(syms_list) != 3:
        sys.exit("Du måste ange exakt tre symboler i --symbols, t.ex. BTC/USDT,ETH/USDT,SOL/USDT")

    quotes = [s.split("/")[1] for s in syms_list]
    if not (quotes[0] == quotes[1] == quotes[2]):
        sys.exit("Alla symboler måste ha samma quote-valuta (t.ex. USDT).")

    quote_ccy = quotes[0]
    base_ccys = [s.split("/")[0] for s in syms_list]

    def load_and_signal(csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
        for col in ("open","high","low","close"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        sig, score, reasons, mom_score = decide_signal(df)
        ts = df.iloc[-1][df.columns[0]]
        last_close = float(df.iloc[-1]["close"])
        return df, sig, score, reasons, ts, last_close, mom_score

    dfs: Dict[str, pd.DataFrame] = {}
    signals: Dict[str, str] = {}
    meta: Dict[str, dict] = {}
    mom_scores: Dict[str, float] = {}

    for name, sym in zip(("A","B","C"), syms_list):
        df, sig, score, reasons, ts, last_close, mom_score = load_and_signal(csv_map[name])
        dfs[sym] = df
        base = sym.split("/")[0]
        signals[base] = sig
        mom_scores[base] = mom_score
        meta[sym] = {"score": score, "reasons": reasons, "ts": ts, "last_close": last_close}

    print("Signaler:")
    for sym in syms_list:
        b = sym.split("/")[0]
        m = meta[sym]
        print(f" - {sym}: {signals[b]} (score {m['score']}) @ {m['ts']}  [mom={mom_scores[b]:.3f}]")
        for r in m["reasons"]:
            print("    ·", r)

    # Broker / portfölj
    if not args.dry_run:
        require_env()
        broker = CCXTBroker(args.exchange, args.api_key, args.api_secret, args.sandbox)
        balances = broker.fetch_balances()
        prices = {s: broker.fetch_price(s) for s in syms_list}
    else:
        print("Dry run-läge: simulerar portfölj med 10 000 i kassa och 0 bas.")
        broker = None
        balances = {
            "free": {quote_ccy: 10_000.0, base_ccys[0]: 0.0, base_ccys[1]: 0.0, base_ccys[2]: 0.0},
            "used": {},
            "total": {},
        }
        prices = {s: float(meta[s]["last_close"]) for s in syms_list}

    current_alloc = get_current_allocations_pct_three(balances, tuple(syms_list), prices, quote_ccy)
    print("Nuvarande allokering (%):", current_alloc)

    # Adaptivt band: medel av tre serier
    tol = args.tolerance
    try:
        # Säkerställ att atr_14 är beräknad (om man vill räkna här)
        # den finns redan i dfs via decide_signal
        tA = adaptive_tolerance(dfs[syms_list[0]], args.tolerance)
        tB = adaptive_tolerance(dfs[syms_list[1]], args.tolerance)
        tC = adaptive_tolerance(dfs[syms_list[2]], args.tolerance)
        tol = float(np.nanmean([tA, tB, tC]))
        print(f"Adaptivt toleransband: {tol:.4f} (tidigare {args.tolerance:.4f})")
    except Exception as e:
        print(f"Adaptivt band misslyckades, använder {args.tolerance:.4f}. Orsak: {e}")
        tol = args.tolerance

    # Momentum-viktad rebalans (SELL=0, aktiva viktas; USD=100% om alla SELL)
    rb = rebalance_three_weighted(
        bases=base_ccys,
        signals=signals,
        current_alloc=current_alloc,
        mom_scores=mom_scores,
        tolerance=tol,
        only_if_outside_band=(not args.rebalance_even_when_inside_band),
    )

    print("Målvikt (%):", rb.target_allocations)
    print("Trades (p.p.):", rb.trades)
    print("Orsak:", rb.reason)

    # Portföljvärde i quote
    equity = 0.0
    for i, sym in enumerate(syms_list):
        base = base_ccys[i]
        if isinstance(balances.get("free"), dict):
            qty = float(balances["free"].get(base, 0.0))
        else:
            qty = float(balances.get(base, 0.0))
        equity += qty * prices[sym]
    quote_bal = float(balances["free"].get(quote_ccy, 0.0)) if isinstance(balances.get("free"), dict) else float(balances.get(quote_ccy, 0.0))
    equity += quote_bal

    planned_orders: List[Tuple[str, str, float]] = []  # (side, symbol, amount); BUY = quote-amount, SELL = base-qty
    fee_mult = (1 - args.fee_bps / 10000.0)

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
        if abs(delta_pp) < 1e-9:
            continue
        usd_delta = (delta_pp / 100.0) * equity

        if usd_delta > 0:
            usd_to_spend = usd_delta * fee_mult
            if usd_to_spend >= args.min_trade and get_free_quote() >= usd_to_spend:
                planned_orders.append(("BUY", sym_pair, usd_to_spend))
        else:
            px = prices[sym_pair]
            qty = (abs(usd_delta) / max(px, 1e-12)) * fee_mult
            qty = min(qty, get_free_base(base))
            if qty * px >= args.min_trade and qty > 0:
                planned_orders.append(("SELL", sym_pair, qty))

    executions = []
    if not planned_orders:
        print("Ingen rebalans behövs (inom band eller små poster).")
    else:
        for side, sym_pair, amount in planned_orders:
            if args.dry_run or ccxt is None:
                if side == "BUY":
                    print(f"[DryRun] Skulle KÖPA {sym_pair} för ~{amount:.2f} {quote_ccy}")
                else:
                    print(f"[DryRun] Skulle SÄLJA {amount:.6f} av {sym_pair.split('/')[0]} (≈{amount*prices[sym_pair]:.2f} {quote_ccy})")
                executions.append({"side": side, "symbol": sym_pair, "amount": amount, "order_id": None})
            else:
                broker = broker  # already exists
                if side == "BUY":
                    order = broker.market_buy_quote(sym_pair, quote_amount=amount)
                else:
                    order = broker.market_sell_base(sym_pair, base_qty=amount)
                executions.append({"side": side, "symbol": sym_pair, "amount": amount, "order_id": order.get("id") if isinstance(order, dict) else str(order)})
                print(f"Order {side} {sym_pair}: {executions[-1]['order_id']}")

    # Loggning
    now = datetime.now(ZoneInfo("Europe/Stockholm")).strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "time": now,
        "symbols": args.symbols,
        "signals": json.dumps(signals),
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

    portfolio_snapshot = {
        "quote": quote_ccy,
        "prices": {s: prices[s] for s in syms_list},
        "executions": executions,
        "rebalance_reason": rb.reason,
        "signals": signals,
        "momentum_scores": mom_scores,
        "targets_pct": rb.target_allocations,
        "trades_pp": rb.trades,
        "current_alloc_pct": current_alloc,
    }
    Path(args.portfolio).write_text(json.dumps(portfolio_snapshot, indent=2, ensure_ascii=False))
    print("Uppdaterade portfolio.json och trades_log.csv.")

if __name__ == "__main__":
    main()
