#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ta_signal_agent.py

En enkel TA-driven "agent" för pappershandel som läser en CSV (OHLCV), räknar signal
(KÖP/SÄLJ/NEUTRAL) och uppdaterar en portfölj (JSON) enligt en enkel allokeringsstrategi.

Regler (default):
- KÖP: sikta på buy_alloc (t.ex. 20% av portföljens totala värde i den här tillgången)
- SÄLJ: sikta på 0% (sälj allt i denna tillgång)
- NEUTRAL: gör inget

Handlar till senaste closepris i CSV, med transaktionsavgift fee_bps (baspunkter).
Loggar affärer till trades_log.csv.

Exempel:
    python3 ta_signal_agent.py --csv data/crypto_bitcoin_usd_20251015_143022.csv --asset BTC --portfolio portfolio.json
    python3 ta_signal_agent.py --csv eth.csv --asset ETH --portfolio portfolio.json --buy-alloc 0.25 --fee-bps 8
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def ensure_float(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi_wilder(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def decide_signal(df, close_col="close"):
    """Returnerar (decision, score, reasons) för senaste datapunkt."""
    # Säkerställ indikatorer
    close = df[close_col]
    if "rsi_14" not in df.columns:
        df["rsi_14"] = rsi_wilder(close, 14)
    if "ema_12" not in df.columns:
        df["ema_12"] = ema(close, 12)
    if "ema_26" not in df.columns:
        df["ema_26"] = ema(close, 26)
    if "ema_200" not in df.columns:
        df["ema_200"] = ema(close, 200)
    if not {"macd", "macd_signal", "macd_hist"}.issubset(df.columns):
        macd_line, sig_line, hist = macd(close, 12, 26, 9)
        df["macd"] = macd_line
        df["macd_signal"] = sig_line
        df["macd_hist"] = hist

    # Volym-spike (valfritt)
    vol_spike = False
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        df["vol_ma20"] = vol.rolling(20, min_periods=1).mean()
        if pd.notna(vol.iloc[-1]) and pd.notna(df["vol_ma20"].iloc[-1]):
            vol_spike = vol.iloc[-1] > 1.5 * df["vol_ma20"].iloc[-1]

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    score = 0
    reasons = []

    # RSI
    rsi = last["rsi_14"]
    if pd.notna(rsi):
        if rsi < 30:
            score += 1; reasons.append(f"RSI(14) {rsi:.1f} (<30) bullish")
        elif rsi > 70:
            score -= 1; reasons.append(f"RSI(14) {rsi:.1f} (>70) bearish")
        else:
            reasons.append(f"RSI(14) {rsi:.1f} neutral")

    # EMA trend + crossover
    e12, e26 = last["ema_12"], last["ema_26"]
    if pd.notna(e12) and pd.notna(e26):
        if e12 > e26:
            score += 1; reasons.append("EMA12 > EMA26 (trend upp)")
        elif e12 < e26:
            score -= 1; reasons.append("EMA12 < EMA26 (trend ned)")

        if prev is not None:
            p12, p26 = prev["ema_12"], prev["ema_26"]
            if pd.notna(p12) and pd.notna(p26):
                if p12 <= p26 and e12 > e26:
                    score += 1; reasons.append("Bullish crossover nyligen (EMA12 korsar upp)")
                if p12 >= p26 and e12 < e26:
                    score -= 1; reasons.append("Bearish crossover nyligen (EMA12 korsar ned)")

    # MACD
    if pd.notna(last["macd"]) and pd.notna(last["macd_signal"]):
        if last["macd"] > last["macd_signal"]:
            score += 1; reasons.append("MACD över signal (bullish momentum)")
        elif last["macd"] < last["macd_signal"]:
            score -= 1; reasons.append("MACD under signal (bearish momentum)")

    # Lång trendfilter
    if pd.notna(last["ema_200"]) and pd.notna(close.iloc[-1]):
        if close.iloc[-1] > last["ema_200"]:
            score += 1; reasons.append("Pris > EMA200 (långsiktigt bullish)")
        else:
            score -= 1; reasons.append("Pris < EMA200 (långsiktigt bearish)")

    # Volym-spike förstärker
    if vol_spike:
        if score > 0:
            score += 1; reasons.append("Volym-spike bekräftar köpsignal")
        elif score < 0:
            score -= 1; reasons.append("Volym-spike bekräftar säljsignal")
        else:
            reasons.append("Volym-spike utan tydlig riktning")

    if score >= 2:
        decision = "BUY"
    elif score <= -2:
        decision = "SELL"
    else:
        decision = "HOLD"

    return decision, int(score), reasons


def load_portfolio(path: Path):
    if not path.exists():
        # Initiera tom portfölj
        return {"cash": 0.0, "positions": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_portfolio(path: Path, portfolio: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)


def total_equity(portfolio: dict, price_map: dict):
    equity = float(portfolio.get("cash", 0.0))
    for asset, units in portfolio.get("positions", {}).items():
        px = price_map.get(asset)
        if px is not None:
            equity += units * px
    return equity


def execute_trade(portfolio: dict, asset: str, price: float, target_alloc: float,
                  min_trade_value: float, fee_bps: float):
    """
    Justerar portföljen mot target_alloc av total equity för given asset.
    Returnerar (executed, side, qty, value_before_fee, fee_paid, new_cash).
    """
    positions = portfolio.setdefault("positions", {})
    units = float(positions.get(asset, 0.0))
    cash = float(portfolio.get("cash", 0.0))

    # Bygg price map för equity
    price_map = {asset: price}
    eq = total_equity(portfolio, price_map)
    target_value = max(0.0, target_alloc) * eq
    current_value = units * price
    diff_value = target_value - current_value

    # Ingen trade om diff är liten
    if abs(diff_value) < max(1e-9, min_trade_value):
        return False, "NONE", 0.0, 0.0, 0.0, cash

    if diff_value > 0:
        # Köp för diff_value (men begränsa av cash)
        buy_value = min(diff_value, cash)
        if buy_value < min_trade_value:
            return False, "NONE", 0.0, 0.0, 0.0, cash
        qty = buy_value / price
        fee = buy_value * (fee_bps / 10000.0)
        cash_after = cash - buy_value - fee
        positions[asset] = units + qty
        portfolio["cash"] = cash_after
        return True, "BUY", qty, buy_value, fee, cash_after
    else:
        # Sälj för -diff_value (men begränsa av innehav)
        sell_value_needed = -diff_value
        qty_needed = sell_value_needed / price
        qty = min(qty_needed, units)
        if qty * price < min_trade_value or qty <= 0:
            return False, "NONE", 0.0, 0.0, 0.0, cash
        value = qty * price
        fee = value * (fee_bps / 10000.0)
        cash_after = cash + value - fee
        positions[asset] = units - qty
        portfolio["cash"] = cash_after
        return True, "SELL", qty, value, fee, cash_after


def main():
    ap = argparse.ArgumentParser(description="TA-signalagent för pappershandel.")
    ap.add_argument("--csv", required=True, help="Sökväg till OHLCV-CSV för en tillgång.")
    ap.add_argument("--asset", required=True, help="Tillgångsnamn/symbol i portföljen, t.ex. BTC eller ETH.")
    ap.add_argument("--portfolio", required=True, help="Sökväg till portfolio.json (skapas om den saknas).")
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--close-col", default="close")
    ap.add_argument("--buy-alloc", type=float, default=0.20, help="Målfördelning vid BUY (0..1). Default 0.20")
    ap.add_argument("--sell-alloc", type=float, default=0.0, help="Målfördelning vid SELL. Default 0.0")
    ap.add_argument("--hold-alloc", type=float, default=None, help="Målfördelning vid HOLD. Default None (ingen förändring)")
    ap.add_argument("--min-trade", type=float, default=10.0, help="Minsta affärsvärde i portföljvalutan. Default 10.0")
    ap.add_argument("--fee-bps", type=float, default=10.0, help="Avgift i bps (10 bps = 0.10%). Default 10")
    ap.add_argument("--log", default="trades_log.csv", help="Fil för affärslogg.")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # Läs CSV
    parse_dates = [args.timestamp_col] if args.timestamp_col in pd.read_csv(csv_path, nrows=0).columns else []
    try:
        df = pd.read_csv(csv_path, parse_dates=parse_dates, infer_datetime_format=True)
    except Exception:
        df = pd.read_csv(csv_path)

    # Sortera kronologiskt om tidskolumn finns
    if args.timestamp_col in df.columns:
        df = df.sort_values(by=args.timestamp_col).reset_index(drop=True)

    # Säkerställ numerik
    ensure_float(df, args.close_col)

    if args.close_col not in df.columns:
        raise ValueError(f"Saknar '{args.close_col}' i {csv_path.name}")

    decision, score, reasons = decide_signal(df, close_col=args.close_col)

    last = df.iloc[-1]
    ts = last.get(args.timestamp_col, "")
    ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    price = float(last[args.close_col])

    # Ladda/Initiera portfölj
    portfolio_path = Path(args.portfolio).expanduser().resolve()
    portfolio = load_portfolio(portfolio_path)

    # Säkerställ nycklar
    portfolio.setdefault("cash", 0.0)
    portfolio.setdefault("positions", {})

    # Bestäm target allocation
    if decision == "BUY":
        target_alloc = args.buy_alloc
    elif decision == "SELL":
        target_alloc = args.sell_alloc
    else:  # HOLD
        target_alloc = args.hold_alloc  # None -> ingen förändring

    executed = False
    side = "NONE"
    qty = value = fee = 0.0

    if target_alloc is not None:
        executed, side, qty, value, fee, new_cash = execute_trade(
            portfolio=portfolio,
            asset=args.asset,
            price=price,
            target_alloc=target_alloc,
            min_trade_value=args.min_trade,
            fee_bps=args.fee_bps
        )
        if executed:
            save_portfolio(portfolio_path, portfolio)

            # Logg
            log_path = Path(args.log).expanduser().resolve()
            log_exists = log_path.exists()
            now_se = datetime.now(ZoneInfo("Europe/Stockholm")).strftime("%Y-%m-%d %H:%M:%S%z")
            row = {
                "time_agent_se": now_se,
                "data_timestamp": ts_str,
                "asset": args.asset,
                "decision": decision,
                "score": score,
                "price": price,
                "side": side,
                "qty": qty,
                "value_before_fee": value,
                "fee_bps": args.fee_bps,
                "fee": fee,
                "cash_after": portfolio.get("cash", None),
                "reason": " | ".join(reasons)
            }
            import csv
            with log_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not log_exists:
                    writer.writeheader()
                writer.writerow(row)

    # Utskrift
    print("=== TA-Agent beslut ===")
    print(f"CSV: {csv_path.name}")
    print(f"Tid (data): {ts_str}")
    print(f"Pris (close): {price:.8f}")
    print(f"Signal: {decision} (score {score})")
    print("Skäl:")
    for r in reasons:
        print(f" - {r}")
    if target_alloc is None:
        print("Policy: HOLD → ingen rebalansering (target_alloc=None).")
    else:
        print(f"Policy: target_alloc={target_alloc:.2%}")
    if executed:
        print(f"Trade: {side} qty={qty:.8f}, värde≈{value:.2f}, avgift≈{fee:.2f}")
        print(f"Ny kassa: {portfolio.get('cash'):,.2f}")
    else:
        print("Ingen trade (antingen HOLD, diff för liten, för lite kassa, eller inget att sälja).")


if __name__ == "__main__":
    main()
