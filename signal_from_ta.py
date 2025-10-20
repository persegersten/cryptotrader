#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
signal_from_ta.py

Läser en OHLCV-CSV (minst 'close', gärna 'open','high','low','volume','timestamp'),
beräknar/återanvänder indikatorer och avgör köp/sälj/neutral på SENASTE datapunkt.

Regler (enkelt, transparent “ensemble”):
- RSI(14): <30 => bullish +1, >70 => bearish -1, annars 0
- EMA cross: EMA12 > EMA26 => +1, EMA12 < EMA26 => -1
- Crossover boost: om korsning skett senaste baren (EMA12 korsar EMA26) => ±1 extra
- MACD(12,26,9): MACD > signal => +1, < => -1
- Pris vs EMA200: close > EMA200 => +1 (trendfilter), < => -1
- Volym-spike (om volume finns): om volym > 1.5 × 20d-snittsvolym och signal bullish/bearish => +1 förstärkning

Summa-poäng:
  score >= +2 => KÖP
  score <= -2 => SÄLJ
  annars => NEUTRAL

Exempel:
    python3 signal_from_ta.py path/to/crypto_bitcoin_usd_20251015_143022.csv
    python3 signal_from_ta.py data.csv --close-col Close --timestamp-col Date
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def ensure_float(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--open-col", default="open")
    ap.add_argument("--high-col", default="high")
    ap.add_argument("--low-col", default="low")
    ap.add_argument("--close-col", default="close")
    ap.add_argument("--volume-col", default="volume")
    args = ap.parse_args()

    path = Path(args.input_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    # Läs och normalisera
    parse_dates = [args.timestamp_col] if args.timestamp_col in pd.read_csv(path, nrows=0).columns else []
    try:
        df = pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        df = pd.read_csv(path)

    for c in [args.open_col, args.high_col, args.low_col, args.close_col, args.volume_col]:
        ensure_float(df, c)

    if args.close_col not in df.columns:
        raise ValueError(f"Saknar kolumn '{args.close_col}' i {path.name}")

    # Sortera kronologiskt om inte redan
    if args.timestamp_col in df.columns:
        df = df.sort_values(by=args.timestamp_col).reset_index(drop=True)

    close = df[args.close_col]

    # Beräkna indikatorer (bara om inte redan finns)
    if "rsi_14" not in df.columns:
        df["rsi_14"] = rsi_wilder(close, 14)
    if "ema_12" not in df.columns:
        df["ema_12"] = ema(close, 12)
    if "ema_26" not in df.columns:
        df["ema_26"] = ema(close, 26)
    if "ema_200" not in df.columns:
        df["ema_200"] = ema(close, 200)

    if not {"macd","macd_signal","macd_hist"}.issubset(df.columns):
        macd_line, sig_line, hist = macd(close, 12, 26, 9)
        df["macd"] = macd_line
        df["macd_signal"] = sig_line
        df["macd_hist"] = hist

    # Volymglidande medel för spike-detektion
    vol_spike = False
    if args.volume_col in df.columns:
        vol = df[args.volume_col]
        df["vol_ma20"] = vol.rolling(20, min_periods=1).mean()
        if pd.notna(vol.iloc[-1]) and pd.notna(df["vol_ma20"].iloc[-1]):
            vol_spike = vol.iloc[-1] > 1.5 * df["vol_ma20"].iloc[-1]

    # Senaste rad
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    # Poängregler
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

    # EMA trend
    e12, e26 = last["ema_12"], last["ema_26"]
    if pd.notna(e12) and pd.notna(e26):
        if e12 > e26:
            score += 1; reasons.append("EMA12 > EMA26 (trend upp)")
        elif e12 < e26:
            score -= 1; reasons.append("EMA12 < EMA26 (trend ned)")

        # Crossover boost
        if prev is not None:
            p12, p26 = prev["ema_12"], prev["ema_26"]
            if pd.notna(p12) and pd.notna(p26):
                if p12 <= p26 and e12 > e26:
                    score += 1; reasons.append("Bullish crossover nyligen (EMA12 korsar upp)")
                if p12 >= p26 and e12 < e26:
                    score -= 1; reasons.append("Bearish crossover nyligen (EMA12 korsar ned)")

    # MACD momentum
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

    # Volym-spike förstärker existerande riktning
    if vol_spike:
        if score > 0:
            score += 1; reasons.append("Volym-spike bekräftar köpsignal")
        elif score < 0:
            score -= 1; reasons.append("Volym-spike bekräftar säljsignal")
        else:
            reasons.append("Volym-spike utan tydlig riktning")

    # Beslut
    if score >= 2:
        decision = "KÖP"
    elif score <= -2:
        decision = "SÄLJ"
    else:
        decision = "NEUTRAL"

    # Utskrift
    ts = last.get(args.timestamp_col, "")
    ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    print("=== Beslut på senaste datapunkt ===")
    print(f"Tid: {ts_str}")
    print(f"Close: {close.iloc[-1]:.8f}" if pd.notna(close.iloc[-1]) else "Close: n/a")
    print(f"Signal: {decision}  (score {score})")
    print("Skäl:")
    for r in reasons:
        print(f" - {r}")

if __name__ == "__main__":
    main()
