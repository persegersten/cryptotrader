#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hämtar 30 dagars OHLC + volym för en given coin via CoinGecko och sparar till CSV.
- Enbart "valutan" som input (teknisk data): tid, open, high, low, close, volume.
- Filnamn taggas med YYYYMMDD_hhmmss i Europe/Stockholm.
"""

import sys
import csv
import time
import math
import requests
from datetime import datetime, UTC
from zoneinfo import ZoneInfo  # Python 3.9+
from pathlib import Path

def fetch_ohlc(coin_id: str, vs_currency: str, days: int):
    """
    CoinGecko OHLC endpoint: list of [timestamp, open, high, low, close]
    Tidsstämplar i millisekunder (ms).
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": vs_currency, "days": days}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Säkerställ list-format
    if not isinstance(data, list):
        raise ValueError(f"OHLC-svar oväntat: {data}")
    return data

def fetch_volumes(coin_id: str, vs_currency: str, days: int):
    """
    CoinGecko market_chart: total_volumes => list of [timestamp(ms), volume]
    För <= 90 dagar brukar det vara ~timupplösning.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    vols = j.get("total_volumes", [])
    if not isinstance(vols, list):
        raise ValueError(f"Volym-svar oväntat: {j}")
    return vols

def ms_to_iso(ms: int) -> str:
    # Vi behåller ISO i UTC för modeller/återuppspelning
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def nearest_volume(ts_ms: int, vol_rows, tol_ms: int = 60 * 60 * 1000):
    """
    Hitta närmaste volym-observation (± tol_ms) till en OHLC-tid.
    CoinGecko OHLC och volumes har inte exakt samma sampling → “nearest join”.
    """
    # Binärsökning hade varit snabbare, men detta duger fint för ~720 punkter
    best = None
    best_diff = tol_ms + 1
    for t_ms, v in vol_rows:
        diff = abs(t_ms - ts_ms)
        if diff < best_diff:
            best_diff = diff
            best = (t_ms, v)
    if best and best_diff <= tol_ms:
        return best[1]
    return None  # saknar närliggande volym

def main(coin_id :str = 'ethereum', vs_currency: str = 'usd', days :int = 30, data_folder :str = './kursdata', history_folder :str = './history'):
    # 1) Hämta data
    ohlc = fetch_ohlc(coin_id, vs_currency, days)
    volumes = fetch_volumes(coin_id, vs_currency, days)

    # 2) Förbered volym-lista (t_ms, vol as float)
    vol_rows = [(int(row[0]), float(row[1])) for row in volumes]

    # 3) Bygg rader: timestamp_iso, open, high, low, close, volume
    rows = []
    for row in ohlc:
        # row = [t_ms, o, h, l, c]
        t_ms = int(row[0])
        o, h, l, c = map(float, row[1:5])
        vol = nearest_volume(t_ms, vol_rows)  # float eller None
        rows.append({
            "timestamp": ms_to_iso(t_ms),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": vol if vol is not None else ""
        })

    # 4) Filnamn med Stockholm-stämpel
    now_se = datetime.now(ZoneInfo("Europe/Stockholm"))
    tag = now_se.strftime("%Y%m%d_%H%M%S")
    fname = f"crypto_{coin_id}_{vs_currency}_{tag}.csv"
    
    # Skapa katalogen om den inte finns
    DATA_PATH = Path(data_folder)
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    # Sätt filnamn med full sökväg
    out = (DATA_PATH / fname).resolve()

    # 5) Skriv CSV
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp","open","high","low","close","volume"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Sparat {len(rows)} rader till: {out}")

if __name__ == "__main__":
    # Liten retry för att vara snäll mot gratis-API:et ifall rate limit
    tries = 2
    for i in range(tries):
        try:
            main()
            break
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429 and i < tries - 1:
                wait_s = 3
                print(f"429 Too Many Requests – försöker igen om {wait_s}s…")
                time.sleep(wait_s)
                continue
            raise
