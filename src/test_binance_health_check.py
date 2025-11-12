#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binance Spot Health Check – verifierar symboltillgång, permissions och minsta orderstorlek.
- Kollar defaultType=spot, fetch_balance, load_markets
- För varje symbol: active/status/permissions/filters + beräknar minsta order som klarar minNotional
- Lägger INGA ordrar (helt säkert). Använd detta innan du kör din bot live.

Kör:
  export CCXT_API_KEY='...'
  export CCXT_API_SECRET='...'
  python3 binance_spot_healthcheck.py --symbols "BNB/USDT,BTC/USDT,ETH/USDT,SOL/USDT"

Flaggor:
  --symbols          Komma-separerade spot-par (default: BNB/USDT,BTC/USDT,ETH/USDT,SOL/USDT)
  --show-options     Skriv ut exchange.options (avkortad)
  --no-balance       Hoppa över fetch_balance (om du bara vill testa marknad/permissions)
"""

import os
import sys
import math
import json
import argparse
from typing import Dict, Any
import ccxt
from pathlib import Path

REQUIRED_ENV = ("CCXT_API_KEY", "CCXT_API_SECRET")

def load_secrets() -> None:
    file_path = "secrets.json"
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

def ceil_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.ceil(x / step) * step

def fmt_num(x, n=8):
    try:
        return f"{float(x):,.{n}f}"
    except Exception:
        return str(x)

def extract_filters(info: Dict[str, Any]) -> Dict[str, Any]:
    res = {
        "minNotional": None,
        "tickSize": None,
        "minQty": None,
        "stepSize": None,
        "maxMarketQty": None,
    }
    for f in info.get("filters", []):
        t = f.get("filterType")
        if t == "NOTIONAL":
            res["minNotional"] = float(f.get("minNotional", 0))
        elif t == "PRICE_FILTER":
            res["tickSize"] = float(f.get("tickSize", 0))
        elif t == "LOT_SIZE":
            res["minQty"] = float(f.get("minQty", 0))
            res["stepSize"] = float(f.get("stepSize", 0))
        elif t == "MARKET_LOT_SIZE":
            res["maxMarketQty"] = float(f.get("maxQty", 0))
    return res

def min_amount_to_meet_notional(price: float, min_notional: float, step_size: float, min_qty: float) -> float:
    if price <= 0:
        return None
    raw = min_notional / price
    stepped = ceil_to_step(raw, step_size if step_size else 0)
    if min_qty:
        stepped = max(stepped, min_qty)
    return stepped

def main():
    load_secrets()

    ap = argparse.ArgumentParser(description="Binance Spot Health Check")
    ap.add_argument("--symbols", default="BNB/USDC,BTC/USDC,ETH/USDC,SOL/USDC,BNB/USDT,BTC/USDT,ETH/USDT,SOL/USDT")
    ap.add_argument("--show-options", action="store_true")
    ap.add_argument("--no-balance", action="store_true")
    args = ap.parse_args()

    api_key = os.getenv("CCXT_API_KEY")
    api_secret = os.getenv("CCXT_API_SECRET")
    if not api_key or not api_secret:
        sys.exit("❌ Sätt CCXT_API_KEY och CCXT_API_SECRET i miljön först.")

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    # (valfritt) Visa options
    if args.show_options:
        opts = dict(exchange.options)  # gör en kopia för snygg print
        # korta ner för läsbarhet
        for k in ("crossMarginPairsData", "isolatedMarginPairsData"):
            if k in opts:
                opts[k] = f"[{k} length = {len(opts[k])}]"
        print("Exchange options (avkortad):")
        print(json.dumps(opts, indent=2, ensure_ascii=False))

    # 1) marknader
    try:
        markets = exchange.load_markets()
        print(f"✅ load_markets OK – {len(markets)} symboler laddade.")
    except Exception as e:
        sys.exit(f"❌ load_markets fel: {e}")

    # 2) saldo (valfritt)
    if not args.no_balance:
        try:
            bal = exchange.fetch_balance()
            total_keys = list((bal.get("total") or {}).keys())
            print(f"✅ fetch_balance OK – total keys: {len(total_keys)} (ex: {total_keys[:8]})")
        except Exception as e:
            print(f"⚠️ fetch_balance misslyckades: {e}")

    # 3) kolla varje symbol
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print("\n=== Symboldiagnos ===")
    for sym in symbols:
        print(f"\n— {sym} —")
        if sym not in markets:
            print("  ❌ Finns ej i load_markets() – fel symbol eller ej spot-par.")
            continue

        m = markets[sym]
        info = m.get("info", {})
        active = m.get("active")
        status = info.get("status")
        perms = info.get("permissions") or []  # kan vara []
        perm_sets = info.get("permissionSets")  # nyare fält

        print(f"  active: {active} | status: {status}")
        print(f"  type: spot={m.get('spot')} margin={m.get('margin')} future={m.get('future')} swap={m.get('swap')}")
        print(f"  permissions: {perms if perms else '[] (tom)'}")
        if perm_sets:
            # skriv bara första setet kort
            first = perm_sets[0] if isinstance(perm_sets, list) and perm_sets else None
            if first:
                print(f"  permissionSets (exempel): {first[:6]}... (tot {len(first)} taggar i setet)")

        # filters
        flt = extract_filters(info)
        print(f"  filters: NOTIONAL(min)={flt['minNotional']}  LOT_SIZE(minQty)={flt['minQty']} step={flt['stepSize']}  PRICE_FILTER(tickSize)={flt['tickSize']}")

        # pris
        try:
            t = exchange.fetch_ticker(sym)
            last = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")
            print(f"  ticker last≈ {fmt_num(last, 2)}")
        except Exception as e:
            print(f"  ⚠️ fetch_ticker fel: {e}")
            last = None

        # beräkna min ordermängd som klarar NOTIONAL (om vi har pris)
        if last and flt["minNotional"] is not None:
            amt = min_amount_to_meet_notional(float(last), flt["minNotional"], flt["stepSize"] or 0.0, flt["minQty"] or 0.0)
            if amt:
                try:
                    amt_precise = exchange.amount_to_precision(sym, amt)
                except Exception:
                    amt_precise = amt
                notional = float(last) * float(amt_precise)
                print(f"  ▶ Minsta market-order som klarar NOTIONAL ≈ qty={amt_precise} {m['base']}  (värde≈ {fmt_num(notional, 2)} {m['quote']})")
        else:
            print("  ⚠️ Kan inte räkna min order (saknar pris eller NOTIONAL).")

        # heuristik: varna om permissions [] (tyst spärr hos Binance)
        if not perms:
            print("  ⚠️ permissions=[] → Din API-nyckel/konto returnerar inga symbolrättigheter.")
            print("     Vanliga orsaker: region/EU-whitelist, symbol-whitelist på API-nyckeln, eller Binance intern TRD_GRP-mappning.")
            print("     Rek: Skapa NY spot-API-nyckel (Enable Spot & Margin Trading), testa utan proxy/IP-restriktioner, eller kontakta Binance support.")

    print("\n✅ Klart. Om någon symbol visar -2010 vid order: det är nästan alltid permissions/region. Detta script visar varför.")

if __name__ == "__main__":
    main()
