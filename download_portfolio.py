#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
download_portfolio.py

Hämtar aktuell portfölj (balanser) från Binance via CCXT.
- Läser API-nycklar från env eller secrets.json
- Stöd för testnet via flaggan --sandbox
- Skriver ut likvida medel (free balance i quote-valutor)
- Sparar hela portföljen till portfolio.json
"""

import os
import json
from pathlib import Path
import argparse
import ccxt

def load_secrets_if_missing(file_path="secrets.json"):
    """Sätter endast env-variabler om de inte redan finns."""
    p = Path(file_path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if k not in os.environ or not os.environ[k]:
                os.environ[k] = str(v)
    except Exception as e:
        print(f"⚠️  Kunde inte läsa {file_path}: {e}")

def main():
    ap = argparse.ArgumentParser(description="Ladda ner Binance-portfölj")
    ap.add_argument("--sandbox", action="store_true", help="Kör mot Binances testnet (låtsaspengar)")
    ap.add_argument("--out", default="portfolio.json", help="Filnamn för export (default: portfolio.json)")
    ap.add_argument("--quote", default="USDT", help="Quote-valuta att betrakta som likvida medel (default: USDT)")
    args = ap.parse_args()

    # Ladda ev. hemligheter från fil
    load_secrets_if_missing("secrets.json")

    # Läs API-nycklar
    api_key = os.getenv("CCXT_API_KEY")
    api_secret = os.getenv("CCXT_API_SECRET")

    if not api_key or not api_secret:
        raise SystemExit(
            "Saknar API-nycklar. Sätt miljövariabler CCXT_API_KEY/CCXT_API_SECRET "
            "eller skapa secrets.json."
        )

    # Initiera Binance
    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })

    if args.sandbox:
        exchange.set_sandbox_mode(True)
        print("🧪 Använder Binance testnet")

    # Hämta balans
    print("🔄 Hämtar portfölj...")
    balance = exchange.fetch_balance()

    # Extrahera användbara delar
    portfolio = {
        "timestamp": exchange.milliseconds(),
        "exchange": "binance-testnet" if args.sandbox else "binance",
        "free": balance.get("free", {}),
        "used": balance.get("used", {}),
        "total": balance.get("total", {}),
    }

    # Filtrera bort nollrader för snyggare visning
    nonzero = {k: v for k, v in portfolio["total"].items() if v and v > 0}
    print("\n📊 Dina tillgångar:")
    if nonzero:
        for asset, amount in sorted(nonzero.items()):
            print(f"  {asset:<6} {amount:.8f}")
    else:
        print("  (Inga tillgångar hittades)")

    # Likvida medel i quote (t.ex. USDT)
    quote = args.quote.upper()
    free_bal = portfolio["free"].get(quote, 0.0)
    print(f"\n💰 Likvida medel: {free_bal:.2f} {quote}")

    # Spara till fil
    out_path = Path(args.out)
    out_path.write_text(json.dumps(portfolio, indent=2, ensure_ascii=False))
    print(f"\n💾 Portföljen sparad till: {out_path.resolve()}")

if __name__ == "__main__":
    main()
