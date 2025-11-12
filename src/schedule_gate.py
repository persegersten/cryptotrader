#!/usr/bin/env python3
"""
schedule_gate.py – exit 0 om det är "rätt timme" att köra jobbet, annars exit 95.
Användbar i shell-script för att tidigt avbryta körningen.

Exempel:
  python schedule_gate.py --every 4 --tz UTC
  python schedule_gate.py --at 0 4 8 12 16 20 --tz Europe/Stockholm
  FORCE_RUN=1 python schedule_gate.py  # bypass – kör alltid

Exit codes:
  0  = OK, kör jobbet nu
  95 = Inte rätt timme, hoppa över
  64 = Ogiltiga args
"""

import argparse
import os
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

SKIP_EXIT_CODE = 95  # custom non-zero för "inte nu"

def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--every", type=int, help="Kör var N:e timme (t.ex. 4)")
    group.add_argument("--at", nargs="+", type=int, help="Lista timmar som är tillåtna (0–23), t.ex. 0 4 8 12 16 20")
    p.add_argument("--offset", type=int, default=0, help="Tim-offset för --every (default 0). Ex: (hour - offset) %% every == 0")
    p.add_argument("--tz", default="UTC", help="Tidszon, t.ex. UTC eller Europe/Stockholm (default UTC)")
    p.add_argument("--grace-minutes", type=int, default=5, help="Tillåt fönster i minuter runt hel timme (default 5)")
    return p.parse_args()

def now_in_tz(tz_name: str) -> datetime:
    if ZoneInfo is None:
        # Fallback – beter sig som UTC om zoneinfo saknas
        return datetime.utcnow()
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        # Okänd tz – fall tillbaka till UTC
        return datetime.utcnow()

def main():
    # Bypass med env
    if os.getenv("FORCE_RUN") in ("1", "true", "True", "YES", "yes"):
        print("FORCE_RUN är satt – kör ändå.")
        return 0

    args = parse_args()
    now = now_in_tz(args.tz)
    hour = now.hour
    minute = now.minute

    # Litet nådefönster runt hel timme (scheduler kan starta några min sent/tidigt)
    within_grace = (minute <= args.grace_minutes) or (minute >= (60 - args.grace_minutes))

    should_run = False
    reason = ""

    if args.every is not None:
        if args.every <= 0:
            print("Ogiltigt värde för --every (måste vara > 0).", flush=True)
            return 64
        should_run = ((hour - args.offset) % args.every == 0) and within_grace
        reason = f"(hour={hour}, offset={args.offset}, every={args.every}, minute={minute}, grace={args.grace_minutes}, tz={args.tz})"
    else:
        # --at lista
        valid_hours = [h % 24 for h in args.at]
        should_run = (hour in valid_hours) and within_grace
        reason = f"(hour={hour}, allowed={valid_hours}, minute={minute}, grace={args.grace_minutes}, tz={args.tz})"

    if should_run:
        print(f"OK_TO_RUN {reason}")
        return 0
    else:
        print(f"SKIP {reason}")
        return SKIP_EXIT_CODE

if __name__ == "__main__":
    raise SystemExit(main())
