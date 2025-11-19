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

def now_in_tz(tz_name: str) -> datetime:
    if ZoneInfo is None:
        # Fallback – beter sig som UTC om zoneinfo saknas
        return datetime.utcnow()
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        # Okänd tz – fall tillbaka till UTC
        return datetime.utcnow()

def run(grace_minutes: int, at_hours : list[int], time_zone: str):
    # Bypass med env
    if os.getenv("FORCE_RUN") in ("1", "true", "True", "YES", "yes"):
        print("FORCE_RUN är satt – kör!!!")
        return True

    now = now_in_tz(time_zone)
    hour = now.hour
    minute = now.minute

    # Litet nådefönster runt hel timme (scheduler kan starta några min sent/tidigt)
    within_grace = (minute <= grace_minutes) or (minute >= (60 - grace_minutes))

    should_run = False
    reason = ""

    # --at lista
    valid_hours = [h % 24 for h in at_hours]
    should_run = (hour in valid_hours) and within_grace
    reason = f"(hour={hour}, allowed={valid_hours}, minute={minute}, grace={grace_minutes}, tz={time_zone})"

    if should_run:
        print(f"OK_TO_RUN {reason}")
        return True
    else:
        print(f"SKIP {reason}")
        return False

if __name__ == "__main__":
    print("Hi here")
