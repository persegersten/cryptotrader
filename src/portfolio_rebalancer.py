# portfolio_rebalancer.py
from dataclasses import dataclass
from typing import Dict, List, Tuple

Signal = str  # 'BUY' | 'SELL' | 'HOLD'

@dataclass(frozen=True)
class RebalanceResult:
    target_allocations: Dict[str, float]  # tillgång->% av total (0..100), USD inkluderas som 'USD'
    trades: Dict[str, float]              # tillgång-> procentenheter att köpa (+) / sälja (-)
    reason: str                           # kort beskrivning av beslutet


def _normalize_allocs(allocs: Dict[str, float]) -> Dict[str, float]:
    s = sum(allocs.values())
    if s == 0:
        return {k: 0.0 for k in allocs}
    return {k: (v * 100.0 / s) for k, v in allocs.items()}


def rebalance_three(
    symbols: List[str],
    signals: Dict[str, Signal],
    current_allocations: Dict[str, float],
    *,
    usd_key: str = "USD",
    tolerance: float = 0.10,          # ±10% band kring målvikter
    only_if_outside_band: bool = True # om True: rör inte portföljen när den redan ligger inom bandet
) -> RebalanceResult:
    """
    Rebalanserar tre krypto + USD enligt regler:
      - SELL => mål 0% för den tillgången.
      - Övriga aktiva (BUY/HOLD) delas lika mellan sig (100% / antal aktiva).
      - USD blir residualen: 100 - sum(aktiva mål).
      - Band: om portfölj redan ligger inom mål ± (tolerance * mål), gör inget (om only_if_outside_band=True).

    Parametrar:
      symbols: exakt 3 tickers, t.ex. ["BTC","ETH","SOL"]
      signals: mapping symbol->"BUY"/"SELL"/"HOLD"
      current_allocations: mapping med 3 symboler + ev. 'USD' (procent 0..100). Summan behöver inte vara exakt 100.
      tolerance: 0.10 betyder ±10% av respektive MÅLVIKT (absolut procentenheter = mål * tolerance).
                 Ex: mål = 33.33% => band ≈ [30.0%, 36.67%]

    Returnerar:
      RebalanceResult med målvikter, föreslagna trades (i procentenheter) och en kort reason.

    Exempel (från dina cases):
      SELL, SELL, BUY  -> (0,0,100)
      SELL, BUY, HOLD  -> (0,50,50)
      BUY, HOLD, BUY   -> (33.33,33.33,33.33)
      SELL, SELL, HOLD -> (0,0,100)
      (55,10,0) + HOLD,HOLD,SELL -> (50,50,0)
    """
    if len(symbols) != 3:
        raise ValueError("symbols måste ha exakt tre tickers.")
    for s in symbols:
        if s not in signals:
            raise ValueError(f"Saknar signal för {s}")

    # Normalisera nuvarande vikter över (3 krypto + ev USD)
    all_keys = set(symbols) | {usd_key}
    cur = {k: float(current_allocations.get(k, 0.0)) for k in all_keys}
    cur = _normalize_allocs(cur)

    # Bestäm aktiva (ej SELL)
    active = [s for s in symbols if signals.get(s, "HOLD").upper() != "SELL"]
    n_active = len(active)

    # Om allt är SELL -> allt till USD
    target: Dict[str, float] = {}
    if n_active == 0:
        target = {s: 0.0 for s in symbols}
        target[usd_key] = 100.0
        reason = "Alla tre har SELL → allt till USD."
    else:
        # Lika vikt mellan aktiva, 0 för SELL
        per = 100.0 / n_active
        target = {s: (per if s in active else 0.0) for s in symbols}
        # USD är residual (bör bli 0 om minst en aktiv finns, men säkrar numerik)
        target[usd_key] = max(0.0, 100.0 - sum(target[s] for s in symbols))

        # Band-regel: rör inte om alla aktiva ligger inom band runt respektive mål
        if only_if_outside_band:
            within = True
            for s in symbols:
                goal = target[s]
                # Bandbredd beräknas från målvikten (proportionell band, inte fast ±10 p.p.)
                band = goal * tolerance
                low = max(0.0, goal - band)
                high = min(100.0, goal + band)
                if not (low <= cur.get(s, 0.0) <= high):
                    within = False
                    break
            # Kolla även USD, fast där finns inget band-krav; USD får fluktuera.
            if within:
                # Behåll nuvarande, inga trades
                target = {k: round(cur.get(k, 0.0), 6) for k in all_keys}
                reason = f"Ingen rebalans: alla aktiva inom ±{int(tolerance*100)}% av mål."
            else:
                reason = (
                    f"Rebalans till lika vikt över {n_active} aktiva "
                    f"(SELL→0, USD residual)."
                )
        else:
            reason = (
                f"Rebalans till lika vikt över {n_active} aktiva "
                f"(SELL→0, USD residual)."
            )

    # Runda och säkerställ att summan blir 100
    def _round_dict(d: Dict[str, float]) -> Dict[str, float]:
        # Runda till två decimaler och korrigera sista nyckeln för att summera 100.00
        keys = list(d.keys())
        rounded = {k: round(d[k], 2) for k in keys}
        diff = round(100.0 - sum(rounded.values()), 2)
        # justera USD om finns, annars sista
        adj_key = usd_key if usd_key in rounded else keys[-1]
        rounded[adj_key] = round(rounded[adj_key] + diff, 2)
        return rounded

    target = _round_dict(target)

    # Beräkna trades i procentenheter (mål - nuvarande)
    # (positivt = köp, negativt = sälj)
    trades = {k: round(target.get(k, 0.0) - round(cur.get(k, 0.0), 2), 2) for k in all_keys}

    return RebalanceResult(target_allocations=target, trades=trades, reason=reason)


# --------- Snabba exempeltester (körs bara om filen körs direkt) ----------
if __name__ == "__main__":
    # Hjälpfunktion för snygg utskrift
    def show(res: RebalanceResult):
        print("Target:", res.target_allocations)
        print("Trades:", res.trades)
        print("Reason:", res.reason)
        print("-" * 60)

    syms = ["A", "B", "C"]

    # Ex 1: SELL, SELL, BUY -> (0,0,100)
    res = rebalance_three(
        syms,
        {"A": "SELL", "B": "SELL", "C": "BUY"},
        {"A": 30, "B": 30, "C": 40, "USD": 0},
        only_if_outside_band=True,
    )
    show(res)

    # Ex 2: SELL, BUY, HOLD -> (0,50,50)
    res = rebalance_three(
        syms,
        {"A": "SELL", "B": "BUY", "C": "HOLD"},
        {"A": 20, "B": 40, "C": 40, "USD": 0},
        only_if_outside_band=True,
    )
    show(res)

    # Ex 3: BUY, HOLD, BUY -> (33,33,33)
    res = rebalance_three(
        syms,
        {"A": "BUY", "B": "HOLD", "C": "BUY"},
        {"A": 10, "B": 20, "C": 30, "USD": 40},
        only_if_outside_band=True,
    )
    show(res)

    # Ex 4: SELL, SELL, HOLD -> (0,0,100)
    res = rebalance_three(
        syms,
        {"A": "SELL", "B": "SELL", "C": "HOLD"},
        {"A": 25, "B": 25, "C": 50, "USD": 0},
        only_if_outside_band=True,
    )
    show(res)

    # Ex 5: Portfölj (55,10,0) + HOLD,HOLD,SELL -> (50,50,0)
    res = rebalance_three(
        syms,
        {"A": "HOLD", "B": "HOLD", "C": "SELL"},
        {"A": 55, "B": 10, "C": 0, "USD": 35},
        only_if_outside_band=True,
    )
    show(res)
