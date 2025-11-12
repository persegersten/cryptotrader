# portfolio_rebalancer.py
from dataclasses import dataclass
from typing import Dict, List

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
    scores: Dict[str, int],
    usd_key: str = "USD",
) -> RebalanceResult:
    """
    Rebalansering enligt de nya enkla reglerna för få-tillgångs-setup:
      1) SELL => alltid sälj 100% av den tillgången (mål 0%).
      2) HOLD => behåll nuvarande (ingen måländring).
      3) BUY => endast köp EN tillgång: välj BUY med högst score och använd
         all tillgänglig USD för att öka den tillgångens mål (dvs flytta USD -> asset).
      4) Minsta köp-logik (t.ex. 20 USD) hanteras i exekveringssteget (ta-agenten)
         eftersom rebalance arbetar i procentenheter.

    Parametrar:
      symbols: exakt 3 bas-ccys, t.ex. ["BTC","ETH","SOL"]
      signals: mapping bas-ccy->"BUY"/"SELL"/"HOLD"
      current_allocations: mapping med 3 bas-ccys + ev. 'USD' (procent 0..100).
                         Summan behöver inte vara exakt 100 — normaliseras internt.
      scores: mapping bas-ccy->int score (används för att välja vilken BUY som är "starkast")
      usd_key: namnet på quote/USDC i alloc-map: default "USD"

    Returnerar:
      RebalanceResult med målvikter, föreslagna trades (i procentenheter) och en kort reason.

    Logik-exempel:
      - Om alla tre SELL -> (0,0,0,100) (allt till USD)
      - Om A=BUY, B=HOLD, C=HOLD och USD=40% -> flytta USD till A (A += 40%, USD=0)
      - Om flera BUY -> välj den med högst score (scores parameter); vid lika score välj
        den som ligger först i symbols (deterministiskt).
      - HOLD betyder att målvikt för den tillgången sätts till dess nuvarande andel (ingen förändring).
    """
    if len(symbols) != 3:
        raise ValueError("symbols måste ha exakt tre tickers.")
    for s in symbols:
        if s not in signals:
            raise ValueError(f"Saknar signal för {s}")

    # Normalisera nuvarande vikter över (3 krypto + ev USD)
    all_keys = list((symbols) + [usd_key])
    cur = {k: float(current_allocations.get(k, 0.0)) for k in all_keys}
    cur = _normalize_allocs(cur)

    # Initiera mål: SELL -> 0.0, HOLD -> behåll nuvarande, BUY -> lämna till senare
    target: Dict[str, float] = {}
    buy_candidates: List[str] = []
    for s in symbols:
        sig = signals.get(s, "HOLD").upper()
        if sig == "SELL":
            target[s] = 0.0
        elif sig == "HOLD":
            target[s] = cur.get(s, 0.0)
        else:  # BUY (eller annat) -> behandlas som kandidat för köp
            buy_candidates.append(s)
            # sätt preliminärt till nuvarande — så att SUM räknas korrekt innan USD flytt
            target[s] = cur.get(s, 0.0)

    usd_pct = cur.get(usd_key, 0.0)

    # Hantera fall med inga BUYs / ett eller flera BUYs
    if len(buy_candidates) == 0:
        residual = max(0.0, 100.0 - sum(target[s] for s in symbols))
        target[usd_key] = residual
        reason = "Ingen BUY: SELL→0, HOLD behålls. USD är residual."
    else:
        # scores är obligatorisk; använd en snabb index-lookup för tie-breaker (O(1))
        index_map = {s: i for i, s in enumerate(symbols)}
        best = max(buy_candidates, key=lambda x: (scores.get(x, 0), -index_map[x]))

        # Flytta hela USD-posten till den valda BUY-tillgången
        target[best] = target.get(best, 0.0) + usd_pct
        target[usd_key] = 0.0
        reason = f"BUY-policy: flyttar USD -> {best} (starkaste BUY). SELL→0, HOLD behålls."

    # Runda och säkerställ att summan blir 100
    def _round_dict(d: Dict[str, float]) -> Dict[str, float]:
        keys = list(d.keys())
        rounded = {k: round(d.get(k, 0.0), 2) for k in keys}
        diff = round(100.0 - sum(rounded.values()), 2)
        adj_key = usd_key if usd_key in rounded else keys[-1]
        rounded[adj_key] = round(rounded.get(adj_key, 0.0) + diff, 2)
        return rounded

    target = _round_dict(target)

    # Beräkna trades i procentenheter (mål - nuvarande)
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
        scores={"A": -2, "B": -1, "C": 3},
    )
    show(res)

    # Ex 2: SELL, BUY, HOLD -> behåll HOLD, flytta USD till BUY
    res = rebalance_three(
        syms,
        {"A": "SELL", "B": "BUY", "C": "HOLD"},
        {"A": 20, "B": 40, "C": 40, "USD": 0},
        scores={"B": 2},
    )
    show(res)

    # Ex 3: BUY, HOLD, BUY -> välj starkaste BUY och flytta USD dit
    res = rebalance_three(
        syms,
        {"A": "BUY", "B": "HOLD", "C": "BUY"},
        {"A": 10, "B": 20, "C": 30, "USD": 40},
        scores={"A": 1, "C": 3},
    )
    show(res)

    # Ex 4: SELL, SELL, HOLD -> (0,0,hold,100)
    res = rebalance_three(
        syms,
        {"A": "SELL", "B": "SELL", "C": "HOLD"},
        {"A": 25, "B": 25, "C": 50, "USD": 0},
        scores={}
    )
    show(res)

    # Ex 5: Portfölj (55,10,0) + HOLD,HOLD,SELL -> (55,10,0,35) (USD residual)
    res = rebalance_three(
        syms,
        {"A": "HOLD", "B": "HOLD", "C": "SELL"},
        {"A": 55, "B": 10, "C": 0, "USD": 35},
        scores={"A": 0, "B": 0, "C": 0},
    )
    show(res)