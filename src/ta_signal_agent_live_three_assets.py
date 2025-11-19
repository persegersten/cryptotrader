# --- Add the following function to src/ta_signal_agent_live_three_assets.py (place it near the top-level main) ---

def run_agent(
    csvA: str,
    csvB: str,
    csvC: str,
    symbols: str,
    exchange: str = "binance",
    api_key: str | None = None,
    api_secret: str | None = None,
    sandbox: bool = False,
    fee_bps: float = 10.0,
    min_trade: float = 20.0,
    dry_run: bool = True,
    log: str = "trades_log.csv",
    portfolio: str = "portfolio.json",
) -> dict:
    """
    Programmatic entrypoint for the TA agent.
    Parameters mirror the --cli args from the original script.
    Returns a portfolio_snapshot dict (same structure that the CLI writes to portfolio.json).
    """

    # Use the same helpers already defined in the module: require_env, CCXTBroker, decide_signal, rebalance_three, etc.
    load_secrets_if_missing("secrets.json")

    # Validate symbols and build internal vars (copied from the original main logic)
    csv_map = {"A": csvA, "B": csvB, "C": csvC}
    syms_list = [s.strip() for s in symbols.split(",")]
    if len(syms_list) != 3:
        raise ValueError("You must provide exactly three comma-separated symbols, e.g. 'BTC/USDT,ETH/USDT,SOL/USDT'")

    # Ensure same quote
    quotes = [s.split("/")[1] for s in syms_list]
    if not (quotes[0] == quotes[1] == quotes[2]):
        raise ValueError("All three symbol pairs must share the same quote currency (e.g. USDT)")

    quote_ccy = quotes[0]
    base_ccys = [s.split("/")[0] for s in syms_list]

    # load csvs and compute signals
    def load_and_signal(csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
        sig, score, reasons = decide_signal(df)
        ts = df.iloc[-1][df.columns[0]]
        last_close = float(df.iloc[-1]["close"])
        return df, sig, score, reasons, ts, last_close

    dfs: Dict[str, pd.DataFrame] = {}
    signals: Dict[str, str] = {}
    meta: Dict[str, dict] = {}
    for name, sym in zip(("A", "B", "C"), syms_list):
        df, sig, score, reasons, ts, last_close = load_and_signal(csv_map[name])
        dfs[sym] = df
        signals[sym.split("/")[0]] = sig
        meta[sym] = {"score": score, "reasons": reasons, "ts": ts, "last_close": last_close}

    # Broker / portfolio: require_env guards that API keys exist in environment
    require_env()
    broker = CCXTBroker(exchange, api_key, api_secret, sandbox)

    balances = broker.fetch_balances()

    # Fetch prices (fall back to last close from CSV on error)
    prices: Dict[str, float] = {}
    for s in syms_list:
        try:
            prices[s] = broker.fetch_price(s)
        except Exception:
            prices[s] = meta[s]["last_close"]

    # current allocations (percent)
    current_alloc = get_current_allocations_pct_three(balances, tuple(syms_list), prices, quote_ccy)

    # build scores_map and call rebalance_three
    scores_map = {}
    for i, full_sym in enumerate(syms_list):
        base = base_ccys[i]
        scores_map[base] = meta[full_sym]["score"]

    rb: RebalanceResult = rebalance_three(
        [b for b in base_ccys],
        {b: signals[b] for b in base_ccys},
        current_alloc,
        scores=scores_map,
    )

    # compute equity in quote
    equity = 0.0
    for i, sym in enumerate(syms_list):
        base = base_ccys[i]
        if isinstance(balances.get("free"), dict):
            qty = float(balances["free"].get(base, 0.0) + balances.get("total", {}).get(base, 0.0) - balances.get("used", {}).get(base, 0.0))
            if qty < 0:
                qty = float(balances["free"].get(base, 0.0))
        else:
            qty = float(balances.get(base, 0.0))
        equity += qty * prices[sym]
    quote_bal = float(balances["free"].get(quote_ccy, 0.0)) if isinstance(balances.get("free"), dict) else float(balances.get(quote_ccy, 0.0))
    equity += quote_bal

    # translate trades -> planned orders (same logic as original)
    planned_orders: List[Tuple[str, str, float]] = []
    fee_mult = (1 - fee_bps / 10000.0)

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
        if abs(delta_pp) < 1e-6:
            continue
        usd_delta = (delta_pp / 100.0) * equity

        if usd_delta > 0:
            usd_to_spend = usd_delta * fee_mult
            if usd_to_spend >= min_trade and get_free_quote() >= usd_to_spend:
                planned_orders.append(("BUY", sym_pair, usd_to_spend))
        else:
            px = prices[sym_pair]
            qty = (abs(usd_delta) / max(px, 1e-12)) * fee_mult
            qty = min(qty, get_free_base(base))
            if qty * px >= min_trade and qty > 0:
                planned_orders.append(("SELL", sym_pair, qty))

    # execute planned orders (dry_run controls whether to place actual orders)
    executions = []
    if planned_orders:
        for side, sym_pair, amount in planned_orders:
            if dry_run:
                if side == "BUY":
                    executions.append({"side": side, "symbol": sym_pair, "amount": amount, "order_id": None})
                else:
                    px = prices[sym_pair]
                    executions.append({"side": side, "symbol": sym_pair, "amount": amount, "order_id": None})
            else:
                if side == "BUY":
                    order = broker.market_buy_quote(sym_pair, quote_amount=amount)
                else:
                    order = broker.market_sell_base(sym_pair, base_qty=amount)
                executions.append({"side": side, "symbol": sym_pair, "amount": amount, "order_id": order.get("id") if isinstance(order, dict) else str(order)})

    # build portfolio snapshot (same structure as CLI)
    portfolio_snapshot = {
        "quote": quote_ccy,
        "prices": {s: prices[s] for s in syms_list},
        "executions": executions,
        "rebalance_reason": rb.reason,
        "signals": {b: signals[b] for b in base_ccys},
        "targets_pct": rb.target_allocations,
        "trades_pp": rb.trades,
        "current_alloc_pct": current_alloc,
    }

    # write outputs as the CLI did
    Path(log).write_text("") if False else None  # noop to avoid lint errors; keep original write logic below
    # write CSV log row + portfolio json (replicate the CLI file-writing behavior)
    now = datetime.now(ZoneInfo("Europe/Stockholm")).strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "time": now,
        "symbols": symbols,
        "signals": json.dumps({b: signals[b] for b in base_ccys}),
        "prices": json.dumps({s: prices[s] for s in syms_list}),
        "current_alloc": json.dumps(current_alloc),
        "targets": json.dumps(rb.target_allocations),
        "trades_pp": json.dumps(rb.trades),
        "reason": rb.reason,
        "dry_run": dry_run,
    }
    log_file = Path(log)
    exists = log_file.exists()
    with log_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    Path(portfolio).write_text(json.dumps(portfolio_snapshot, indent=2, ensure_ascii=False))
    return portfolio_snapshot


# --- Change the existing main() at the end of the file to call run_agent(...) so CLI continues to work:
# Replace the current main() body (argparse + logic) with a thin wrapper:
def main():
    ap = argparse.ArgumentParser(description="Tre-tillgångars TA-agent med CCXT och rebalansering")
    ap.add_argument("--csvA", required=True, help="CSV med OHLCV för symbol A")
    ap.add_argument("--csvB", required=True, help="CSV med OHLCV för symbol B")
    ap.add_argument("--csvC", required=True, help="CSV med OHLCV för symbol C")
    ap.add_argument("--symbols", required=True, help="Komma-separerade tre symbolpar, t.ex. BTC/USDT,ETH/USDT,SOL/USDT")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--api-secret", default=None)
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--fee-bps", type=float, default=10.0, help="avgifter i bps")
    ap.add_argument("--min-trade", type=float, default=20.0, help="minsta order i USD (quote). Default 20 USDC per nya regler")
    ap.add_argument("--dry-run", action="store_true", help="Simulera utan riktiga ordrar (hämtar ändå verklig portfolio och priser)")
    ap.add_argument("--log", default="trades_log.csv")
    ap.add_argument("--portfolio", default="portfolio.json", help="Filen att spara täckt portfölj-snapshot till")
    args = ap.parse_args()

    snapshot = run_agent(
        csvA=args.csvA,
        csvB=args.csvB,
        csvC=args.csvC,
        symbols=args.symbols,
        exchange=args.exchange,
        api_key=args.api_key,
        api_secret=args.api_secret,
        sandbox=args.sandbox,
        fee_bps=args.fee_bps,
        min_trade=args.min_trade,
        dry_run=args.dry_run,
        log=args.log,
        portfolio=args.portfolio,
    )
    # main prints/writes as run_agent already wrote files
    if args.dry_run:
        print("Dry-run: used real balances/prices but did not place live orders.")
    else:
        print("Production: orders placed where planned.")