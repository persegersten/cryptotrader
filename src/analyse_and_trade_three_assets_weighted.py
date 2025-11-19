#!/usr/bin/env python3
"""
Python replacement for analyse_and_trade_three_assets_weighted.sh

This runner imports and calls functions directly (no subprocess).
It expects the following programmatic entrypoints to exist:
 - schedule_gate.run(...) or schedule_gate.main()
 - download_portfolio.run() or download_portfolio.main()
 - download_binance_ohlcv.run(symbol, folder) or download_binance_ohlcv.main(...)
 - the TA agent function: src.ta_signal_agent_live_three_assets.run_agent(...)
If some of those modules lack a run_* function, add a small wrapper in each module mirroring the pattern used in the TA agent above.
"""

from pathlib import Path
import shutil
import logging
from typing import Tuple

# Import the agent function directly
from ta_signal_agent_live_three_assets import run_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
HISTORY_DIR = ROOT / "history"
DATA_FOLDERS = {
    "bnb": ROOT / "bnb_data",
    "ethereum": ROOT / "ethereum_data",
    "solana": ROOT / "solana_data",
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def rotate_history():
    ensure_dir(HISTORY_DIR)
    for folder in DATA_FOLDERS.values():
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.is_file():
                shutil.move(str(f), str(HISTORY_DIR / f.name))

def find_latest_file(folder: Path) -> Path | None:
    files = [p for p in folder.glob("*") if p.is_file()]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]

def locate_input_files() -> Tuple[str, str, str]:
    file_bnb = find_latest_file(DATA_FOLDERS["bnb"])
    file_eth = find_latest_file(DATA_FOLDERS["ethereum"])
    file_sol = find_latest_file(DATA_FOLDERS["solana"])
    if not file_bnb or not file_eth or not file_sol:
        missing = [str(p) for k,p in DATA_FOLDERS.items() if not find_latest_file(p)]
        raise RuntimeError(f"Missing files in: {missing}")
    return str(file_bnb), str(file_eth), str(file_sol)

def main(dry_run: bool = True):
    # Optional: call schedule gate programmatically if schedule_gate exposes a run() function
    try:
        import schedule_gate as schedule_gate
        if not schedule_gate.run(grace_minutes=5, at_hours=[0,4,7,8,16,20,21], time_zone="Europe/Stockholm"):
            return
    except Exception:
        raise

    # move old CSVs to history
    rotate_history()

    # download OHLCV using programmatic wrappers
    try:
        import download_binance_ohlcv as dl
        # the module should provide a run(symbol, data_folder) or similar; try common names
        for sym, folder in (("BNBUSDT", DATA_FOLDERS["bnb"]), ("ETHUSDT", DATA_FOLDERS["ethereum"]), ("SOLUSDT", DATA_FOLDERS["solana"])):
            ensure_dir(folder)
            dl.run(symbol=sym, data_folder=str(folder))
    except Exception:
        raise

    # download portfolio
    try:
        import download_portfolio as download_portfolio
        download_portfolio.run("portfolio.json", "USDC")
    except Exception:
        raise

    # locate CSVs and call the agent directly
    csvA, csvB, csvC = locate_input_files()
    snapshot = run_agent(
        csvA=csvA,
        csvB=csvB,
        csvC=csvC,
        symbols="BNB/USDC,ETH/USDC,SOL/USDC",
        exchange="binance",
        dry_run=dry_run,
    )
    log.info("Agent finished. Snapshot keys: %s", ", ".join(snapshot.keys()))

if __name__ == "__main__":
    main(dry_run=True)