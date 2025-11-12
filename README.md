```markdown
# CryptoHunk

Lightweight scripts and tools to analyse and (optionally) trade a small multi-asset crypto portfolio using downloaded OHLCV data and a TA-based signal agent.

This README describes what the repository does, how to run it (dry-run vs live), and what the included shell scripts do. The README was written using the two primary runner scripts as starting points:

- `analyse_and_trade_three_assets_weighted.sh`
- `dryrun_analyse_and_trade.sh`

## Overview / How it works

High level steps performed by the runner scripts:

1. Optionally gate execution to scheduled times using `src/schedule_gate.py`.
2. Download the portfolio (`src/download_portfolio.py`).
3. Move previous downloaded data into `./history/`.
4. Download fresh OHLCV files for each asset with `src/download_binance_ohlcv.py`.
5. Locate the newest CSV files in each asset folder (e.g. `bnb_data`, `ethereum_data`, `solana_data`).
6. Run the TA signal agent `src/ta_signal_agent_live_three_assets.py` using the three CSVs and configured symbols.
   - The dry-run script passes `--dry-run` so no live trades are executed.

The "three asset" flow used in the scripts is: BNB, ETH, SOL (symbols used when invoking the agent are `BNB/USDC,ETH/USDC,SOL/USDC`).

## Quickstart

Prerequisites
- Python 3.8+ (virtualenv recommended)
- pip
- (Optional for live trading) Binance API key & secret configured for any trading scripts that actually place orders.

Setup
1. Clone the repository:
   git clone https://github.com/persegersten/cryptohunk.git
   cd cryptohunk

2. Create and activate a virtual environment, install dependencies (if a requirements file exists):
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt   # if present

3. Ensure the following folders exist (scripts expect them):
   - ./bnb_data
   - ./ethereum_data
   - ./solana_data
   - ./history

4. (Optional) Set environment variables for any exchange APIs you will use. For Binance examples:
   export BINANCE_API_KEY="your_key"
   export BINANCE_SECRET="your_secret"

Usage — dry run (recommended for testing)
- Make executable (if not already):
  chmod +x dryrun_analyse_and_trade.sh

- Run:
  ./dryrun_analyse_and_trade.sh

This runs the full data download and analysis pipeline but passes `--dry-run` to the TA agent so it will not place live trades.

Usage — live (weighted three-asset run)
- Make executable:
  chmod +x analyse_and_trade_three_assets_weighted.sh

- Run:
  ./analyse_and_trade_three_assets_weighted.sh

This runs the same pipeline but without `--dry-run`. Use only if you have configured the trading credentials and are ready to place orders.

Notes about scheduling
- The live script uses `src/schedule_gate.py` with a list of execution times (configured in the script).
- For local testing you may comment out the schedule gate or adjust the times / timezone in the script.

Key scripts and what they do
- `src/schedule_gate.py` — gate that prevents the script from running unless it is an allowed time window.
- `src/download_portfolio.py` — download or refresh portfolio metadata (generates `portfolio.json`).
- `src/download_binance_ohlcv.py` — download OHLCV CSVs from Binance for a given symbol into a specified folder.
- `src/ta_signal_agent_live_three_assets.py` — the TA signal agent that ingests three CSVs and decides trades for the configured symbols.

Directory / file expectations
- `bnb_data/`, `ethereum_data/`, `solana_data/` — folders where fresh downloads are stored.
- `history/` — older CSVs are moved here by the runner scripts.
- `portfolio.json` — generated/updated by `src/download_portfolio.py`.

Important behavior details
- The runner scripts look for the first file in each data folder (via `find ... | head -n 1`) and set `IN_DATA_BNB`, `IN_DATA_ETHEREUM`, `IN_DATA_SOLANA` environment variables to those file paths before invoking the TA agent.
- The dry-run script is identical except it adds `--dry-run` to the TA agent invocation, preventing live trades.

Configuration hints
- To change assets or symbols, update the `--symbols` argument passed to `src/ta_signal_agent_live_three_assets.py` in the shell scripts.
- To add or remove assets, add/remove corresponding data download commands and folder handling code in the runner scripts.
- If you want to run this from cron instead of `schedule_gate.py`, comment out the gate lines and create a cron job to run the desired script at your chosen times.

Troubleshooting
- "Ingen fil hittades i ./bnb_data" — the download step failed or produced no files. Check logs/output of `src/download_binance_ohlcv.py`.
- Permission issues running scripts — ensure files are executable (chmod +x).
- Missing API credentials — ensure environment variables for APIs are present if the download or trading code needs them.

Security and safety
- Always use `dryrun_analyse_and_trade.sh` to validate the pipeline before enabling live trading.
- Keep API keys and secrets out of the repo. Use environment variables or a secrets manager.
- Test on small sizes or a paper account before increasing risk.

Contributing
- Please open issues or pull requests if you want to add new supported assets, improve signal logic, or enhance safety checks.
- When adding assets, update the runner scripts to download the new asset OHLCV files, move files to history, and pass the new CSVs to the TA agent.

License
- No license file is included in the repo by default; add a LICENSE file if you intend to make this open source.

If you'd like, I can:
- Create this README.md in the repository,
- Add example environment variable and cron/systemd unit snippets,
- Or convert the runner scripts into a single configurable script that accepts a list of assets and schedules.
```