#!/usr/bin/env bash
# Debug-script för Binance signed request från Heroku-dyno
# Usage:
# 1) If your API key env var is literally named A and signature env var is literally named B:
#      ./binance_debug.sh A B
# 2) Or use full form:
#      ./binance_debug.sh --key-env BINANCE_API_KEY --sig-env BINANCE_SIGNATURE
# 3) To compute signature from secret instead of using precomputed sig:
#      ./binance_debug.sh --key-env CCXT_API_KEY --secret-env CCXT_API_SECRET --compute-sig
#
# NOTE: This script prints debug info (status, headers, body). Do NOT paste logs containing secrets publicly.

set -euo pipefail

# Defaults (change if you want)
PROXY_URL="${PROXY_URL:-socks5h://fixie:J0duPrq3gn6EEdi@bici.usefixie.com:1080}"
BINANCE_BASE="https://api.binance.com"

# Parse args (simple)
KEY_ENV=""
SIG_ENV=""
SECRET_ENV=""
COMPUTE_SIG=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --key-env) KEY_ENV="$2"; shift 2 ;;
    --sig-env) SIG_ENV="$2"; shift 2 ;;
    --secret-env) SECRET_ENV="$2"; shift 2 ;;
    --compute-sig) COMPUTE_SIG=1; shift ;;
    -h|--help) echo "Usage: $0 [--key-env NAME] [--sig-env NAME] [--secret-env NAME] [--compute-sig]"; exit 0;;
    *) 
      # positional short form: first two args -> KEY_ENV SIG_ENV
      if [[ -z "$KEY_ENV" ]]; then KEY_ENV="$1"; else SIG_ENV="$1"; fi
      shift ;;
  esac
done

if [[ -z "$KEY_ENV" ]]; then
  echo "Error: missing API key env name. Provide --key-env or first positional arg."
  exit 2
fi

# resolve env var values by name
get_env_val() {
  local name="$1"
  # Indirect expansion
  printf '%s' "${!name:-}"
}

API_KEY="$(get_env_val "$KEY_ENV")"
if [[ -z "$API_KEY" ]]; then
  echo "Error: API key empty for env name: $KEY_ENV"
  exit 2
fi

if [[ $COMPUTE_SIG -eq 1 ]]; then
  if [[ -z "$SECRET_ENV" ]]; then
    echo "Error: --compute-sig requires --secret-env to be set with your API secret env name."
    exit 2
  fi
  API_SECRET="$(get_env_val "$SECRET_ENV")"
  if [[ -z "$API_SECRET" ]]; then
    echo "Error: API secret empty for env name: $SECRET_ENV"
    exit 2
  fi
fi

if [[ -z "$SIG_ENV" && $COMPUTE_SIG -eq 0 ]]; then
  echo "Error: no signature source given. Provide --sig-env or use --compute-sig with --secret-env."
  exit 2
fi

SIG_VAL=""
if [[ $COMPUTE_SIG -eq 1 ]]; then
  :
else
  SIG_VAL="$(get_env_val "$SIG_ENV")"
  if [[ -z "$SIG_VAL" ]]; then
    echo "Error: signature empty for env name: $SIG_ENV"
    exit 2
  fi
fi

# Helper: timestamp in ms
TS_MS() {
  date +%s%3N
}

mask_key() {
  local k="$1"
  if [[ -z "$k" ]]; then echo ""; return; fi
  echo "${k:0:6}...${k: -4}"
}

echo "Using API key from env: $KEY_ENV (masked: $(mask_key "$API_KEY"))"
if [[ $COMPUTE_SIG -eq 1 ]]; then
  echo "Signature will be computed from secret in env: $SECRET_ENV"
else
  echo "Using precomputed signature from env: $SIG_ENV (not printing)"
fi
echo

# Public IP check (via proxy and without proxy)
echo "Fetching public IP WITHOUT proxy..."
curl -sS --max-time 10 https://api.ipify.org?format=json || echo "ipify failed (no proxy)"
echo
echo "Fetching public IP WITH proxy (if PROXY_URL set)..."
if [[ -n "$PROXY_URL" ]]; then
  # Using curl -x for socks5
  curl -sS --max-time 10 -x "$PROXY_URL" https://api.ipify.org?format=json || echo "ipify via proxy failed"
else
  echo "(PROXY_URL empty, skipping)"
fi
echo

# Get Binance server time (via proxy)
echo "Getting Binance server time (via proxy if PROXY_URL set)..."
if [[ -n "$PROXY_URL" ]]; then
  BIN_TIME_JSON="$(curl -sS -x "$PROXY_URL" "$BINANCE_BASE/api/v3/time" || true)"
else
  BIN_TIME_JSON="$(curl -sS "$BINANCE_BASE/api/v3/time" || true)"
fi
echo "Binance /api/v3/time => $BIN_TIME_JSON"
echo

# Build request for /api/v3/account
TS="$(TS_MS)"
QUERY="timestamp=${TS}"

if [[ $COMPUTE_SIG -eq 1 ]]; then
  # compute HMAC SHA256 hex via openssl
  SIG="$(printf "%s" "$QUERY" | openssl dgst -sha256 -hmac "$API_SECRET" | sed 's/^.* //')"
else
  SIG="$SIG_VAL"
fi

URL="${BINANCE_BASE}/api/v3/account?${QUERY}&signature=${SIG}"

echo "Final request URL (signature included, not printing signature value)."
echo "Timestamp used: $TS"
echo "Calling Binance /api/v3/account (this is a signed endpoint)."

# Curl with detailed output
if [[ -n "$PROXY_URL" ]]; then
  echo "Using proxy: $PROXY_URL"
  curl -v -x "$PROXY_URL" \
    -H "X-MBX-APIKEY: ${API_KEY}" \
    "$URL" || echo "curl exit with nonzero status"
else
  curl -v \
    -H "X-MBX-APIKEY: ${API_KEY}" \
    "$URL" || echo "curl exit with nonzero status"
fi

echo
echo "Done. If you received 4xx with msg 'Invalid API-key, IP, or permissions for action.', check:"
echo "- That the public IP shown above (with proxy) is whitelisted in Binance"
echo "- That the key has the correct permissions and isn't restricted"
echo "- That timestamp delta isn't too large (compare Binance serverTime above vs local)"