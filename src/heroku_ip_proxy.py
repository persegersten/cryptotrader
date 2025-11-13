import os
import requests
import time, logging, json

def dbg_binance_context(proxies=None, api_key_name="CCXT_API_KEY"):
    log = logging.getLogger("binance_dbg")
    key = os.environ.get(api_key_name, "")
    def mask(k): return (k[:6] + "..." + k[-4:]) if k else ""
    # public IP
    try:
        ip = requests.get("https://api.ipify.org?format=json", proxies=proxies, timeout=8).json().get("ip")
    except Exception as e:
        ip = f"<ipify failed: {e}>"
    # binance server time
    try:
        bt = requests.get("https://api.binance.com/api/v3/time", proxies=proxies, timeout=8).json().get("serverTime")
    except Exception as e:
        bt = f"<time failed: {e}>"
    log.info("BINANCE DBG: key_env=%s key_mask=%s proxy=%s public_ip=%s binance_serverTime=%s local_ms=%d",
             api_key_name, mask(key), os.environ.get("PROXY_URL","<none>"), ip, bt, int(time.time()*1000))
    return dict(public_ip=ip, server_time=bt)

def getProxy():
    raw = os.environ.get("FIXIE_SOCKS_HOST") 

    proxies = None
    if raw:
        # Låt proxyn göra DNS (socks5h) så går allting via Fixie
        fixie_url = f"socks5h://{raw}"
        proxies = {
            "http": fixie_url,
            "https": fixie_url,
        }

    dbg_binance_context(proxies)
    return proxies
