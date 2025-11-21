import os
import requests
import time, logging, json
import hashlib
import hmac

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

def _mask_val(v):
    if not v:
        return "<none>"
    try:
        s = str(v)
        # mask credentials in urls like user:pass@
        if "@" in s and ":" in s:
            parts = s.split("@")
            creds = parts[0]
            rest = "@".join(parts[1:])
            return s.replace(creds + "@", "***:***@")
        return s
    except Exception:
        return "<mask_error>"

def _probe_proxy(proxies):
    """
    Additional runtime checks to log:
    - public IP without proxy
    - public IP with proxy (if proxies provided)
    - Binance serverTime via the same proxies
    - optional lightweight signed /api/v3/account call (if keys present)
    """
    log = logging.getLogger("binance_dbg")

    # public IP without proxy
    try:
        r = requests.get("https://api.ipify.org?format=json", timeout=8)
        r.raise_for_status()
        ip_no_proxy = r.json().get("ip")
    except Exception as e:
        ip_no_proxy = f"<failed: {e}>"
    log.info("DBG: public IP without proxy=%s", ip_no_proxy)

    # public IP with proxy
    if proxies:
        try:
            r = requests.get("https://api.ipify.org?format=json", proxies=proxies, timeout=8)
            r.raise_for_status()
            ip_with_proxy = r.json().get("ip")
        except Exception as e:
            ip_with_proxy = f"<failed: {e}>"
        log.info("DBG: public IP with proxy=%s", ip_with_proxy)
    else:
        log.info("DBG: no proxy configured, skipping ip-with-proxy check")

    # Binance server time via proxies (if configured) else direct
    try:
        if proxies:
            r = requests.get("https://api.binance.com/api/v3/time", proxies=proxies, timeout=8)
        else:
            r = requests.get("https://api.binance.com/api/v3/time", timeout=8)
        r.raise_for_status()
        server_time = r.json().get("serverTime")
        local_ms = int(time.time() * 1000)
        log.info("DBG: binance serverTime=%s local_ms=%s delta_ms=%s", server_time, local_ms, local_ms - (server_time or 0))
    except Exception as e:
        log.exception("DBG: binance time check failed: %s", e)

    # Optional: do a lightweight signed account call if keys are available
    api_key = os.environ.get("CCXT_API_KEY") or os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("CCXT_API_SECRET") or os.environ.get("BINANCE_API_SECRET")
    if api_key and api_secret:
        try:
            ts = str(int(time.time() * 1000))
            qs = f"timestamp={ts}"
            sig = hmac.new(api_secret.encode("utf-8"), qs.encode("utf-8"), hashlib.sha256).hexdigest()
            headers = {"X-MBX-APIKEY": api_key}
            url = f"https://api.binance.com/api/v3/account?{qs}&signature={sig}"
            r = requests.get(url, headers=headers, proxies=proxies, timeout=10)
            # Avoid logging secrets; log masked key and status/body
            log.info("DBG: account call using masked key=%s status=%s", (api_key[:6] + "..." + api_key[-4:]), r.status_code)
            try:
                log.debug("DBG: account body: %s", r.text[:2000])
            except Exception:
                pass
        except Exception as e:
            log.exception("DBG: account test failed: %s", e)
    else:
        log.info("DBG: no api key/secret available in env, skipping account test")

def get_proxy():
    raw = os.environ.get("FIXIE_SOCKS_HOST")

    proxies = None
    if raw:
        # Låt proxyn göra DNS (socks5h) så går allting via Fixie
        fixie_url = f"socks5h://{raw}"
        proxies = {
            "http": fixie_url,
            "https": fixie_url,
        }

    # Log and probe the proxy before returning
    logging.getLogger("binance_dbg").info("get_proxy() built proxies: %s", _mask_val(proxies))
    # dbg_binance_context(proxies)
    # _probe_proxy(proxies)
    return proxies

if __name__ == "__main__":
    proxies = getProxy()
    print("Proxies:", proxies)

    r = requests.get("https://api.ipify.org?format=json", proxies=proxies, timeout=10)
    print("IP enligt ipify:", r.text)
