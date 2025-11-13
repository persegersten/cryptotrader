import os

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

    return proxies
