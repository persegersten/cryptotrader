import os
import requests

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

        print("Proxies:", proxies)
        r = requests.get("https://api.ipify.org?format=json", proxies=proxies, timeout=10)
        print("Utgående IP via Fixie:", r.text)

    return proxies
