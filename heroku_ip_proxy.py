import os


def getProxy():
    # Hämta Fixie Socks URL från Heroku config
    fixie_url = os.environ.get("FIXIE_SOCKS_HOST")

    # Ställ in proxies för requests
    # Sätt upp proxies bara om Fixie är satt
    proxies = None
    if fixie_url:
        proxies = {
            "http": fixie_url,
            "https": fixie_url,
        }

    return proxies