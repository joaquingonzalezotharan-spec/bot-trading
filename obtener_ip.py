"""
Script temporal para verificar la IP pública saliente desde el Background Worker (Render).
Consulta https://ipify.org y la imprime en logs con flush=True.
"""

from urllib.request import urlopen


def main() -> None:
    url = "https://ipify.org"
    with urlopen(url, timeout=20) as resp:
        ip = resp.read().decode("utf-8").strip()
    print(f"[IPIFY] IP pública detectada: {ip}", flush=True)


if __name__ == "__main__":
    main()

