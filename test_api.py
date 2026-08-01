"""
test_api.py

Script de diagnóstico aislado para verificar conectividad y permisos con Binance Futuros (USDT-M).

Qué hace:
- Lee variables de entorno: BINANCE_API_KEY, BINANCE_API_SECRET, PROXY_IP, PROXY_USER, PROXY_PASSWORD
- Inicializa python-binance Client con requests_params(proxies=...)
- Imprime URLs base internas del cliente (si existen)
- Llama una acción simple: client.futures_account_balance()
- Imprime error completo con traceback si falla

Nota de seguridad:
- NO imprime la API key (ni aunque sea parcial).
- Solo imprime metadatos (presencia, longitud, espacios).
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Optional
from urllib.parse import quote


def _env_debug(name: str) -> None:
    val = os.environ.get(name)
    if not val:
        print(f"[ENV] {name}: MISSING", flush=True)
        return
    # Metadatos (sin exponer el secreto)
    print(
        f"[ENV] {name}: PRESENT | len={len(val)} | has_surrounding_whitespace={val != val.strip()}",
        flush=True,
    )


def build_client() -> Any:
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    proxy_ip = os.environ.get("PROXY_IP")
    proxy_user = os.environ.get("PROXY_USER")
    proxy_password = os.environ.get("PROXY_PASSWORD")

    _env_debug("BINANCE_API_KEY")
    _env_debug("BINANCE_API_SECRET")
    _env_debug("PROXY_IP")
    _env_debug("PROXY_USER")
    _env_debug("PROXY_PASSWORD")

    if not api_key or not api_secret:
        raise RuntimeError("Faltan BINANCE_API_KEY y/o BINANCE_API_SECRET en variables de entorno.")

    from binance import Client  # local import para evitar errores si no hay dependencias

    requests_params: Optional[dict] = None
    if proxy_ip and proxy_user and proxy_password:
        proxy_user_enc = quote(proxy_user, safe="")
        proxy_pass_enc = quote(proxy_password, safe="")
        proxy_url = f"http://{proxy_user_enc}:{proxy_pass_enc}@{proxy_ip}:50100"
        proxies = {"http": proxy_url, "https": proxy_url}
        requests_params = {"proxies": proxies}
        # No imprimimos user/pass del proxy en logs.
        print(f"[PROXY] proxy configurado: {proxy_ip}:50100", flush=True)

    client = Client(api_key=api_key, api_secret=api_secret, requests_params=requests_params)

    # Imprimir URLs base internas del cliente (si la librería las expone con esos nombres)
    candidates = [
        "API_URL",
        "API_PUBLIC_URL",
        "API_PRIVATE_URL",
        "FUTURES_URL",
        "FUTURES_API_URL",
        "BASE_URL",
    ]
    for attr in candidates:
        if hasattr(client, attr):
            print(f"[CLIENT] {attr} = {getattr(client, attr)}", flush=True)

    # Algunos clientes guardan un "base" en propiedades privadas; lo intentamos sin romper.
    for attr in ["_base_url", "_api_url"]:
        try:
            if hasattr(client, attr):
                print(f"[CLIENT] {attr} = {getattr(client, attr)}", flush=True)
        except Exception:
            pass

    return client


def main() -> None:
    try:
        client = build_client()
        print("[TEST] Llamando a futures_account_balance() ...", flush=True)
        result = client.futures_account_balance()
        print("[TEST] OK. Respuesta (primeros 2 elementos):", flush=True)
        print(result[:2] if isinstance(result, list) else result, flush=True)
    except Exception:
        print("[TEST] ERROR al llamar futures_account_balance():", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()

