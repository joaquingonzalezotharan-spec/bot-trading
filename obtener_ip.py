import requests
url = "https://api.ipify.org"
try:
    respuesta = requests.get(url, timeout=10)
    mi_ip = respuesta.text.strip()
    print("\n" + "="*40)
    print(f"[IPIFY] MI IP PUBLICA EN RENDER ES: {mi_ip}")
    print("="*40 + "\n")
except Exception as e:
    print(f"Error al obtener la IP: {e}")

