from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

app = FastAPI(title="WhatsApp Insurance Chatbot - Federación Patronal")

WEBHOOK_VERIFY_TOKEN = "token_secreto_grupo_ernandes"

# Configuración Meta WhatsApp Cloud API (recomendado cargar por variables de entorno)
# - WHATSAPP_TOKEN: token del sistema / acceso
# - WHATSAPP_PHONE_NUMBER_ID: identificador del número en Meta
GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "19.0")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")


# ----------------------------
# 1) Base de datos integrada
# ----------------------------
# Esta "base" se mantiene en memoria por ahora (diccionario/strings).
# En un siguiente paso podés reemplazarla por una BD real sin tocar
# el resto de la lógica de la app.
KB: Dict[str, Any] = {
    "agencia": {
        "nombre": "Federación Patronal - Grupo Ernandes (Mar del Plata)",
        "direccion": "20 de Septiembre 1253, Mar del Plata",
        "contacto_general": "Decinos si te interesa Siniestros o Producción y te derivamos.",
        "protocolo_denuncia_72hs": (
            "Protocolo de denuncia dentro de las 72 horas:\n"
            "1) Contanos lo ocurrido y confirmá tus datos (nombre y teléfono de contacto).\n"
            "2) Enviá la información esencial del siniestro (fecha, lugar y tipo de evento).\n"
            "3) Si tenés, adjuntá/compartí fotos, denuncia policial o cualquier comprobante disponible.\n"
            "4) Te indicaremos los siguientes pasos para completar la denuncia.\n"
            "5) Importante: iniciá la denuncia lo antes posible para cumplir con el plazo."
        ),
    },
    "whatsapp": {
        "siniestros": "223-586-4410",
        "produccion": "223-535-2604",
    },
}


# -------------------------------------------------------
# 2) Helpers: extraer mensaje entrante de WhatsApp (Meta)
# -------------------------------------------------------
def extract_whatsapp_message(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Intenta obtener (sender_wa_id, message_text) del payload típico de Meta WhatsApp Cloud API.

    Si Meta cambia el formato o el mensaje no tiene texto, retornará (None, None).
    """
    try:
        # Estructura típica:
        # entry[0].changes[0].value.messages[0].from
        # entry[0].changes[0].value.messages[0].text.body
        entry = payload.get("entry") or []
        if not entry:
            return None, None

        changes = (entry[0].get("changes") or [])
        if not changes:
            return None, None

        value = changes[0].get("value") or {}
        messages = value.get("messages") or []
        if not messages:
            return None, None

        msg0 = messages[0] or {}
        sender = msg0.get("from")  # WA ID
        msg_type = msg0.get("type")

        if msg_type == "text":
            text = (msg0.get("text") or {}).get("body")
            return sender, text

        # Si llegara un mensaje que no sea de texto, lo dejamos como no soportado por ahora.
        return sender, None
    except Exception:
        # Nunca romper el webhook por un payload raro.
        return None, None


# -------------------------------------------------------
# 3) Respuestas (exclusivamente con la base de conocimiento)
# -------------------------------------------------------
def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def generate_reply_exclusively_from_db(user_text: str) -> str:
    """
    Genera una respuesta usando ÚNICAMENTE KB (sin llamar a APIs externas).
    """
    t = normalize_text(user_text)

    # Heurísticas simples para derivar contenido.
    if any(k in t for k in ["siniestro", "siniestros", "accidente", "daño", "denuncia", "denunciar", "robo"]):
        return (
            f"Para Siniestros, comunicáte por WhatsApp: {KB['whatsapp']['siniestros']}.\n\n"
            f"{KB['agencia']['protocolo_denuncia_72hs']}"
        )

    if any(k in t for k in ["produccion", "producción", "cotizacion", "cotización", "cotizar", "seguro", "seguros", "presupuesto", "plan"]):
        return (
            f"Para Producción / cotizaciones, comunicáte por WhatsApp: {KB['whatsapp']['produccion']}.\n\n"
            f"{KB['agencia']['contacto_general']}"
        )

    if any(k in t for k in ["direccion", "dirección", "donde", "dónde", "ubicacion", "ubicación", "local"]):
        return f"Nuestra dirección es: {KB['agencia']['direccion']}."

    # Respuesta general (si no detectamos intención).
    return (
        f"Hola. Soy el bot de {KB['agencia']['nombre']}.\n\n"
        f"Contanos si tu consulta es por *Siniestros* o por *Producción* (cotizaciones):\n"
        f"- Siniestros: {KB['whatsapp']['siniestros']}\n"
        f"- Producción: {KB['whatsapp']['produccion']}\n\n"
        f"Si necesitas denunciar dentro de 72 horas, podés usar este protocolo:\n"
        f"{KB['agencia']['protocolo_denuncia_72hs']}"
    )


# -------------------------------------------------------
# Espacio listo para integrar OpenAI (ChatGPT)
# -------------------------------------------------------
def should_use_openai() -> bool:
    """
    Control por env var.
    Por defecto False para asegurar "exclusiva base de datos".
    """
    return os.getenv("USE_OPENAI", "false").strip().lower() in {"1", "true", "yes", "on"}


async def generate_reply_with_openai(user_text: str) -> str:
    """
    Estructura lista para integrar OpenAI respetando el requisito:
    - Responder usando exclusivamente nuestra base de datos (KB).
    """
    # TODO: Integración real
    # Recomendación (ejemplo conceptual):
    # 1) Construir un prompt con KB completa.
    # 2) Indicar al modelo: "Responde SOLO con la información de KB. No inventes datos."
    # 3) Llamar a la API de OpenAI con el mensaje del usuario.
    #
    # Importante: dejar esto como placeholder para no depender de una clave/red
    # en este primer paso del esqueleto.
    #
    # Si querés, en el próximo paso lo conectamos con la SDK oficial o HTTP.
    _ = user_text
    return generate_reply_exclusively_from_db(user_text)


def send_whatsapp_text(to_wa_id: str, text: str) -> Tuple[bool, str]:
    """
    Envía un mensaje de texto a WhatsApp vía Meta Graph API.

    Retorna (ok, details).
    """
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        return False, "Faltan WHATSAPP_TOKEN y/o WHATSAPP_PHONE_NUMBER_ID (env vars)."

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    body = {
        "messaging_product": "whatsapp",
        "to": to_wa_id,
        "type": "text",
        "text": {"body": text},
    }
    data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Authorization": f"Bearer {WHATSAPP_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_text = resp.read().decode("utf-8")
            # No fallamos si no podemos parsear la respuesta.
            return True, resp_text
    except urllib.error.HTTPError as e:
        try:
            err_text = e.read().decode("utf-8")
        except Exception:
            err_text = str(e)
        return False, f"HTTPError: {err_text}"
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {e}"


# -------------------------
# Endpoint /webhook (POST)
# -------------------------
@app.post("/webhook")
async def webhook(request: Request) -> JSONResponse:
    payload = await request.json()
    sender_wa_id, incoming_text = extract_whatsapp_message(payload)

    if not incoming_text:
        # Aun así contestamos 200 para que Meta no reintente.
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "received_sender": sender_wa_id,
                "reply": "Recibí el mensaje, pero por ahora el bot solo responde a mensajes de texto.",
            },
        )

    if should_use_openai():
        reply_text = await generate_reply_with_openai(incoming_text)
    else:
        # Por defecto: respuesta 100% basada en nuestra KB.
        reply_text = generate_reply_exclusively_from_db(incoming_text)

    sent = False
    send_details = "No intentado (sin sender_wa_id)."
    if sender_wa_id:
        sent, send_details = send_whatsapp_text(sender_wa_id, reply_text)

    # NOTA:
    # Meta WhatsApp normalmente requiere que vos envíes la respuesta vía API
    # (POST a Graph API). En este paso dejamos el "reply_text" listo.
    # Si querés, en el próximo prompt lo conectamos para que el bot responda
    # automáticamente enviando el mensaje al remitente.
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "received_sender": sender_wa_id,
            "reply": reply_text,
            "sent": sent,
            "send_details": send_details,
        },
    )


# --------------------------------
# Endpoint /webhook (GET) - Verificar
# --------------------------------
@app.get("/webhook")
async def webhook_verify(
    hub_mode: Optional[str] = None,
    hub_verify_token: Optional[str] = None,
    hub_challenge: Optional[str] = None,
) -> PlainTextResponse:
    """
    WhatsApp exige validar el webhook antes de poder enviar mensajes.
    Espera parámetros: hub.mode, hub.verify_token, hub.challenge.
    """
    if hub_mode != "subscribe":
        return PlainTextResponse(content="Invalid hub.mode", status_code=403)

    if hub_verify_token != WEBHOOK_VERIFY_TOKEN:
        return PlainTextResponse(content="Invalid verify token", status_code=403)

    if not hub_challenge:
        return PlainTextResponse(content="Missing hub.challenge", status_code=400)

    return PlainTextResponse(content=hub_challenge, status_code=200)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

