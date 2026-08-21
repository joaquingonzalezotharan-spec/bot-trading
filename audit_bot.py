#!/usr/bin/env python3
"""
audit_bot.py

Script independiente para auditar toda la operativa histórica de futuros (BTCUSDT por defecto).
- Usa las mismas variables de entorno: BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
- Descarga trades de futures (paginado) y calcula métricas clave
- Envía un reporte resumido a Telegram en un único mensaje

Ejecutar:
  python3 audit_bot.py
"""
from __future__ import annotations
import os
import time
import requests
from datetime import datetime
from typing import List, Dict, Any
from binance.client import Client
from dotenv import load_dotenv

# Cargar variables desde .env en la raíz del proyecto (si existe)
load_dotenv()


def load_env_keys() -> Dict[str, str]:
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    tg_token = os.environ.get("TELEGRAM_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("TELEGRAM_CHAT")

    missing = []
    if not api_key:
        missing.append("BINANCE_API_KEY")
    if not api_secret:
        missing.append("BINANCE_API_SECRET")
    if not tg_token:
        missing.append("TELEGRAM_TOKEN")
    if not tg_chat:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        raise EnvironmentError(f"Faltan variables de entorno: {', '.join(missing)}")

    return {
        "api_key": api_key,
        "api_secret": api_secret,
        "tg_token": tg_token,
        "tg_chat": tg_chat,
    }


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=15)
    except Exception as e:
        print(f"[TELEGRAM] Error enviando mensaje: {e}")


def fetch_all_futures_trades(client: Client, symbol: str = "BTCUSDT") -> List[Dict[str, Any]]:
    """
    Descarga de forma paginada todas las trades de futures para el símbolo indicado.
    Paginación por timestamp: avanzamos startTime = last_time_ms + 1 hasta agotar.
    """
    all_trades: List[Dict[str, Any]] = []
    fetch_start_ms = 0
    while True:
        try:
            batch = client.futures_account_trades(symbol=symbol, startTime=fetch_start_ms, endTime=int(time.time() * 1000), limit=1000)
        except Exception as e:
            print(f"[BINANCE] Error fetching trades: {e}")
            break

        if not batch:
            break

        all_trades.extend(batch)
        last_time = batch[-1].get("time")
        if last_time is None:
            break
        last_time_ms = int(last_time)
        # Avanzar 1ms para no repetir
        fetch_start_ms = last_time_ms + 1
        if len(batch) < 1000:
            break
        # pequeño sleep para no golpear rate limits
        time.sleep(0.2)

    return all_trades


def compute_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filtra trades con realizedPnl != 0 y computa las métricas pedidas.
    """
    closed_trades = []
    for t in trades:
        try:
            rp = float(t.get("realizedPnl", 0.0) or 0.0)
        except Exception:
            rp = 0.0
        if rp == 0.0:
            continue
        closed_trades.append(t)

    total = len(closed_trades)
    wins = 0
    losses = 0
    gross_gains = 0.0
    gross_losses = 0.0
    total_commissions = 0.0
    sum_realized = 0.0

    for t in closed_trades:
        try:
            rp = float(t.get("realizedPnl", 0.0) or 0.0)
        except Exception:
            rp = 0.0
        try:
            comm = float(t.get("commission", 0.0) or 0.0)
        except Exception:
            comm = 0.0

        sum_realized += rp
        total_commissions += abs(comm)

        if rp > 0:
            wins += 1
            gross_gains += rp
        elif rp < 0:
            losses += 1
            gross_losses += rp

    win_rate = (wins / total * 100.0) if total > 0 else 0.0
    net_pnl = sum_realized - total_commissions

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "gross_gains": gross_gains,
        "gross_losses": gross_losses,
        "total_commissions": total_commissions,
        "net_pnl": net_pnl,
    }


def format_report(metrics: Dict[str, Any], symbol: str, generated_at: datetime) -> str:
    lines = [
        "📊 Informe histórico de operativa - Bot Futuros",
        f"Par: {symbol}",
        f"Generado: {generated_at.strftime('%d/%m/%Y %H:%M UTC')}",
        "----------------------------------------",
        f"Total trades cerrados: {metrics['total_trades']}",
        f"Tasa de acierto global: {metrics['win_rate_pct']:.2f}% ({metrics['wins']}/{metrics['total_trades']})",
        f"Ganancias brutas (winners): {metrics['gross_gains']:+.4f} USDT",
        f"Pérdidas brutas (losers): {metrics['gross_losses']:.4f} USDT",
        f"Comisiones totales: -{metrics['total_commissions']:.4f} USDT",
        "----------------------------------------",
        f"PNL Neto final (realized - commissions): {metrics['net_pnl']:+.4f} USDT",
        "",
        "Este reporte cubre TODO el historial disponible en Binance Futures para el par indicado.",
    ]
    return "\n".join(lines)


def main():
    try:
        env = load_env_keys()
    except EnvironmentError as e:
        print(f"[ERROR] {e}")
        return

    api_key = env["api_key"]
    api_secret = env["api_secret"]
    tg_token = env["tg_token"]
    tg_chat = env["tg_chat"]

    symbol = os.environ.get("AUDIT_SYMBOL", "BTCUSDT")

    client = Client(api_key, api_secret)
    print("[INFO] Conectando a Binance y descargando trades (esto puede tardar)...")
    trades = fetch_all_futures_trades(client, symbol=symbol)
    print(f"[INFO] Trades descargados: {len(trades)} (incluye trades con realizedPnl==0)")

    metrics = compute_metrics(trades)
    report = format_report(metrics, symbol, datetime.utcnow())

    print(report)
    print("[INFO] Enviando reporte a Telegram...")
    send_telegram_message(tg_token, tg_chat, report)
    print("[OK] Reporte enviado.")


if __name__ == "__main__":
    main()

