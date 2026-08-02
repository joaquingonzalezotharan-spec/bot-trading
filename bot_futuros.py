"""
bot_futuros.py

Bot educativo para BTCUSDT (Binance Futuros USDT-M) con 3 estrategias según el régimen:

1) Modo Lateral (Long): señal basada en Bollinger Bands + RSI (sin filtro EMA200).
2) Modo Alcista (Long): señal basada en Bollinger Bands + RSI.
3) Modo Bajista (Short): señal basada en Bollinger Bands + RSI.

Incluye:
- Detector automático de régimen (LATERAL/ALCISTA/BAJISTA) dentro de `main()`.
- Órdenes con SL/TP y redondeo a `stepSize`/`tickSize`.
- Sizing basado en riesgo (risk fraction) y apalancamiento `cfg.leverage`.

Descarga de datos:
- Candles históricos desde Binance Futures vía `futures_klines`.
"""

from __future__ import annotations

from binance import Client  # type: ignore

import argparse
import os
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP


def enviar_telegram(mensaje: str) -> None:
    """
    Envía una notificación a Telegram usando variables de entorno:
    - TELEGRAM_TOKEN
    - TELEGRAM_CHAT_ID
    """
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    try:
        import requests  # type: ignore

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(
            url,
            data={"chat_id": chat_id, "text": mensaje},
            timeout=10,
        )
    except Exception as e:
        # Nunca detengamos el bot por fallos de notificación.
        print(f"[TELEGRAM] No se pudo enviar mensaje: {e}", flush=True)


REPORT_INTERVAL_S = 4 * 60 * 60  # 4 horas


def _align_next_report_epoch_seconds(now_epoch_s: float, interval_s: float) -> float:
    """
    Alinea el próximo reporte a un múltiplo del intervalo (referencia Unix epoch, UTC).
    """
    interval_s = float(interval_s)
    now_epoch_s = float(now_epoch_s)
    next_k = int(now_epoch_s // interval_s) + 1
    return next_k * interval_s


def _format_signed_usdt(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.2f} USDT"


def _calc_pnl_usdt(is_long: bool, entry_price: float, exit_price: float, quantity: float) -> float:
    """
    Aproximación de PnL en USDT por diferencia de precio * cantidad.
    Nota: no incluye comisiones ni funding.
    """
    if is_long:
        return (exit_price - entry_price) * quantity
    return (entry_price - exit_price) * quantity


def _build_periodic_report_message(
    *,
    regime: str,
    operations: List[str],
    net_pnl_usdt: float,
    available_balance_usdt: Optional[float],
) -> str:
    ops_preview_limit = 30
    ops_count = len(operations)
    if ops_count == 0:
        ops_block = "Sin operaciones en las últimas 4 horas."
    else:
        shown = operations[:ops_preview_limit]
        ops_block = "\n".join([f"- {x}" for x in shown])
        if ops_count > ops_preview_limit:
            ops_block += f"\n- ... y {ops_count - ops_preview_limit} operación(es) adicionales."

    if available_balance_usdt is None:
        balance_block = "Balance Binance disponible: N/A"
    else:
        balance_block = f"Balance Binance disponible: {available_balance_usdt:.2f} USDT"

    if net_pnl_usdt >= 0:
        net_block = f"Ganancias netas (USDT): {_format_signed_usdt(net_pnl_usdt)}"
    else:
        net_block = f"Pérdidas netas (USDT): {_format_signed_usdt(net_pnl_usdt)}"

    return (
        "📊 RESUMEN PERIÓDICO DEL BOT\n"
        f"Régimen de Mercado actual: {regime}\n"
        f"Operaciones ejecutadas (últimas 4 horas):\n{ops_block}\n"
        f"{net_block}\n"
        f"{balance_block}"
    )


# -----------------------------
# Configuración de la estrategia
# -----------------------------
@dataclass
class StrategyConfig:
    # Indicadores
    rsi_length: int = 14
    bb_length: int = 20
    bb_std_mult: float = 2.0
    ema_length: int = 200

    # Fracción del margen disponible a usar por trade (en USDT).
    # Ej: 0.10 => 10% del margen disponible (antes de convertir con leverage).
    risk_fraction: float = 0.10

    # Apalancamiento (configurado en Binance al iniciar)
    leverage: int = 5

    # Lateral (Rango)
    lateral_rsi_entry: float = 30.0
    lateral_rsi_exit: float = 70.0
    sl_lateral_pct: float = 0.0030   # -0.30%
    tp_lateral_pct: float = 0.0035   # +0.35%

    # Alcista (Up)
    bullish_rsi_entry: float = 45.0
    sl_bullish_pct: float = 0.0040   # -0.40%
    tp_bullish_pct: float = 0.0100   # +1.00%

    # Bajista (Down / Short)
    # (Live) más flexible para que el modo BAJISTA pueda disparar antes.
    bearish_rsi_entry: float = 55.0
    bearish_rsi_exit: float = 35.0
    sl_bearish_pct: float = 0.0040   # +0.40% (SL arriba para Short)
    tp_bearish_pct: float = 0.0080   # -0.80% (TP abajo para Short)


# -----------------------------
# Descarga de datos (OHLCV)
# -----------------------------
def _interval_to_seconds(interval: str) -> int:
    """
    Convierte un intervalo Binance (ej: '5m', '1h', '1d') a segundos.
    """
    s = interval.strip().lower()
    if not s:
        return 300
    unit = s[-1]
    try:
        n = int(s[:-1])
    except ValueError:
        return 300

    if unit == "m":
        return n * 60
    if unit == "h":
        return n * 3600
    if unit == "d":
        return n * 86400
    return 300


def fetch_binance_futures_candles(
    client: "Client",
    *,
    symbol: str,
    lookback_days: int,
    interval: str,
) -> pd.DataFrame:
    """
    Fetch histórico vía Binance Futures `futures_klines`.

    Retorna DataFrame con columnas:
    - date: open time como `pd.Timestamp` UTC
    - open/high/low/close/volume
    """
    interval_seconds = _interval_to_seconds(interval)
    interval_ms = int(interval_seconds * 1000)

    end_dt = datetime.now(timezone.utc)
    end_ms = int(end_dt.timestamp() * 1000)
    start_ms = end_ms - int(lookback_days * 86400 * 1000)

    # Alineamos al "grid" del intervalo para que las velas coincidan mejor.
    if interval_ms > 0:
        start_ms = start_ms - (start_ms % interval_ms)

    klines: list[list] = []
    cursor_ms = int(start_ms)
    # Evitamos hacer suposiciones de límites; pero para lookbacks típicos
    # (2-10 días) 1000-1500 candelas suelen bastar por request.
    limit_per_call = 1000
    safety_iter = 0
    while cursor_ms < end_ms and safety_iter < 50:
        safety_iter += 1
        chunk = client.futures_klines(
            symbol=symbol,
            interval=interval,
            startTime=cursor_ms,
            endTime=end_ms,
            limit=limit_per_call,
        )
        if not chunk:
            break
        klines.extend(chunk)

        last_open_time_ms = int(chunk[-1][0])
        # Avanzamos a la siguiente vela (open time + intervalo).
        cursor_ms = last_open_time_ms + interval_ms

        # Si devolvió menos del máximo, presumimos que ya no hay más.
        if len(chunk) < limit_per_call:
            break

    columns = ["date", "open", "high", "low", "close", "volume"]
    if not klines:
        return pd.DataFrame(columns=columns)

    rows = []
    for k in klines:
        # k: [openTime, open, high, low, close, volume, closeTime, ...]
        open_time_ms = int(k[0])
        rows.append(
            [
                pd.to_datetime(open_time_ms, unit="ms", utc=True),
                float(k[1]),
                float(k[2]),
                float(k[3]),
                float(k[4]),
                float(k[5]),
            ]
        )

    df = pd.DataFrame(rows, columns=columns)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


# -----------------------------
# Cálculo de indicadores y backtest
# -----------------------------
def prepare_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Calcula indicadores base para 3 regímenes:
    - RSI (14)
    - Bollinger Bands (20, 2σ): bb_middle / bb_upper / bb_lower
    - EMA 200 (filtro de tendencia)

    Además, prepara señales booleanas para cada estrategia:
    - lateral: long_entry_lateral / long_exit_lateral
    - bullish: long_entry_bullish / long_exit_bullish
    - bearish (short): short_entry_bearish / short_exit_bearish
    """
    out = df.copy()

    # -----------------------------
    # RSI (Wilder) nativo con pandas
    # -----------------------------
    delta = out["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / cfg.rsi_length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / cfg.rsi_length, adjust=False).mean()

    rs = avg_gain / avg_loss
    out["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # -----------------------------
    # Bollinger Bands (20, 2σ)
    # -----------------------------
    out["bb_middle"] = out["close"].rolling(cfg.bb_length).mean()
    out["bb_std"] = out["close"].rolling(cfg.bb_length).std(ddof=0)
    out["bb_upper"] = out["bb_middle"] + cfg.bb_std_mult * out["bb_std"]
    out["bb_lower"] = out["bb_middle"] - cfg.bb_std_mult * out["bb_std"]

    out["ema200"] = out["close"].ewm(span=cfg.ema_length, adjust=False).mean()

    out = out.dropna(subset=["rsi", "bb_upper", "bb_lower", "ema200"]).reset_index(drop=True)

    # BB superior expandida (4σ): bb_middle + 4 * std  (como bb_upper ya es 2σ)
    out["bb_upper_expanded"] = out["bb_middle"] + 2.0 * (out["bb_upper"] - out["bb_middle"])

    # Cruce por debajo de la banda inferior (close cruza bajo BB inferior)
    prev_close = out["close"].shift(1)
    prev_lower = out["bb_lower"].shift(1)
    out["bb_cross_below_lower"] = (prev_close >= prev_lower) & (out["close"] < out["bb_lower"])

    # LATERAL (Rango)
    out["long_entry_lateral"] = out["bb_cross_below_lower"] & (out["rsi"] <= cfg.lateral_rsi_entry)
    out["long_exit_lateral"] = (out["high"] >= out["bb_upper"]) | (out["rsi"] > cfg.lateral_rsi_exit)

    # ALCISTA (Up)
    # Señal LONG basada en Bollinger Bands + RSI:
    # - retroceso hacia la línea media (low <= bb_middle)
    # - RSI en zona de entrada
    #
    # El EMA200 se utiliza en el DETECTOR de régimen, pero NO como filtro de la señal
    # en este modo (para que la entrada sea BB+RSI puro).
    out["long_entry_bullish"] = (out["low"] <= out["bb_middle"]) & (out["rsi"] <= cfg.bullish_rsi_entry)
    out["long_exit_bullish"] = out["high"] >= out["bb_upper_expanded"]

    # BAJISTA (Down / Short)
    # Rebound hacia arriba: toca banda superior o línea media + RSI alto
    out["short_entry_bearish"] = (
        ((out["high"] >= out["bb_upper"]) | (out["high"] >= out["bb_middle"]))
        & (out["rsi"] >= cfg.bearish_rsi_entry)
    )
    out["short_exit_bearish"] = (out["low"] <= out["bb_lower"]) | (out["rsi"] < cfg.bearish_rsi_exit)

    return out


def run_backtest(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Simula una sola posición Long a la vez.
    Usa high/low de la vela actual para decidir SL/TP si se tocan.
    """
    position = 0
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_price: Optional[float] = None
    highest_price: Optional[float] = None

    trades: List[dict] = []

    for i in range(len(df)):
        row = df.iloc[i]

        if position == 0:
            if bool(row["long_signal"]):
                position = 1
                entry_price = float(row["close"])
                stop_price = entry_price * (1.0 - cfg.stop_loss_pct)
                take_price = entry_price * (1.0 + cfg.take_profit_pct)
                highest_price = entry_price  # se irá actualizando con el máximo observado

                trades.append({
                    "entry_index": i,
                    "entry_date": row.get("date", None),
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "take_price": take_price,
                    "exit_index": None,
                    "exit_date": None,
                    "exit_price": None,
                    "exit_reason": None,
                    "pnl_pct": None,
                })

        else:
            trade = trades[-1]

            high = float(row["high"])
            low = float(row["low"])

            # Actualizamos el máximo alcanzado desde la entrada.
            if highest_price is not None and high > highest_price:
                highest_price = high

            # Stop efectivo (SL fijo vs trailing).
            # Para que el trailing tenga sentido, solo lo activamos una vez que el precio
            # haya subido por encima del precio de entrada.
            effective_stop = float(stop_price)
            stop_reason = "stop_loss"

            if highest_price is not None and entry_price is not None and highest_price > entry_price:
                trailing_stop_price = highest_price * (1.0 - cfg.trailing_stop_pct)
                effective_stop = max(float(stop_price), float(trailing_stop_price))
                stop_reason = "trailing_stop" if effective_stop > float(stop_price) else "stop_loss"

            hit_stop = low <= effective_stop
            hit_take = high >= float(take_price)
            exit_by_rsi = bool(row["exit_signal"])

            # Determinamos salida por prioridades si ocurrieran ambos
            exit_reason = None
            exit_price = None

            if hit_stop and hit_take:
                if cfg.tp_first:
                    exit_reason = "take_profit (prioridad TP)"
                    exit_price = float(take_price)
                else:
                    exit_reason = f"{stop_reason} (prioridad SL)"
                    exit_price = effective_stop
            elif hit_take:
                exit_reason = "take_profit"
                exit_price = float(take_price)
            elif hit_stop:
                exit_reason = stop_reason
                exit_price = effective_stop
            elif exit_by_rsi:
                exit_reason = "rsi_exit"
                exit_price = float(row["close"])
            else:
                continue

            trade["exit_index"] = i
            trade["exit_date"] = row.get("date", None)
            trade["exit_price"] = exit_price
            trade["exit_reason"] = exit_reason
            trade["pnl_pct"] = (exit_price - float(trade["entry_price"])) / float(trade["entry_price"])

            # Cerramos la posición
            position = 0
            entry_price = None
            stop_price = None
            take_price = None
            highest_price = None

    return pd.DataFrame(trades)


def print_summary(trades_df: pd.DataFrame) -> None:
    """
    Muestra un resumen simple para principiantes.
    """
    print("\n=== RESUMEN DE TRADES ===", flush=True)
    if trades_df.empty:
        print("No se generaron operaciones con estas reglas.", flush=True)
        return

    cols = ["entry_date", "entry_price", "exit_date", "exit_price", "exit_reason", "pnl_pct"]
    existing_cols = [c for c in cols if c in trades_df.columns]
    print(trades_df[existing_cols].to_string(index=False), flush=True)

    print("\nEstadísticas:", flush=True)
    print(f"Operaciones: {len(trades_df)}", flush=True)
    print(f"PnL promedio (%): {trades_df['pnl_pct'].mean():.4f}", flush=True)
    print(f"PnL total (%): {trades_df['pnl_pct'].sum():.4f}", flush=True)


# -----------------------------
# Modo live (primer paso)
# -----------------------------
def fetch_live_prices(
    symbol: str,
    exchange_id: str = "binance",
    timeframe: str = "15m",
    limit: int = 300,
) -> pd.DataFrame:
    """
    Descarga OHLCV (no solo "precio actual") usando ccxt en modo público.

    Importante:
    - Esto NO usa llaves/API keys.
    - Requiere instalar ccxt: pip install ccxt
    - Retorna DataFrame con: date/open/high/low/close/volume
    """
    try:
        import ccxt  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Para modo LIVE necesitas 'ccxt'. Instálalo con: pip install ccxt"
        ) from e

    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Exchange '{exchange_id}' no existe en tu instalación de ccxt.")

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    return df


def run_live(
    symbol: str,
    exchange_id: str,
    timeframe: str,
    cfg: StrategyConfig,
) -> None:
    """
    Bucle continuo que revisa señales y mantiene una posición Long simple.

    Nota educativa:
    - La decisión SL/TP/trailing se toma usando high/low de la última vela descargada.
    - No incluye apalancamiento, fees, ni ejecución real con ordenes.
    """
    position = 0
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_price: Optional[float] = None
    highest_price: Optional[float] = None
    last_processed_ts: Optional[pd.Timestamp] = None

    # Cada 15m (o el timeframe que uses) esperamos alineados al reloj para reducir drift
    # Convertimos '15m' a segundos: este script asume minutos (m).
    tf_minutes = 15
    if timeframe.endswith("m"):
        try:
            tf_minutes = int(timeframe[:-1])
        except ValueError:
            tf_minutes = 15
    tf_seconds = tf_minutes * 60

    while True:
        print("[LIVE] Buscando señales en el mercado en vivo...", flush=True)
        df = fetch_live_prices(symbol=symbol, exchange_id=exchange_id, timeframe=timeframe, limit=300)

        # Si no hay datos, esperamos y reintentamos
        if df.empty:
            time.sleep(5)
            continue

        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)

        df = df.sort_values("date").reset_index(drop=True)
        ind = prepare_indicators(df, cfg)
        if ind.empty:
            time.sleep(5)
            continue

        last = ind.iloc[-1]
        current_ts = ind["date"].iloc[-1]

        # Evitamos procesar la misma vela repetidamente
        if last_processed_ts is not None and current_ts == last_processed_ts:
            sleep_s = max(1.0, tf_seconds - (time.time() % tf_seconds) + 1.0)
            time.sleep(sleep_s)
            continue

        last_processed_ts = current_ts

        if position == 0:
            if bool(last["long_signal"]):
                position = 1
                entry_price = float(last["close"])
                stop_price = entry_price * (1.0 - cfg.stop_loss_pct)
                take_price = entry_price * (1.0 + cfg.take_profit_pct)
                highest_price = float(last["high"])

                print(
                    f"[LIVE] ENTRADA LONG @ {entry_price:.2f} | SL={stop_price:.2f} | TP={take_price:.2f}",
                    flush=True,
                )
        else:
            # Actualizamos trailing
            high = float(last["high"])
            low = float(last["low"])

            if highest_price is not None and high > highest_price:
                highest_price = high

            effective_stop = float(stop_price)
            stop_reason = "stop_loss"
            if highest_price is not None and entry_price is not None and highest_price > entry_price:
                trailing_stop_price = highest_price * (1.0 - cfg.trailing_stop_pct)
                effective_stop = max(float(stop_price), float(trailing_stop_price))
                stop_reason = "trailing_stop" if effective_stop > float(stop_price) else "stop_loss"

            hit_stop = low <= effective_stop
            hit_take = high >= float(take_price)
            exit_by_rsi = bool(last["exit_signal"])

            exit_reason = None
            exit_price = None

            if hit_stop and hit_take:
                if cfg.tp_first:
                    exit_reason = "take_profit (prioridad TP)"
                    exit_price = float(take_price)
                else:
                    exit_reason = f"{stop_reason} (prioridad SL)"
                    exit_price = effective_stop
            elif hit_take:
                exit_reason = "take_profit"
                exit_price = float(take_price)
            elif hit_stop:
                exit_reason = stop_reason
                exit_price = effective_stop
            elif exit_by_rsi:
                exit_reason = "rsi_exit"
                exit_price = float(last["close"])

            if exit_reason is not None:
                pnl_pct = (exit_price - float(entry_price)) / float(entry_price)
                print(
                    f"[LIVE] SALIDA {exit_reason} @ {exit_price:.2f} | PnL={pnl_pct*100:.4f}%",
                    flush=True,
                )

                position = 0
                entry_price = None
                stop_price = None
                take_price = None
                highest_price = None

        sleep_s = max(1.0, tf_seconds - (time.time() % tf_seconds) + 1.0)
        time.sleep(sleep_s)


def get_binance_futures_client() -> Optional["Client"]:
    """
    Crea un cliente de Binance USDT-M Futures leyendo credenciales desde variables de entorno.
    """
    if Client is None:
        return None

    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        return None

    # Proxy residencial (Render) con credenciales
    proxy_ip = os.environ.get("PROXY_IP")
    proxy_user = os.environ.get("PROXY_USER")
    proxy_password = os.environ.get("PROXY_PASSWORD")

    proxy_url = None
    requests_params = None

    if proxy_ip and proxy_user and proxy_password:
        # En caso de caracteres especiales en user/pass, los codificamos para que el URL sea válido.
        from urllib.parse import quote

        proxy_user_enc = quote(proxy_user, safe="")
        proxy_pass_enc = quote(proxy_password, safe="")
        proxy_url = f"http://{proxy_user_enc}:{proxy_pass_enc}@{proxy_ip}:50100"

        proxies = {"http": proxy_url, "https": proxy_url}
        requests_params = {"proxies": proxies}

    return Client(api_key=api_key, api_secret=api_secret, requests_params=requests_params)


def execute_futures_market_buy(
    client: "Client",
    symbol: str = "BTCUSDT",
    quantity: float = 0.001,
) -> None:
    """
    Ejemplo de orden de mercado para futuros perp (USDT-M) sin llaves hardcodeadas.
    """
    client.futures_create_order(
        symbol=symbol,
        side="BUY",
        type="MARKET",
        quantity=quantity,
    )


def _round_down_to_step(quantity: float, step_size: float) -> float:
    """
    Redondea hacia abajo según el stepSize para evitar errores de LOT_SIZE.
    """
    step = Decimal(str(step_size))
    qty = Decimal(str(quantity))
    rounded = (qty // step) * step
    return float(rounded)


def _round_up_to_step(quantity: float, step_size: float) -> float:
    """
    Redondea hacia arriba según el stepSize para no caer por debajo del mínimo.
    """
    step = Decimal(str(step_size))
    qty = Decimal(str(quantity))
    rounded = (qty / step).to_integral_value(rounding=ROUND_UP) * step
    return float(rounded)


def _round_down_to_tick(price: float, tick_size: float) -> float:
    """
    Redondea hacia abajo según el tickSize para evitar errores de PRICE_FILTER.
    """
    tick = Decimal(str(tick_size))
    p = Decimal(str(price))
    rounded = (p // tick) * tick
    return float(rounded)


def _get_futures_symbol_filters(client: "Client", symbol: str) -> tuple[float, float]:
    """
    Devuelve (step_size, tick_size) para el símbolo en Binance Futuros USDT-M.
    """
    info = client.futures_exchange_info()
    sym = next((s for s in info["symbols"] if s["symbol"] == symbol), None)
    if sym is None:
        raise RuntimeError(f"No encontré el símbolo {symbol} en futures_exchange_info().")

    step_size: Optional[float] = None
    tick_size: Optional[float] = None

    for f in sym.get("filters", []):
        if f.get("filterType") == "LOT_SIZE":
            step_size = float(f["stepSize"])
        if f.get("filterType") == "PRICE_FILTER":
            tick_size = float(f["tickSize"])

    if step_size is None or tick_size is None:
        raise RuntimeError(f"No pude obtener step/tick para {symbol}.")

    return step_size, tick_size


def _get_available_margin_usdt(client: "Client") -> float:
    """
    Lee el margen disponible en USDT desde futures_account().
    """
    account = client.futures_account()
    # En binance, suele existir "availableBalance"
    for key in ("availableBalance", "available_balance"):
        if key in account and account[key] is not None:
            return float(account[key])
    # fallback
    return float(account.get("totalWalletBalance", 0.0))


def _compute_quantity_from_risk(
    *,
    available_margin_usdt: float,
    cfg: StrategyConfig,
    entry_price: float,
    step_size: float,
) -> float:
    """
    Convierte margen disponible -> cantidad (qty) según cfg.risk_fraction y cfg.leverage.
    """
    if available_margin_usdt <= 0:
        return 0.0
    if entry_price <= 0:
        return 0.0

    risk_budget_usdt = float(available_margin_usdt) * float(cfg.risk_fraction)
    if risk_budget_usdt <= 0:
        return 0.0

    # notional ≈ margin * leverage
    desired_qty = (risk_budget_usdt * float(cfg.leverage)) / float(entry_price)
    qty_up = _round_up_to_step(desired_qty, step_size)
    if qty_up <= 0:
        return 0.0

    margin_used_up = (qty_up * float(entry_price)) / float(cfg.leverage)
    if margin_used_up <= risk_budget_usdt + 1e-9:
        return qty_up

    # Si el redondeo excede el presupuesto, bajamos.
    qty_down = _round_down_to_step(desired_qty, step_size)
    if qty_down <= 0:
        return 0.0

    margin_used_down = (qty_down * float(entry_price)) / float(cfg.leverage)
    if margin_used_down <= risk_budget_usdt + 1e-9:
        return qty_down

    # Safety: ajustar un paso adicional si aún excede por errores numéricos.
    qty = qty_down
    while qty > 0:
        margin_used = (qty * float(entry_price)) / float(cfg.leverage)
        if margin_used <= risk_budget_usdt + 1e-9:
            return qty
        qty = _round_down_to_step(qty - step_size, step_size)
    return 0.0


def _get_open_position_amt(client: "Client", symbol: str) -> float:
    """
    Devuelve positionAmt (positivo = LONG).
    """
    pos = client.futures_position_information(symbol=symbol)
    if not pos:
        return 0.0
    return float(pos[0]["positionAmt"])


def _cancel_all_open_orders(client: "Client", symbol: str) -> None:
    try:
        client.futures_cancel_all_open_orders(symbol=symbol)
    except Exception:
        # Si el endpoint difiere o falla, lo ignoramos (los create-order del stop pueden fallar más tarde).
        pass


def _place_long_with_stop(
    client: "Client",
    symbol: str,
    tick_size: float,
    entry_price: float,
    sl_pct: float,
    tp_pct: float,
    quantity: float,
) -> None:
    """
    Abre una posición LONG MARKET y crea órdenes:
    - STOP_MARKET (Stop Loss) a entry*(1 - sl_pct)
    - TAKE_PROFIT_MARKET (Take Profit) a entry*(1 + tp_pct)
    """
    if quantity <= 0:
        raise RuntimeError("Cantidad inválida (<= 0).")

    client.futures_create_order(
        symbol=symbol,
        side="BUY",
        type="MARKET",
        quantity=quantity,
    )

    stop_price = _round_down_to_tick(entry_price * (1.0 - sl_pct), tick_size)
    client.futures_create_order(
        symbol=symbol,
        side="SELL",
        type="STOP_MARKET",
        quantity=quantity,
        stopPrice=stop_price,
        reduceOnly=True,
    )

    take_profit_price = _round_down_to_tick(entry_price * (1.0 + tp_pct), tick_size)
    client.futures_create_order(
        symbol=symbol,
        side="SELL",
        type="TAKE_PROFIT_MARKET",
        quantity=quantity,
        stopPrice=take_profit_price,
        reduceOnly=True,
    )

    enviar_telegram(
        f"[OPEN LONG] {symbol} | entrada={entry_price:.2f} | SL={stop_price:.2f} | TP={take_profit_price:.2f} | qty={quantity}"
    )


def _close_long_market(
    client: "Client",
    symbol: str,
    step_size: float,
) -> None:
    """
    Cierra LONG con una orden MARKET SELL (reduceOnly) usando el positionAmt actual.
    """
    position_amt = _get_open_position_amt(client, symbol)
    if position_amt <= 0:
        return

    quantity = _round_down_to_step(position_amt, step_size)
    if quantity <= 0:
        return

    _cancel_all_open_orders(client, symbol)

    client.futures_create_order(
        symbol=symbol,
        side="SELL",
        type="MARKET",
        quantity=quantity,
        reduceOnly=True,
    )


def _place_short_with_sl_tp(
    client: "Client",
    symbol: str,
    tick_size: float,
    entry_price: float,
    sl_pct: float,
    tp_pct: float,
    quantity: float,
) -> None:
    """
    Abre una posición SHORT MARKET y crea órdenes:
    - STOP_MARKET (Stop Loss) a entry*(1 + sl_pct)
    - TAKE_PROFIT_MARKET (Take Profit) a entry*(1 - tp_pct)
    """
    if quantity <= 0:
        raise RuntimeError("Cantidad inválida (<= 0).")

    # Entrada SHORT
    client.futures_create_order(
        symbol=symbol,
        side="SELL",
        type="MARKET",
        quantity=quantity,
    )

    # Usamos el precio actual (marca/último) para evitar que Binance rechace
    # las órdenes condicionales con: "Order would immediately trigger".
    current_price = float(entry_price)
    try:
        ticker = client.futures_symbol_ticker(symbol=symbol)
        if ticker and "price" in ticker:
            current_price = float(ticker["price"])
    except Exception:
        pass

    stop_price = _round_down_to_tick(entry_price * (1.0 + sl_pct), tick_size)
    # Para STOP_MARKET BUY en SHORT, el stop debe quedar por ENCIMA del precio actual.
    if stop_price <= current_price:
        stop_price = _round_down_to_tick(current_price + 2.0 * tick_size, tick_size)

    client.futures_create_order(
        symbol=symbol,
        side="BUY",
        type="STOP_MARKET",
        quantity=quantity,
        stopPrice=stop_price,
        reduceOnly=True,
    )

    take_profit_price = _round_down_to_tick(entry_price * (1.0 - tp_pct), tick_size)
    # Para un TP en SHORT (cierre con BUY cuando cae el precio), evitamos que quede
    # por encima/igual al precio actual. Si ocurre, lo empujamos hacia abajo 2 ticks.
    if take_profit_price >= current_price:
        take_profit_price = _round_down_to_tick(current_price - 2.0 * tick_size, tick_size)

    # Nota: en algunos casos Binance rechaza TAKE_PROFIT_MARKET inmediato.
    # Usamos STOP_MARKET inverso para el TP del SHORT: side=BUY con stopPrice por debajo.
    client.futures_create_order(
        symbol=symbol,
        side="BUY",
        type="STOP_MARKET",
        quantity=quantity,
        stopPrice=take_profit_price,
        reduceOnly=True,
    )

    enviar_telegram(
        f"[OPEN SHORT] {symbol} | entrada={entry_price:.2f} | SL={stop_price:.2f} | TP={take_profit_price:.2f} | qty={quantity}"
    )


def _close_short_market(
    client: "Client",
    symbol: str,
    step_size: float,
) -> None:
    """
    Cierra SHORT con una orden MARKET BUY (reduceOnly) usando positionAmt actual.
    """
    position_amt = _get_open_position_amt(client, symbol)
    if position_amt >= 0:
        return

    quantity = _round_down_to_step(abs(position_amt), step_size)
    if quantity <= 0:
        return

    _cancel_all_open_orders(client, symbol)

    client.futures_create_order(
        symbol=symbol,
        side="BUY",
        type="MARKET",
        quantity=quantity,
        reduceOnly=True,
    )


# -----------------------------
# Punto de entrada (main)
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Bot Futuros BTC con 3 regímenes: Lateral/Alcista/Bajista (Long/Short).")
    parser.add_argument("--lookback-days", type=int, default=2, help="Días hacia atrás para descargar datos desde Binance.")
    parser.add_argument("--interval", type=str, default="5m", help="Intervalo (por defecto: 5m).")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Símbolo Binance USDT-M (ej: BTCUSDT).")
    args = parser.parse_args()

    cfg = StrategyConfig()
    binance_client = get_binance_futures_client()
    step_size: Optional[float] = None
    tick_size: Optional[float] = None

    if binance_client is None:
        print("[LIVE] Sin credenciales Binance o python-binance; modo señal (sin órdenes).", flush=True)
    else:
        try:
            binance_client.futures_change_leverage(symbol=args.symbol, leverage=cfg.leverage)
            print(f"[LIVE] Leverage configurado: {cfg.leverage}x para {args.symbol}.", flush=True)
            enviar_telegram("¡Hola! El bot de trading se ha conectado correctamente a Render y las notificaciones están activas. ✅")
        except Exception as e:
            print(f"[LIVE] No pude configurar leverage ({e}). Continuo.", flush=True)

        step_size, tick_size = _get_futures_symbol_filters(binance_client, args.symbol)
        print(f"[LIVE] Filtros Binance: step_size={step_size} | tick_size={tick_size}", flush=True)

        # Asegura modo ISOLATED margin type (si falla porque ya está aislado, lo ignoramos).
        try:
            binance_client.futures_change_margin_type(symbol=args.symbol, marginType="ISOLATED")
        except Exception as e:
            msg = str(e).lower()
            if "isolated" in msg or "margin type" in msg or "no need" in msg or "already" in msg:
                pass
            else:
                print(f"[LIVE] No pude cambiar marginType a ISOLATED ({e}). Continuo.", flush=True)

    enviar_telegram(
        f"[BOT INICIADO] symbol={args.symbol} | interval={args.interval} | leverage={cfg.leverage}x | risk_fraction={cfg.risk_fraction}"
    )

    last_processed_ts: Optional[pd.Timestamp] = None
    in_long_state = False
    in_short_state = False
    entry_price_state: Optional[float] = None
    entry_quantity_state: Optional[float] = None

    regime_lookback = 10
    cross_window = 6

    # Alineamos descargas/decisiones a cierres de vela para reducir requests.
    tf_seconds = 300.0
    if args.interval.endswith("m"):
        try:
            tf_seconds = float(int(args.interval[:-1]) * 60)
        except ValueError:
            tf_seconds = 300.0
    elif args.interval.endswith("h"):
        try:
            tf_seconds = float(int(args.interval[:-1]) * 3600)
        except ValueError:
            tf_seconds = 300.0
    elif args.interval.endswith("d"):
        try:
            tf_seconds = float(int(args.interval[:-1]) * 86400)
        except ValueError:
            tf_seconds = 300.0
    if tf_seconds <= 0:
        tf_seconds = 300.0

    print("[LIVE] Bot activo: detectando régimen y operando con BB+RSI+EMA200...", flush=True)

    # Métricas de la ventana actual de 4 horas (reiniciadas tras cada reporte).
    period_operations: List[str] = []
    period_net_pnl_usdt: float = 0.0
    last_detected_regime: str = "LATERAL"
    next_report_epoch_s = _align_next_report_epoch_seconds(time.time(), REPORT_INTERVAL_S)

    def _sleep_with_report(max_sleep_s: float) -> None:
        """
        Duerme como máximo `max_sleep_s`, pero sin pasar la hora exacta
        del próximo reporte para reducir el retraso.
        """
        remaining = next_report_epoch_s - time.time()
        if remaining <= 0:
            return
        time.sleep(min(float(max_sleep_s), float(remaining)))

    def _sleep_until_next_candle() -> None:
        """
        Duerme hasta el siguiente cierre de vela según `tf_seconds`,
        sin retrasar el próximo reporte periódico (4h).
        """
        now = time.time()
        next_candle_epoch_s = (math.floor(now / tf_seconds) + 1) * tf_seconds
        sleep_s = max(0.0, next_candle_epoch_s - now + 1.0)  # margen para que Binance tenga la vela lista

        remaining_to_report = next_report_epoch_s - now
        if remaining_to_report <= 0:
            return

        # No dormir más que lo que falta para el reporte 4h.
        sleep_s = min(sleep_s, remaining_to_report)
        if sleep_s > 0:
            time.sleep(sleep_s)

    while True:
        try:
            # Si ya toca enviar el reporte de 4 horas, lo hacemos antes de calcular
            # señales/operaciones para minimizar el retraso.
            now_epoch_s = time.time()
            if now_epoch_s >= next_report_epoch_s:
                available_balance_usdt: Optional[float] = None
                if binance_client is not None:
                    try:
                        available_balance_usdt = _get_available_margin_usdt(binance_client)
                    except Exception:
                        available_balance_usdt = None

                report_msg = _build_periodic_report_message(
                    regime=last_detected_regime,
                    operations=period_operations,
                    net_pnl_usdt=period_net_pnl_usdt,
                    available_balance_usdt=available_balance_usdt,
                )
                enviar_telegram(report_msg)

                # Reinicio de métricas para el nuevo ciclo de 4 horas.
                period_operations = []
                period_net_pnl_usdt = 0.0

                next_report_epoch_s = next_report_epoch_s + REPORT_INTERVAL_S
                if next_report_epoch_s <= now_epoch_s:
                    next_report_epoch_s = _align_next_report_epoch_seconds(now_epoch_s, REPORT_INTERVAL_S)
                print("[REPORT] Enviado resumen periódico 4h.", flush=True)

            # Reducimos rate-limit alineando la descarga a cierres de vela.
            if binance_client is None:
                _sleep_with_report(60)
                continue

            _sleep_until_next_candle()
            print("[LIVE] Revisando mercado real...", flush=True)

            try:
                df_live = fetch_binance_futures_candles(
                    binance_client,
                    symbol=args.symbol,
                    lookback_days=args.lookback_days,
                    interval=args.interval,
                )
            except Exception as e:
                print(f"[BINANCE] Error descargando velas: {e}", flush=True)
                _sleep_with_report(180)
                continue
            if df_live.empty:
                _sleep_with_report(60)
                continue

            for c in ["open", "high", "low", "close"]:
                df_live[c] = df_live[c].astype(float)
            df_live = df_live.sort_values("date").reset_index(drop=True)

            df_ind = prepare_indicators(df_live, cfg)
            if df_ind.empty:
                _sleep_with_report(60)
                continue

            last = df_ind.iloc[-1]
            current_ts = last["date"]
            if last_processed_ts is not None and current_ts == last_processed_ts:
                _sleep_with_report(60)
                continue
            last_processed_ts = current_ts

            last_close = float(last["close"])
            last_ema200 = float(last["ema200"])

            window = df_ind.tail(regime_lookback)
            slope_bb = float(window["bb_middle"].iloc[-1] - window["bb_middle"].iloc[0])
            rel_slope = slope_bb / last_close if last_close else 0.0

            rel = window["close"] - window["bb_middle"]
            crosses = (((rel.shift(1) >= 0) & (rel < 0)) | ((rel.shift(1) <= 0) & (rel > 0))).tail(cross_window).sum()
            bands_horizontal = abs(rel_slope) < 0.00005

            if crosses >= 2 and bands_horizontal:
                regime = "LATERAL"
            elif last_close > last_ema200 and slope_bb > 0:
                regime = "ALCISTA"
            elif last_close < last_ema200 and slope_bb < 0:
                regime = "BAJISTA"
            else:
                regime = "LATERAL"

            if regime == "BAJISTA":
                print("[LIVE] Régimen Detectado: MERCADO BAJISTA (Buscando Shorts)", flush=True)
            elif regime == "ALCISTA":
                print("[LIVE] Régimen Detectado: MERCADO ALCISTA (Buscando Longs)", flush=True)
            else:
                print("[LIVE] Régimen Detectado: MERCADO LATERAL (Rango)", flush=True)

            last_rsi = float(last["rsi"])
            last_bb_lower = float(last["bb_lower"])
            last_bb_cross_below_lower = bool(last["bb_cross_below_lower"])
            last_long_entry_lateral = bool(last["long_entry_lateral"])
            last_long_entry_bullish = bool(last["long_entry_bullish"])

            print(
                f"[DIAGÓSTICO] Régimen: {regime} | Close: {last_close} | RSI: {last_rsi:.2f} | BB_Lower: {last_bb_lower:.2f} | ¿Cruce_Lower?: {last_bb_cross_below_lower} | ¿Entry_Lateral?: {last_long_entry_lateral} | ¿Entry_Bullish?: {last_long_entry_bullish}",
                flush=True,
            )

            last_detected_regime = regime

            if binance_client is None:
                _sleep_with_report(60)
                continue

            assert step_size is not None and tick_size is not None

            position_amt = _get_open_position_amt(binance_client, symbol=args.symbol)
            in_long = position_amt > 0
            in_short = position_amt < 0

            # Cierres por STOP/TP (notificar al detectar transición)
            if in_long_state and (not in_long) and entry_price_state is not None:
                exit_price = last_close
                if entry_quantity_state is not None:
                    pnl_usdt = _calc_pnl_usdt(
                        is_long=True,
                        entry_price=float(entry_price_state),
                        exit_price=float(exit_price),
                        quantity=float(entry_quantity_state),
                    )
                    period_net_pnl_usdt += pnl_usdt
                    now_utc = datetime.now(timezone.utc)
                    period_operations.append(
                        f"{now_utc.strftime('%H:%M UTC')} CLOSE LONG | entrada={entry_price_state:.2f} | salida={exit_price:.2f} | PnL={pnl_usdt:.2f} USDT"
                    )
                pnl_pct = (exit_price - entry_price_state) / entry_price_state * 100.0
                signo = "GANANCIA" if pnl_pct >= 0 else "PERDIDA"
                enviar_telegram(f"[CLOSE LONG] {args.symbol} | entrada={entry_price_state:.2f} | salida={exit_price:.2f} | PnL={pnl_pct:.4f}% ({signo})")
                in_long_state = False
                entry_price_state = None
                entry_quantity_state = None

            if in_short_state and (not in_short) and entry_price_state is not None:
                exit_price = last_close
                if entry_quantity_state is not None:
                    pnl_usdt = _calc_pnl_usdt(
                        is_long=False,
                        entry_price=float(entry_price_state),
                        exit_price=float(exit_price),
                        quantity=float(entry_quantity_state),
                    )
                    period_net_pnl_usdt += pnl_usdt
                    now_utc = datetime.now(timezone.utc)
                    period_operations.append(
                        f"{now_utc.strftime('%H:%M UTC')} CLOSE SHORT | entrada={entry_price_state:.2f} | salida={exit_price:.2f} | PnL={pnl_usdt:.2f} USDT"
                    )
                pnl_pct = (entry_price_state - exit_price) / entry_price_state * 100.0
                signo = "GANANCIA" if pnl_pct >= 0 else "PERDIDA"
                enviar_telegram(f"[CLOSE SHORT] {args.symbol} | entrada={entry_price_state:.2f} | salida={exit_price:.2f} | PnL={pnl_pct:.4f}% ({signo})")
                in_short_state = False
                entry_price_state = None
                entry_quantity_state = None

            long_entry_lateral = bool(last["long_entry_lateral"])
            long_exit_lateral = bool(last["long_exit_lateral"])
            long_entry_bullish = bool(last["long_entry_bullish"])
            long_exit_bullish = bool(last["long_exit_bullish"])
            short_entry_bearish = bool(last["short_entry_bearish"])
            short_exit_bearish = bool(last["short_exit_bearish"])

            # Aperturas si no hay posición
            if (not in_long) and (not in_short):
                if regime == "LATERAL" and long_entry_lateral:
                    assert step_size is not None
                    available_margin_usdt = _get_available_margin_usdt(binance_client)
                    quantity = _compute_quantity_from_risk(
                        available_margin_usdt=available_margin_usdt,
                        cfg=cfg,
                        entry_price=last_close,
                        step_size=step_size,
                    )
                    if quantity <= 0:
                        print("[LIVE] Qty calculada <= 0. Saltando apertura LONG.", flush=True)
                    else:
                        _place_long_with_stop(
                            binance_client,
                            args.symbol,
                            tick_size,
                            last_close,
                            cfg.sl_lateral_pct,
                            cfg.tp_lateral_pct,
                            quantity,
                        )
                        in_long_state = True
                        entry_price_state = last_close
                        entry_quantity_state = quantity
                elif regime == "ALCISTA" and long_entry_bullish:
                    assert step_size is not None
                    available_margin_usdt = _get_available_margin_usdt(binance_client)
                    quantity = _compute_quantity_from_risk(
                        available_margin_usdt=available_margin_usdt,
                        cfg=cfg,
                        entry_price=last_close,
                        step_size=step_size,
                    )
                    if quantity <= 0:
                        print("[LIVE] Qty calculada <= 0. Saltando apertura LONG.", flush=True)
                    else:
                        _place_long_with_stop(
                            binance_client,
                            args.symbol,
                            tick_size,
                            last_close,
                            cfg.sl_bullish_pct,
                            cfg.tp_bullish_pct,
                            quantity,
                        )
                        in_long_state = True
                        entry_price_state = last_close
                        entry_quantity_state = quantity
                elif regime == "BAJISTA" and short_entry_bearish:
                    assert step_size is not None
                    available_margin_usdt = _get_available_margin_usdt(binance_client)
                    quantity = _compute_quantity_from_risk(
                        available_margin_usdt=available_margin_usdt,
                        cfg=cfg,
                        entry_price=last_close,
                        step_size=step_size,
                    )
                    if quantity <= 0:
                        print("[LIVE] Qty calculada <= 0. Saltando apertura SHORT.", flush=True)
                    else:
                        _place_short_with_sl_tp(
                            binance_client,
                            args.symbol,
                            tick_size,
                            last_close,
                            cfg.sl_bearish_pct,
                            cfg.tp_bearish_pct,
                            quantity,
                        )
                        in_short_state = True
                        entry_price_state = last_close
                        entry_quantity_state = quantity

            # Cierres manuales por señal
            else:
                if in_long_state:
                    if regime == "LATERAL" and long_exit_lateral:
                        _close_long_market(binance_client, args.symbol, step_size=step_size)
                        exit_price = last_close
                        if entry_quantity_state is not None:
                            pnl_usdt = _calc_pnl_usdt(
                                is_long=True,
                                entry_price=float(entry_price_state) if entry_price_state is not None else 0.0,
                                exit_price=float(exit_price),
                                quantity=float(entry_quantity_state),
                            )
                            period_net_pnl_usdt += pnl_usdt
                            now_utc = datetime.now(timezone.utc)
                            period_operations.append(
                                f"{now_utc.strftime('%H:%M UTC')} CLOSE LONG | entrada={entry_price_state:.2f} | salida={exit_price:.2f} | PnL={pnl_usdt:.2f} USDT"
                            )
                        pnl_pct = (exit_price - entry_price_state) / entry_price_state * 100.0
                        signo = "GANANCIA" if pnl_pct >= 0 else "PERDIDA"
                        enviar_telegram(f"[CLOSE LONG] {args.symbol} | entrada={entry_price_state:.2f} | salida={exit_price:.2f} | PnL={pnl_pct:.4f}% ({signo})")
                        in_long_state = False
                        entry_price_state = None
                        entry_quantity_state = None
                    elif regime == "ALCISTA" and long_exit_bullish:
                        _close_long_market(binance_client, args.symbol, step_size=step_size)
                        exit_price = last_close
                        if entry_quantity_state is not None:
                            pnl_usdt = _calc_pnl_usdt(
                                is_long=True,
                                entry_price=float(entry_price_state) if entry_price_state is not None else 0.0,
                                exit_price=float(exit_price),
                                quantity=float(entry_quantity_state),
                            )
                            period_net_pnl_usdt += pnl_usdt
                            now_utc = datetime.now(timezone.utc)
                            period_operations.append(
                                f"{now_utc.strftime('%H:%M UTC')} CLOSE LONG | entrada={entry_price_state:.2f} | salida={exit_price:.2f} | PnL={pnl_usdt:.2f} USDT"
                            )
                        pnl_pct = (exit_price - entry_price_state) / entry_price_state * 100.0
                        signo = "GANANCIA" if pnl_pct >= 0 else "PERDIDA"
                        enviar_telegram(f"[CLOSE LONG] {args.symbol} | entrada={entry_price_state:.2f} | salida={exit_price:.2f} | PnL={pnl_pct:.4f}% ({signo})")
                        in_long_state = False
                        entry_price_state = None
                        entry_quantity_state = None

                if in_short_state:
                    if regime == "BAJISTA" and short_exit_bearish:
                        _close_short_market(binance_client, args.symbol, step_size=step_size)
                        exit_price = last_close
                        if entry_quantity_state is not None:
                            pnl_usdt = _calc_pnl_usdt(
                                is_long=False,
                                entry_price=float(entry_price_state) if entry_price_state is not None else 0.0,
                                exit_price=float(exit_price),
                                quantity=float(entry_quantity_state),
                            )
                            period_net_pnl_usdt += pnl_usdt
                            now_utc = datetime.now(timezone.utc)
                            period_operations.append(
                                f"{now_utc.strftime('%H:%M UTC')} CLOSE SHORT | entrada={entry_price_state:.2f} | salida={exit_price:.2f} | PnL={pnl_usdt:.2f} USDT"
                            )
                        pnl_pct = (entry_price_state - exit_price) / entry_price_state * 100.0
                        signo = "GANANCIA" if pnl_pct >= 0 else "PERDIDA"
                        enviar_telegram(f"[CLOSE SHORT] {args.symbol} | entrada={entry_price_state:.2f} | salida={exit_price:.2f} | PnL={pnl_pct:.4f}% ({signo})")
                        in_short_state = False
                        entry_price_state = None
                        entry_quantity_state = None

            # Enviar reporte periódico cada 4 horas (alineado por epoch, UTC).
            now_epoch_s = time.time()
            if now_epoch_s >= next_report_epoch_s:
                available_balance_usdt: Optional[float] = None
                if binance_client is not None:
                    try:
                        available_balance_usdt = _get_available_margin_usdt(binance_client)
                    except Exception:
                        available_balance_usdt = None

                report_msg = _build_periodic_report_message(
                    regime=last_detected_regime,
                    operations=period_operations,
                    net_pnl_usdt=period_net_pnl_usdt,
                    available_balance_usdt=available_balance_usdt,
                )
                enviar_telegram(report_msg)

                # Reinicio de métricas para el nuevo ciclo de 4 horas.
                period_operations = []
                period_net_pnl_usdt = 0.0

                next_report_epoch_s = next_report_epoch_s + REPORT_INTERVAL_S
                if next_report_epoch_s <= now_epoch_s:
                    next_report_epoch_s = _align_next_report_epoch_seconds(now_epoch_s, REPORT_INTERVAL_S)
                print("[REPORT] Enviado resumen periódico 4h.", flush=True)

        except Exception as e:
            print(f"[LIVE] Error en el bucle: {e}", flush=True)

        # Evitamos saltarnos el próximo reporte.
        _sleep_with_report(60)


if __name__ == "__main__":
    main()

