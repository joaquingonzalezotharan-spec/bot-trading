"""
bot_futuros.py

Backtest educativo (simulación) de una estrategia en "futuros" para BTC/USDT usando:
- RSI (14)
- MACD (12, 26, 9)
- Long: el RSI estuvo < 42 en cualquiera de las últimas 5 velas y hay cruce alcista MACD > MACD Signal
- Salida: RSI > 70
- Stop Loss: 1.5%
- Take Profit: 4.5%
- Trailing stop: 1.0%

Descarga automática de datos históricos:
- Usa yfinance (ticker: BTC-USD)

Uso rápido (en tu terminal):
1) Instala dependencias:
   pip install pandas yfinance
2) Ejecuta:
   python bot_futuros.py

Notas para principiantes:
- Esto es una simulación simple (no incluye apalancamiento, fees ni slippage).
- Para un backtest serio se agregan costos, tamaño de posición, ejecución realista, etc.
"""

from __future__ import annotations

import argparse
import os
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import time
from decimal import Decimal, ROUND_DOWN

try:
    # Para ejecución en vivo (futuros USDT-M). No se usa en el backtest.
    from binance import Client  # type: ignore
except Exception:
    Client = None  # type: ignore


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


# -----------------------------
# Configuración de la estrategia
# -----------------------------
@dataclass
class StrategyConfig:
    # Parámetros de la estrategia
    interval_minutes: int = 5

    # Indicadores
    rsi_length: int = 7
    bb_length: int = 20
    bb_std_mult: float = 2.0

    rsi_entry_level: float = 25.0
    rsi_exit_level: float = 75.0

    # Riesgo / ejecución
    leverage: int = 5
    margin_fraction: float = 0.20  # 20% del margen disponible

    stop_loss_pct: float = 0.015  # 1.5% desde el precio de entrada


# -----------------------------
# Descarga de datos (OHLCV)
# -----------------------------
def download_btc_ohlcv(
    lookback_days: int,
    interval: str,
) -> pd.DataFrame:
    """
    Descarga OHLCV de BTC.

    Usa únicamente yfinance (BTC-USD) y retorna columnas:
    open/high/low/close (y volume si está disponible).
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)

    return _download_with_yfinance(start_dt, end_dt, interval)


def _download_with_yfinance(
    start_dt: datetime,
    end_dt: datetime,
    interval: str,
) -> pd.DataFrame:
    """
    Descarga con yfinance.

    Importante:
    - yfinance no suele tener "BTC/USDT" como símbolo exacto.
    - Usamos BTC-USD (fuente gratuita de Yahoo Finance).
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "Falta la dependencia 'yfinance'. Instálala con: pip install yfinance"
        ) from e

    df = yf.download(
        "BTC-USD",
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        # No lanzamos RuntimeError por pedido del usuario.
        # Devolvemos un DataFrame vacío con el formato esperado.
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    # Limpieza robusta del formato de Yahoo Finance.
    # Ejecuta la normalización exactamente después de descargar.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()

    # Aseguramos que existan columnas esperadas.
    # (Yahoo a veces usa "adj close"; si no hay "close", lo copiamos.)
    if "close" not in df.columns and "adj close" in df.columns:
        df["close"] = df["adj close"]
    if "volume" not in df.columns:
        # Si no viene volumen, la estrategia no lo necesita estrictamente, pero
        # aquí lo creamos para que el filtrado df[['open',...,'volume']] no falle.
        df["volume"] = 0.0

    # Si por algún motivo faltara open/high/low/close, intentamos con alternativas comunes.
    if "open" not in df.columns and "o" in df.columns:
        df["open"] = df["o"]
    if "high" not in df.columns and "h" in df.columns:
        df["high"] = df["h"]
    if "low" not in df.columns and "l" in df.columns:
        df["low"] = df["l"]

    # Filtramos columnas esperadas (sin lanzar RuntimeError).
    keep_cols = ["open", "high", "low", "close", "volume"]
    for c in keep_cols:
        if c not in df.columns:
            # Creamos columnas faltantes como NaN para evitar errores de selección.
            df[c] = float("nan")

    df = df[keep_cols].copy()

    df = df.sort_index().copy()
    df["date"] = df.index
    df = df.reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]].copy()


# -----------------------------
# Cálculo de indicadores y backtest
# -----------------------------
def prepare_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Calcula RSI corto (7) y Bandas de Bollinger (20, 2σ) con pandas y genera señales:
    - long_signal: cruza por debajo de BB inferior y RSI <= 25
    - exit_signal: toca/supera BB superior o RSI > 75
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

    out = out.dropna(subset=["rsi", "bb_upper", "bb_lower"]).reset_index(drop=True)

    # -----------------------------
    # Reglas del bot
    # -----------------------------
    # Cruce por debajo de la BB inferior:
    # - en la vela anterior el close estaba >= bb_lower
    # - en la vela actual el close está < bb_lower
    prev_close = out["close"].shift(1)
    prev_lower = out["bb_lower"].shift(1)
    out["bb_cross_below_lower"] = (prev_close >= prev_lower) & (out["close"] < out["bb_lower"])

    # Compra (LONG)
    out["long_signal"] = out["bb_cross_below_lower"] & (out["rsi"] <= cfg.rsi_entry_level)

    # Venta / Cierre de LONG:
    # - toca o supera BB superior (usamos high para ser más realista en velas)
    # - o RSI > 75
    out["exit_signal"] = (out["high"] >= out["bb_upper"]) | (out["rsi"] > cfg.rsi_exit_level)

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
    cfg: StrategyConfig,
    step_size: float,
    tick_size: float,
    entry_price: float,
) -> None:
    """
    Abre una posición LONG MARKET y crea un STOP_MARKET de protección 1.5%.
    """
    available_usdt = _get_available_margin_usdt(client)
    if available_usdt <= 0:
        raise RuntimeError("availableBalance <= 0, no se puede dimensionar la orden.")

    # Queremos usar exactamente el 20% del margen disponible.
    margin_to_use = available_usdt * cfg.margin_fraction
    notional = margin_to_use * cfg.leverage
    raw_qty = notional / entry_price
    quantity = _round_down_to_step(raw_qty, step_size)
    if quantity <= 0:
        raise RuntimeError("Cantidad redondeada a 0; revisa stepSize / precio / margen.")

    client.futures_create_order(
        symbol=symbol,
        side="BUY",
        type="MARKET",
        quantity=quantity,
    )

    stop_price = _round_down_to_tick(entry_price * (1.0 - cfg.stop_loss_pct), tick_size)
    client.futures_create_order(
        symbol=symbol,
        side="SELL",
        type="STOP_MARKET",
        quantity=quantity,
        stopPrice=stop_price,
        reduceOnly=True,
    )

    enviar_telegram(
        f"[OPEN LONG] {symbol} | entrada={entry_price:.2f} | SL={stop_price:.2f} | qty={quantity}"
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


# -----------------------------
# Punto de entrada (main)
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Bot futuros BTC (5m) con RSI(7) + Bollinger y órdenes reales en Binance.")
    parser.add_argument("--lookback-days", type=int, default=2, help="Días hacia atrás para descargar datos con yfinance.")
    parser.add_argument("--interval", type=str, default="5m", help="Intervalo (por defecto: 5m).")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Símbolo Binance USDT-M (ej: BTCUSDT).")
    args = parser.parse_args()

    cfg = StrategyConfig()

    binance_client = get_binance_futures_client()
    step_size: Optional[float] = None
    tick_size: Optional[float] = None

    if binance_client is None:
        print(
            "[LIVE] Sin credenciales BINANCE_API_KEY/BINANCE_API_SECRET o python-binance no instalado; se ejecutará en modo señal (sin órdenes).",
            flush=True,
        )
    else:
        # 1) Configurar apalancamiento 5x al inicio
        try:
            binance_client.futures_change_leverage(symbol=args.symbol, leverage=cfg.leverage)
            print(f"[LIVE] Leverage configurado: {cfg.leverage}x para {args.symbol}.", flush=True)
        except Exception as e:
            print(f"[LIVE] No pude configurar leverage ({e}). Continuo.", flush=True)

        # 2) Obtener filtros para rounding correcto
        step_size, tick_size = _get_futures_symbol_filters(binance_client, args.symbol)
        print(
            f"[LIVE] Filtros Binance: step_size={step_size} | tick_size={tick_size}",
            flush=True,
        )

    enviar_telegram(
        f"[BOT INICIADO] symbol={args.symbol} | interval={args.interval} | leverage={cfg.leverage}x | margin_fraction={cfg.margin_fraction}"
    )

    last_processed_ts: Optional[pd.Timestamp] = None
    in_long_state = False
    entry_price_state: Optional[float] = None
    print("[LIVE] Estrategia activa. Revisando mercado continuamente...", flush=True)

    while True:
        try:
            print("[LIVE] Revisando mercado real...", flush=True)

            df_live = download_btc_ohlcv(lookback_days=args.lookback_days, interval=args.interval)
            for c in ["open", "high", "low", "close"]:
                df_live[c] = df_live[c].astype(float)
            df_live = df_live.sort_values("date").reset_index(drop=True)

            df_ind_live = prepare_indicators(df_live, cfg)
            if df_ind_live.empty:
                time.sleep(60)
                continue

            last = df_ind_live.iloc[-1]
            current_ts = last["date"]
            if last_processed_ts is not None and current_ts == last_processed_ts:
                time.sleep(60)
                continue

            last_processed_ts = current_ts

            long_signal = bool(last["long_signal"])
            exit_signal = bool(last["exit_signal"])
            last_close = float(last["close"])

            if binance_client is None:
                if long_signal:
                    print("[LIVE] Señal LONG detectada (modo señal, sin órdenes).", flush=True)
                continue

            assert step_size is not None and tick_size is not None

            position_amt = _get_open_position_amt(binance_client, symbol=args.symbol)
            in_long = position_amt > 0

            # Notifica cierres que ocurren por STOP_MARKET u otras órdenes
            if in_long_state and (not in_long) and entry_price_state is not None:
                exit_price = last_close
                pnl_pct = (exit_price - entry_price_state) / entry_price_state * 100.0
                signo = "GANANCIA" if pnl_pct >= 0 else "PERDIDA"
                enviar_telegram(
                    f"[CLOSE LONG] {args.symbol} | entrada={entry_price_state:.2f} | salida={exit_price:.2f} | PnL={pnl_pct:.4f}% ({signo})"
                )
                in_long_state = False
                entry_price_state = None

            if not in_long:
                if long_signal:
                    print(
                        f"[LIVE] Ejecutando LONG: close={last_close:.2f} (RSI <= {cfg.rsi_entry_level}, BB cross).",
                        flush=True,
                    )
                    _place_long_with_stop(
                        client=binance_client,
                        symbol=args.symbol,
                        cfg=cfg,
                        step_size=step_size,
                        tick_size=tick_size,
                        entry_price=last_close,
                    )
                    in_long_state = True
                    entry_price_state = last_close
            else:
                if exit_signal:
                    print(
                        f"[LIVE] Ejecutando CLOSE LONG: close={last_close:.2f} (BB tocada o RSI> {cfg.rsi_exit_level}).",
                        flush=True,
                    )
                    _close_long_market(
                        client=binance_client,
                        symbol=args.symbol,
                        step_size=step_size,
                    )
                    if entry_price_state is not None:
                        exit_price = last_close
                        pnl_pct = (exit_price - entry_price_state) / entry_price_state * 100.0
                        signo = "GANANCIA" if pnl_pct >= 0 else "PERDIDA"
                        enviar_telegram(
                            f"[CLOSE LONG] {args.symbol} | entrada={entry_price_state:.2f} | salida={exit_price:.2f} | PnL={pnl_pct:.4f}% ({signo})"
                        )
                    in_long_state = False
                    entry_price_state = None

        except Exception as e:
            # No detenemos el proceso por fallos de red/datos
            print(f"[LIVE] Error en el bucle: {e}", flush=True)

        # Mantener Render activo
        time.sleep(60)


if __name__ == "__main__":
    main()

