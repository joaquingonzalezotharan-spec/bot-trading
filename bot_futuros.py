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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import time

try:
    # Para ejecución en vivo (futuros USDT-M). No se usa en el backtest.
    from binance import Client  # type: ignore
except Exception:
    Client = None  # type: ignore


# -----------------------------
# Configuración de la estrategia
# -----------------------------
@dataclass
class StrategyConfig:
    rsi_length: int = 14
    ema_length: int = 200

    stop_loss_pct: float = 0.015   # 1.5%
    take_profit_pct: float = 0.045 # 4.5%
    trailing_stop_pct: float = 0.01  # 1.0%

    # Si en la misma vela se tocan SL y TP:
    # - False: asume primero el Stop Loss (caso conservador).
    tp_first: bool = False


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
    Calcula RSI(14) y MACD(12,26,9) de forma nativa con pandas y genera columnas para las señales.
    """
    out = df.copy()

    # -----------------------------
    # RSI (Wilder) con pandas
    # -----------------------------
    delta = out["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder: promedio suavizado con alpha=1/length
    avg_gain = gain.ewm(alpha=1.0 / cfg.rsi_length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / cfg.rsi_length, adjust=False).mean()

    rs = avg_gain / avg_loss
    out["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # -----------------------------
    # MACD estándar (12, 26, 9)
    # -----------------------------
    ema_fast = out["close"].ewm(span=12, adjust=False).mean()
    ema_slow = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    out = out.dropna(subset=["rsi", "macd", "macd_signal"]).reset_index(drop=True)

    # Señal de compra (Long)
    # Cruce alcista: MACD pasa de <= Señal a > Señal
    out["macd_cross_up"] = (out["macd"] > out["macd_signal"]) & (out["macd"].shift(1) <= out["macd_signal"].shift(1))
    # Ventana flexible:
    # El RSI pudo estar por debajo de RSI_ENTRY_LEVEL en cualquiera de las últimas N velas.
    RSI_WINDOW = 5
    RSI_ENTRY_LEVEL = 42
    out["rsi_below_recent"] = (out["rsi"] < RSI_ENTRY_LEVEL).rolling(window=RSI_WINDOW, min_periods=1).max().astype(bool)
    out["long_signal"] = out["rsi_below_recent"] & out["macd_cross_up"]
    # Señal de salida por RSI
    out["exit_signal"] = out["rsi"] > 70

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

    return Client(api_key=api_key, api_secret=api_secret)


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


# -----------------------------
# Punto de entrada (main)
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Bot de futuros educativos para BTC usando RSI+EMA.")
    parser.add_argument("--lookback-days", type=int, default=30, help="Cuántos días hacia atrás descargar.")
    parser.add_argument("--interval", type=str, default="15m", help="Intervalo (ej: 15m, 1h, 4h, 1d).")
    parser.add_argument("--tp-first", action="store_true", help="Si en la misma vela se toca SL y TP, prioriza TP.")
    parser.add_argument("--live", action="store_true", help="Ejecuta el bot en modo live (requiere ccxt).")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange para live (ej: binance, bingx).")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Símbolo para live (ej: BTC/USDT).")
    args = parser.parse_args()

    cfg = StrategyConfig(tp_first=args.tp_first)

    if args.live:
        run_live(symbol=args.symbol, exchange_id=args.exchange, timeframe=args.interval, cfg=cfg)
    else:
        print(
            f"[INFO] Descargando datos BTC (lookback={args.lookback_days} días, interval={args.interval})...",
            flush=True,
        )
        df = download_btc_ohlcv(lookback_days=args.lookback_days, interval=args.interval)

        # Convertimos a float por seguridad
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)

        df = df.sort_values("date").reset_index(drop=True)

        print("[INFO] Calculando indicadores (RSI/EMA)...", flush=True)
        df_ind = prepare_indicators(df, cfg)

        print("[INFO] Ejecutando backtest simple...", flush=True)
        trades_df = run_backtest(df_ind, cfg)

        print_summary(trades_df)

        last_processed_ts: Optional[pd.Timestamp] = None
        while True:
            try:
                print("[LIVE] Esperando señal en el mercado real...", flush=True)

                # Mantener el proceso activo en Render
                time.sleep(60)

                df_live = download_btc_ohlcv(lookback_days=5, interval=args.interval)
                for c in ["open", "high", "low", "close"]:
                    df_live[c] = df_live[c].astype(float)

                df_live = df_live.sort_values("date").reset_index(drop=True)
                df_ind_live = prepare_indicators(df_live, cfg)

                if df_ind_live.empty:
                    continue

                current_ts = df_ind_live.iloc[-1]["date"]
                should_react = current_ts != last_processed_ts
                if should_react and bool(df_ind_live.iloc[-1]["long_signal"]):
                    print("[LIVE] Señal LONG detectada (long_signal=True).", flush=True)

                    binance_client = get_binance_futures_client()
                    if binance_client is not None:
                        # Ejemplo: compra market futures cuando long_signal es True.
                        execute_futures_market_buy(binance_client, symbol="BTCUSDT", quantity=0.001)
                        print("[LIVE] Orden futures MARKET BUY enviada (ejemplo).", flush=True)
                    else:
                        print(
                            "[LIVE] No se enviará orden: faltan credenciales BINANCE_API_KEY/BINANCE_API_SECRET o python-binance no instalado.",
                            flush=True,
                        )

                    last_processed_ts = current_ts
            except Exception as e:
                # No detenemos el bucle si ocurre un problema puntual de red/datos
                print(f"[LIVE] Error revisando señales: {e}", flush=True)


if __name__ == "__main__":
    main()

