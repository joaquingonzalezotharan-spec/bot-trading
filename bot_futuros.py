import os
import time
import argparse
import logging
import math
import sys
from datetime import datetime, timedelta
import pytz
from urllib.parse import quote
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
# =====================================================================
# CONFIGURACIÓN DE LOGGING
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# =====================================================================
# RIESGO FIJO POR OPERACIÓN (USD)
# =====================================================================
# Pérdida máxima fija estimada en USD por operación (bajo el SL configurado).
RISK_PER_TRADE = 7.5


def calculate_qty_fixed_risk(
    client: Client,
    symbol: str,
    entry_price: float,
    sl_pct: float,
    cfg: StrategyConfig,
    decimals: int = 3,
    min_qty: float = 0.001,
) -> float:
    """
    qty = RISK_PER_TRADE / (entry_price * sl_pct)
    - truncado estricto hacia abajo a `decimals` para no inflar por precisión
    - capado por margen disponible con leverage cfg.leverage
    """
    if entry_price <= 0 or sl_pct <= 0:
        return 0.0

    qty = RISK_PER_TRADE / (entry_price * sl_pct)

    factor = 10**decimals
    qty = math.floor(qty * factor) / factor
    if qty < min_qty:
        qty = min_qty

    # Cap por margen disponible (estimación: required_margin ~= notional/leverage)
    try:
        account = client.futures_account()
        balance = float(account["totalMarginBalance"])
    except Exception:
        return qty

    max_qty = (balance * float(cfg.leverage)) / entry_price
    max_qty = math.floor(max_qty * factor) / factor

    if qty > max_qty:
        qty = max_qty

    if qty < min_qty:
        return 0.0

    return float(qty)


def get_effective_position_amt(
    client: Client,
    symbol: str,
    *,
    position_amt: float,
    lookback_seconds: int = 180,
) -> float:
    """
    Binance puede tardar en reflejar cambios de positionAmt tras cierres rápidos.
    Si detectamos que el SL/TP reduceOnly ya se llenó recientemente, forzamos a 0
    para evitar bloqueos en la lógica de entrada.
    """
    if abs(position_amt) == 0:
        return 0.0

    # Si aún hay órdenes abiertas, mantenemos la posición como activa.
    try:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        if open_orders:
            return position_amt
    except Exception:
        # Si no podemos consultar, no hacemos suposiciones.
        return position_amt

    now_ms = int(time.time() * 1000)
    lookback_ms = lookback_seconds * 1000

    # Revisamos órdenes recientes llenadas (reduceOnly) de SL/TP.
    try:
        all_orders = client.futures_get_all_orders(symbol=symbol, limit=20)
        for o in all_orders:
            status = o.get("status")
            o_type = o.get("type")
            reduce_only = o.get("reduceOnly")
            if not reduce_only:
                continue
            if status != "FILLED":
                continue
            if o_type not in ("STOP_MARKET", "LIMIT"):
                continue

            update_time = o.get("updateTime") or o.get("time")
            if update_time is None:
                continue

            try:
                update_ms = int(update_time)
            except Exception:
                continue

            if now_ms - update_ms <= lookback_ms:
                return 0.0
    except Exception:
        return position_amt

    return position_amt
# =====================================================================
# ENVÍO DE NOTIFICACIONES TELEGRAM (con el mismo proxy que Binance)
# =====================================================================
def enviar_telegram(mensaje: str, *, proxies: dict | None = None) -> None:
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
            proxies=proxies,
        )
    except Exception as e:
        logger.warning(f"[TELEGRAM] No se pudo enviar mensaje: {e}", exc_info=True)
# =====================================================================
# CONFIGURACIÓN DE LA ESTRATEGIA (StrategyConfig)
# =====================================================================
class StrategyConfig:
    rsi_length = 14
    bb_length = 20
    bb_std_mult = 2.0
    ema_length = 200
    
    risk_fraction = 0.10             
    max_margin_per_trade_pct = 0.05  
    leverage = 5                     

    # Interruptor de seguridad por drawdown diario (UTC).
    max_daily_loss_usd = 20.0
    last_pnl_check_date = None
    
    lateral_rsi_entry = 32.0
    lateral_rsi_exit = 70.0
    sl_lateral_pct = 0.0040          
    tp_lateral_pct = 0.0035          
    
    bullish_rsi_entry = 65.0
    sl_bullish_pct = 0.0035          # Stop Loss tendencial: 0.35%
    tp_bullish_pct = 0.0065          # Take Profit tendencial: 0.65%
    
    bearish_rsi_entry = 55.0
    bearish_rsi_exit = 35.0
    sl_bearish_pct = 0.0040          
    tp_bearish_pct = 0.0085          # Take Profit tendencial: 0.85%
    
    regime_lookback = 10
    cross_window = 6
# =====================================================================
# CÁLCULO DE INDICADORES (prepare_indicators)
# =====================================================================
def prepare_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = df.copy()
    df['ema_200'] = df['close'].ewm(span=cfg.ema_length, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/cfg.rsi_length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/cfg.rsi_length, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['bb_middle'] = df['close'].rolling(window=cfg.bb_length).mean()
    df['bb_std'] = df['close'].rolling(window=cfg.bb_length).std()
    df['bb_upper'] = df['bb_middle'] + (cfg.bb_std_mult * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (cfg.bb_std_mult * df['bb_std'])
    
    df['bb_upper_expanded'] = df['bb_middle'] + 1.5 * (df['bb_upper'] - df['bb_middle'])
    df['slope_bb'] = df['bb_middle'].diff(periods=3)

    # Filtro de volumen: promedio de las últimas 20 velas.
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    # Filtro de volumen alternativo para régimen lateral (más sensible a rangos cortos).
    df["vol_avg5"] = df["volume"].rolling(5).mean()
    
    return df
# =====================================================================
# DETECCIÓN DE RÉGIMEN
# =====================================================================
def detect_regime(df: pd.DataFrame, cfg: StrategyConfig) -> str:
    if len(df) < max(cfg.ema_length, cfg.regime_lookback):
        return "LATERAL"
    
    last_row = df.iloc[-1]
    if last_row['close'] > last_row['ema_200'] and last_row['slope_bb'] > 0:
        return "ALCISTA"
    elif last_row['close'] < last_row['ema_200'] and last_row['slope_bb'] < 0:
        return "BAJISTA"
    else:
        return "LATERAL"
# =====================================================================
# GESTIÓN DE RIESGO Y REDONDEO
# =====================================================================
def calculate_position_size(client: Client, symbol: str, entry_price: float, sl_pct: float, cfg: StrategyConfig) -> float:
    try:
        account = client.futures_account()
        balance = float(account['totalMarginBalance'])
        
        info = client.futures_exchange_info()
        symbol_info = next(item for item in info['symbols'] if item['symbol'] == symbol)
        
        lot_size_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
        step_size = float(lot_size_filter['stepSize'])
        precision = int(round(-math.log10(step_size)))
        
        risk_usd = balance * cfg.risk_fraction
        stop_loss_dist = entry_price * sl_pct
        qty_by_risk = risk_usd / stop_loss_dist
        
        max_margin_usd = balance * cfg.max_margin_per_trade_pct
        max_qty_by_margin = (max_margin_usd * cfg.leverage) / entry_price
        
        final_qty = min(qty_by_risk, max_qty_by_margin)
        final_qty = math.floor(final_qty / step_size) * step_size
        # IMPORTANTE: nunca redondear hacia arriba.
        # Dado que `final_qty` ya está truncado a múltiplos de `step_size`,
        # devolvemos el valor tal cual (evita inflar qty por `round()`/float).
        return float(final_qty)
    except Exception as e:
        logger.error(f"Error al calcular el tamaño de la posición: {e}")
        return 0.0
def round_price(price: float, symbol_info: dict) -> float:
    price_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER')
    tick_size = float(price_filter['tickSize'])
    precision = int(round(-math.log10(tick_size)))
    return round(math.floor(price / tick_size) * tick_size, precision)
# =====================================================================
# EJECUCIÓN DE ÓRDENES (TP Maker LIMIT / SL Market)
# =====================================================================
def estimate_fee_friction_rate() -> float:
    """
    Ajuste de fricción para absorber fees de ida y vuelta.
    Se usa el ajuste fijo requerido por las ecuaciones de compensación.
    """
    maker_fee = 0.0002  # 0.02%
    taker_fee = 0.0005  # 0.05%
    logger.info(f"[FEES] Maker={maker_fee:.6f} Taker={taker_fee:.6f} | friction_used=0.0007")
    return 0.0007

def set_trade_exits(client, symbol, side, entry_price, qty, tp_pct, sl_pct):
    """
    Coloca de forma obligatoria y blindada las órdenes de Take Profit (Limit) 
    y Stop Loss (Stop Limit) calculando los precios según el régimen actual.
    """
    try:
        friction = estimate_fee_friction_rate()
        # Ajuste matemático de los precios según la dirección del trade
        if side == "LONG":
            tp_price = entry_price * (1 + tp_pct + friction)
            sl_trigger = entry_price * (1 - sl_pct)
            sl_limit = sl_trigger * 0.9995  # Holgura de protección para asegurar ejecución
            exit_side = "SELL"
        elif side == "SHORT":
            tp_price = entry_price * (1 - tp_pct - friction)
            sl_trigger = entry_price * (1 + sl_pct)
            sl_limit = sl_trigger * 1.0005  # Holgura de protección para asegurar ejecución
            exit_side = "BUY"
        else:
            print("[ERROR] Dirección de trade 'side' no válida.")
            return

        # Redondeo estricto para BTCUSDT (Paso de precio: 0.10 USDT)
        tp_price = round(float(tp_price), 1)
        sl_trigger = round(float(sl_trigger), 1)
        sl_limit = round(float(sl_limit), 1)
        qty = abs(float(qty))
        print(f"[API] Configurando salidas para {side}. Entrada: {entry_price} | TP Objetivo: {tp_price} | SL Disparador: {sl_trigger}")

        # Envío y reintento de la orden de Take Profit (LIMIT)
        tp_placed = False
        for intento in range(3):
            try:
                client.futures_create_order(
                    symbol=symbol,
                    side=exit_side,
                    type="LIMIT",
                    timeInForce="GTC",
                    quantity=qty,
                    price=str(tp_price),
                    reduceOnly=True
                )
                print(f"[API] Orden de Take Profit Limit sembrada con éxito a un precio de: {tp_price}")
                tp_placed = True
                break
            except Exception as e_tp:
                print(f"[ALERTA] Intento {intento + 1} fallido para TP. Redondeando y reintentando... Error: {e_tp}")
                tp_price = round(tp_price, 1)

        if not tp_placed:
            print("[CRÍTICO] No se pudo colocar el Take Profit tras 3 intentos. Revisar Binance manualmente.")

        # Envío de la orden de Stop Loss (STOP_LIMIT) para mitigar el deslizamiento de precios
        try:
            client.futures_create_order(
                symbol=symbol,
                side=exit_side,
                type="STOP",
                quantity=qty,
                stopPrice=str(sl_trigger),
                price=str(sl_limit),
                reduceOnly=True
            )
            print(f"[API] Orden de Stop Limit sembrada con éxito a un precio de: {sl_limit} (Disparador: {sl_trigger})")
        except Exception as e_sl:
            print(f"[CRÍTICO] Falló el envío del Stop Loss Limit: {e_sl}. Reintentando con orden de emergencia de mercado...")
            client.futures_create_order(
                symbol=symbol,
                side=exit_side,
                type="STOP_MARKET",
                quantity=qty,
                stopPrice=str(sl_trigger),
                reduceOnly=True
            )
            print(f"[API] Orden de Stop Market de emergencia colocada a un precio de: {sl_trigger}")
    except Exception as e_general:
        print(f"[ERROR GENERAL EN ÓRDENES DE SALIDA] No se pudo procesar la lógica de TP/SL: {e_general}")

def _place_long_with_stop(
    client: Client,
    symbol: str,
    qty: float,
    entry_price: float,
    sl_pct: float,
    tp_pct: float,
    symbol_info: dict,
    proxies: dict | None = None,
):
    try:
        logger.info(f"[ORDEN] Abriendo posición LONG en LIMIT. Cantidad: {qty} | price={entry_price}")
        entry_order = client.futures_create_order(
            symbol=symbol,
            side="BUY",
            type="LIMIT",
            timeInForce="GTC",
            quantity=qty,
            price=str(entry_price),
        )
        order_id = entry_order.get("orderId")

        # Espera máxima: 30s para que el LIMIT ejecute.
        confirmed_pos_amt = 0.0
        entry_price_exec = float(entry_price)
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                pos_info = client.futures_position_information(symbol=symbol)
                pos_amt = float(pos_info[0]["positionAmt"]) if pos_info else 0.0
                if pos_amt > 0:
                    confirmed_pos_amt = pos_amt
                    try:
                        ep = pos_info[0].get("entryPrice") or pos_info[0].get("entry_price")
                        entry_price_exec = float(ep) if ep is not None else entry_price_exec
                    except Exception:
                        pass
                    break
            except Exception:
                pass
            time.sleep(1)
        
        if confirmed_pos_amt <= 0:
            # Cancelamos el LIMIT no ejecutado.
            if order_id is not None:
                try:
                    client.futures_cancel_order(symbol=symbol, orderId=order_id)
                    logger.warning(f"[ORDEN] LIMIT LONG no ejecutado en 30s. Cancelado orderId={order_id}.")
                except BinanceAPIException as e_cancel:
                    logger.warning(f"[ORDEN] Falló cancel de LIMIT LONG (orderId={order_id}): {e_cancel}", exc_info=True)
            logger.warning("[ORDEN] No se confirmó positionAmt LONG antes de SL/TP; omitiendo.")
            return
        
        # Blindaje TP/SL (Stop Limit + TP Limit) con redondeo BTCUSDT paso 0.10.
        friction = 0.0007
        tp_trigger_price = round(float(entry_price_exec * (1 + tp_pct + friction)), 1)
        sl_trigger_price = round(float(entry_price_exec * (1 - sl_pct)), 1)
        set_trade_exits(
            client=client,
            symbol=symbol,
            side="LONG",
            entry_price=entry_price_exec,
            qty=qty,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
        )

        # Notificación visual inmediata desde el móvil.
        volumen_btc = float(confirmed_pos_amt)
        msg = (
            "*APERTURA DE POSICION EJECUTADA*\n"
            f"Activo: {symbol}\n"
            "Direccion: LONG\n"
            f"Volumen: {volumen_btc:.3f} BTC\n"
            f"Precio de Entrada: {entry_price_exec:,.2f} USDT\n"
            f"Take Profit: {tp_trigger_price:,.2f} USDT\n"
            f"Stop Loss: {sl_trigger_price:,.2f} USDT"
        )
        enviar_telegram(msg, proxies=proxies)
    except BinanceAPIException as e:
        logger.error(f"Error de Binance al ejecutar Long Setup: {e}")
    except Exception as e:
        logger.error(f"Error inesperado al ejecutar Long Setup: {e}", exc_info=True)
def _place_short_with_sl_tp(
    client: Client,
    symbol: str,
    qty: float,
    entry_price: float,
    sl_pct: float,
    tp_pct: float,
    symbol_info: dict,
    proxies: dict | None = None,
):
    try:
        logger.info(f"[ORDEN] Abriendo posición SHORT en LIMIT. Cantidad: {qty} | price={entry_price}")
        entry_order = client.futures_create_order(
            symbol=symbol,
            side="SELL",
            type="LIMIT",
            timeInForce="GTC",
            quantity=qty,
            price=str(entry_price),
        )
        order_id = entry_order.get("orderId")

        # Espera máxima: 30s para que el LIMIT ejecute.
        confirmed_pos_amt = 0.0
        entry_price_exec = float(entry_price)
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                pos_info = client.futures_position_information(symbol=symbol)
                pos_amt = float(pos_info[0]["positionAmt"]) if pos_info else 0.0
                if pos_amt < 0:
                    confirmed_pos_amt = pos_amt
                    try:
                        ep = pos_info[0].get("entryPrice") or pos_info[0].get("entry_price")
                        entry_price_exec = float(ep) if ep is not None else entry_price_exec
                    except Exception:
                        pass
                    break
            except Exception:
                pass
            time.sleep(1)
        
        if confirmed_pos_amt >= 0:
            # Cancelamos el LIMIT no ejecutado.
            if order_id is not None:
                try:
                    client.futures_cancel_order(symbol=symbol, orderId=order_id)
                    logger.warning(f"[ORDEN] LIMIT SHORT no ejecutado en 30s. Cancelado orderId={order_id}.")
                except BinanceAPIException as e_cancel:
                    logger.warning(f"[ORDEN] Falló cancel de LIMIT SHORT (orderId={order_id}): {e_cancel}", exc_info=True)
            logger.warning("[ORDEN] No se confirmó positionAmt SHORT antes de SL/TP; omitiendo.")
            return
        
        # Blindaje TP/SL (Stop Limit + TP Limit) con redondeo BTCUSDT paso 0.10.
        friction = 0.0007
        tp_trigger_price = round(float(entry_price_exec * (1 - tp_pct - friction)), 1)
        sl_trigger_price = round(float(entry_price_exec * (1 + sl_pct)), 1)
        set_trade_exits(
            client=client,
            symbol=symbol,
            side="SHORT",
            entry_price=entry_price_exec,
            qty=qty,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
        )

        # Notificación visual inmediata desde el móvil.
        volumen_btc = float(confirmed_pos_amt)
        msg = (
            "*APERTURA DE POSICION EJECUTADA*\n"
            f"Activo: {symbol}\n"
            "Direccion: SHORT\n"
            f"Volumen: {volumen_btc:.3f} BTC\n"
            f"Precio de Entrada: {entry_price_exec:,.2f} USDT\n"
            f"Take Profit: {tp_trigger_price:,.2f} USDT\n"
            f"Stop Loss: {sl_trigger_price:,.2f} USDT"
        )
        enviar_telegram(msg, proxies=proxies)
    except BinanceAPIException as e:
        logger.error(f"Error de Binance al ejecutar Short Setup: {e}")
    except Exception as e:
        logger.error(f"Error inesperado al ejecutar Short Setup: {e}", exc_info=True)
# =====================================================================
# LOOP PRINCIPAL Y ARRANQUE
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', default='5m')
    parser.add_argument('--lookback-days', type=int, default=2)
    args = parser.parse_args()
    
    cfg = StrategyConfig()
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        logger.warning(
            "WARNING: Credenciales Binance no configuradas (BINANCE_API_KEY/BINANCE_API_SECRET). "
            "El bot continuará en modo no garantizado, pero no abortará el contenedor."
        )
    # -----------------------------
    # PROXY (HTTP autenticado) para Binance Futures
    # -----------------------------
    # Usuario/contraseña del proveedor (los datos ya fueron provistos).
    proxy_user = "joaquingonzalezotharan"
    proxy_password = "JbADCjWM8g"
    proxy_port = 50100

    # IMPORTANTE: aquí debes colocar el HOST/IP del proxy que te dio tu proveedor.
    # Ejemplo: proxy_host = "203.0.113.10" o "mi-proxy.midominio.com"
    # Host/Dominio del proxy (proveedor).
    # IMPORTANTE: si el proveedor NO quiere permitir IPs “dinámicas”,
    # usa el DOMINIO principal del proveedor aquí (vía variable de entorno)
    # y deja que la autenticación dependa solo de user/pass.
    proxy_host = "31.59.239.182"

    # Codificamos usuario/clave para que el URL sea válido.
    proxy_user_enc = quote(proxy_user, safe="")
    proxy_password_enc = quote(proxy_password, safe="")
    proxy_url = f"http://{proxy_user_enc}:{proxy_password_enc}@{proxy_host}:{proxy_port}"
    requests_params = {"proxies": {"http": proxy_url, "https": proxy_url}}

    client = Client(api_key or "", api_secret or "", requests_params=requests_params)
    enviar_telegram(
        f"[BOT INICIADO] symbol={args.symbol} | interval={args.interval} | leverage={cfg.leverage}x",
        proxies=requests_params.get("proxies"),
    )
    bot_start_ts_ms = int(time.time() * 1000)

    def hard_reset_monthly_pnl_cache() -> None:
        # Seguridad anti-persistencia física (caché viejo corrupto).
        # Borra archivos locales típicos si existieran.
        cache_candidates = [
            "pnl_monthly_cache.json",
            "pnl_monthly_cache.csv",
            "pnl_monthly_cache.txt",
            "monthly_pnl_cache.json",
            "monthly_pnl_cache.csv",
            "monthly_pnl_cache.txt",
            "pnl_cache.json",
            "pnl_cache.csv",
            "pnl_cache.txt",
        ]
        for fn in cache_candidates:
            try:
                if os.path.exists(fn):
                    os.remove(fn)
            except Exception:
                # Nunca romper el bot por una limpieza de caché.
                pass

    hard_reset_monthly_pnl_cache()

    def enviar_reporte_historico_total() -> None:
        """
        Reporte histórico total alineado con la operativa desde un origen fijo hasta el día actual.
        Importante: Binance Futures limita el intervalo máximo por request (7 días), por eso paginamos.
        """
        try:
            # Origen estimado del sistema (ajustable): 01/01/2025 00:00:00 UTC.
            origen_dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
            current_start = int(origen_dt.timestamp() * 1000)
            now_ms = int(time.time() * 1000)

            if current_start >= now_ms:
                return

            block_ms = 7 * 24 * 60 * 60 * 1000  # 604800000 ms

            total_trades = 0
            win_trades = 0
            loss_trades = 0
            pnl_bruto = 0.0
            comisiones = 0.0

            while current_start < now_ms:
                current_end = min(now_ms, current_start + block_ms)
                fetch_start = current_start

                # Paginación interna dentro del mismo bloque (si hay más de 1000 trades).
                while fetch_start < current_end:
                    month_batch = client.futures_account_trades(
                        symbol="BTCUSDT",
                        startTime=fetch_start,
                        endTime=current_end,
                        limit=1000,
                    )
                    if not month_batch:
                        break

                    for t in month_batch:
                        total_trades += 1
                        rp = float(t.get("realizedPnl", 0.0) or 0.0)
                        c = float(t.get("commission", 0.0) or 0.0)
                        pnl_bruto += rp
                        comisiones += c
                        if rp >= 0:
                            win_trades += 1
                        else:
                            loss_trades += 1

                    last_time = month_batch[-1].get("time")
                    if last_time is None:
                        break
                    last_time_ms = int(last_time)
                    if last_time_ms >= current_end:
                        break
                    fetch_start = last_time_ms + 1

                    if len(month_batch) < 1000:
                        break

                current_start = current_end + 1

            fecha_hoy_utc = datetime.now(pytz.utc).strftime("%d/%m/%Y")
            pnl_neto = float(pnl_bruto) - float(comisiones)

            # Formato institucional sin emojis ni exclamaciones.
            mensaje = (
                "*REPORTE DE RENDIMIENTO ALINEADO HISTORICO*\n"
                f"Periodo: Origen del Sistema - {fecha_hoy_utc}\n"
                "Par Operativo: BTCUSDT\n"
                f"• Volumen Total de Trades: {total_trades}\n"
                f"• PNL Bruto Acumulado: {pnl_bruto:,.4f} USDT\n"
                f"• Costos de Friccion (Comisiones): -{abs(comisiones):,.4f} USDT\n"
                f"• RETORNO NETO DE CAPITAL: {pnl_neto:,.4f} USDT"
            )

            enviar_telegram(mensaje, proxies=requests_params.get("proxies"))
        except BinanceAPIException as e:
            logger.warning(f"[HISTORICO] No se pudo generar reporte histórico total: {e}", exc_info=True)
        except Exception as e:
            logger.warning(f"[HISTORICO] Error inesperado generando reporte histórico total: {e}", exc_info=True)

    # Ejecución única al arrancar en Render.
    enviar_reporte_historico_total()
    
    logger.info(f"=== Inicializando bot para {args.symbol} ===")
    
    try:
        logger.info(f"[STARTUP] Purgando órdenes huérfanas en Binance para {args.symbol}...")
        client.futures_cancel_all_open_orders(symbol=args.symbol)
        logger.info("[STARTUP] Purga automática completada de manera exitosa.")
    except BinanceAPIException as e:
        logger.warning(f"[STARTUP] No se pudieron purgar las órdenes en el arranque: {e}")
    
    # Ajustes de margen/apalancamiento: best-effort.
    # Si fallan (p.ej. APIError -2015 por credenciales/permiso), NO rompemos el contenedor.
    try:
        logger.info(f"[STARTUP] Ajustando Margin Type a ISOLATED y Leverage a {cfg.leverage}x...")

        try:
            # Bloque completo dentro del mismo try para que cualquier APIError no se escape.
            logger.info(
                f"[STARTUP] Cancelando órdenes abiertas antes de marginType ISOLATED ({args.symbol})..."
            )
            client.futures_cancel_all_open_orders(symbol=args.symbol)

            try:
                client.futures_change_margin_type(
                    symbol=args.symbol,
                    marginType="ISOLATED",
                )
            except BinanceAPIException as e:
                code = getattr(e, "code", None)
                if code is None and getattr(e, "args", None):
                    code = e.args[0] if len(e.args) > 0 else None

                # Binance devuelve -4046 cuando el tipo ya está configurado.
                if code == -4046 or "No need to change margin type" in str(e):
                    logger.warning(
                        "[STARTUP] El tipo de margen ya estaba en ISOLATED. Continuando configuración de leverage..."
                    )
                else:
                    raise

            client.futures_change_leverage(symbol=args.symbol, leverage=cfg.leverage)
            
            try:
                # False significa "One-Way Mode" (Modo Unidireccional)
                client.futures_change_position_mode(dualSidePosition="False")
                print("[STARTUP] Modo de posición configurado a Unidireccional.")
            except Exception as e:
                if "No need to change position side" in str(e) or "-4059" in str(e):
                    print("[STARTUP] El modo de posición ya era Unidireccional. Continuando...")
                else:
                    print(f"[WARNING] No se pudo cambiar el modo de posición: {e}")
        except BinanceAPIException as e:
            code = getattr(e, "code", None)
            if code is None and getattr(e, "args", None):
                code = e.args[0] if len(e.args) > 0 else None

            if code == -4067:
                logger.warning(
                    "Advertencia: el tipo de margen ya está configurado o existen posiciones/órdenes abiertas. Omitiendo configuración inicial."
                )
            else:
                logger.warning(
                    "WARNING: No se pudo cambiar el margen o apalancamiento, saltando configuración..."
                )
                logger.warning(f"Detalle BinanceAPIException: {e}", exc_info=True)

                # Diagnóstico read-only para saber si la API key sirve para lecturas.
                try:
                    diag_price = client.futures_symbol_ticker(symbol=args.symbol).get("price")
                    diag_balances = client.futures_account_balance()
                    diag_usdt = None
                    for bal in diag_balances:
                        if str(bal.get("asset", "")).upper() == "USDT":
                            diag_usdt = bal.get("availableBalance") or bal.get("available_balance")
                            break
                    logger.warning(
                        f"[DIAG] ticker({args.symbol}).price={diag_price} | USDT.available={diag_usdt}"
                    )
                except Exception as diag_e:
                    logger.warning(
                        f"[DIAG] No se pudo completar diagnóstico read-only: {diag_e}",
                        exc_info=True,
                    )
        except Exception as e:
            logger.warning(
                "WARNING: No se pudo cambiar el margen o apalancamiento, saltando configuración..."
            )
            logger.warning(f"Detalle Exception: {e}", exc_info=True)
    except Exception as e:
        logger.error(
            f"[STARTUP] Error inesperado durante configuración de margen/apalancamiento: {e}",
            exc_info=True,
        )
    info = client.futures_exchange_info()
    symbol_info = next(item for item in info['symbols'] if item['symbol'] == args.symbol)
    logger.info("Service is live.")
    
    # Reporte periódico a Telegram cada 2 horas (sin interferir con el loop de 60s).
    report_interval_s = 2 * 60 * 60
    next_report_ts = time.time() + report_interval_s

    def enviar_reporte_estado_2h(
        *,
        position_block: str,
        regime: str,
        current_close: float,
        current_rsi: float,
        volume_ok: bool,
        total_usdt: float | None,
        available_usdt: float | None,
    ) -> None:
        total_usdt_str = f"{total_usdt:.6f}" if total_usdt is not None else "N/A"
        available_usdt_str = f"{available_usdt:.6f}" if available_usdt is not None else "N/A"

        msg = (
            "Estado de la Cuenta: \n"
            f"Balance total: {total_usdt_str} USDT\n"
            f"Margen Disponible: {available_usdt_str} USDT\n\n"
            "Estado de la Posición: \n"
            f"{position_block}\n\n"
            "Métricas del Mercado: \n"
            f"Régimen: {regime}\n"
            f"Close actual: {current_close:.6f}\n"
            f"RSI actual: {current_rsi:.2f}\n\n"
            "Filtro de Volumen: \n"
            f"volume_ok={volume_ok}"
        )

        enviar_telegram(msg, proxies=requests_params.get("proxies"))
    
    def send_telegram_alert(mensaje: str) -> None:
        enviar_telegram(mensaje, proxies=requests_params.get("proxies"))

    def ejecutar_auditoria_historica() -> None:
        # Historial (últimos 1000 trades) sin restricciones de días.
        trades = client.futures_account_trades(symbol="BTCUSDT", limit=1000)

        total_operaciones_cerradas = 0
        ganancias_acumuladas = 0.0
        perdidas_acumuladas = 0.0
        balance_neto_total = 0.0

        for t in trades or []:
            try:
                t_time_ms = int(t.get("time", 0) or 0)
            except Exception:
                t_time_ms = 0

            if t_time_ms < bot_start_ts_ms:
                continue

            try:
                pnl = float(t.get("realizedPnl", 0.0) or 0.0)
            except Exception:
                pnl = 0.0

            if pnl == 0.0:
                continue

            total_operaciones_cerradas += 1
            balance_neto_total += pnl
            if pnl >= 0:
                ganancias_acumuladas += pnl
            else:
                perdidas_acumuladas += pnl

        mensaje_auditoria = (
            "*Bot Futuros: Balance Historico Total*\n"
            "Desde el inicio de operaciones hasta hoy\n"
            f"Total operaciones cerradas: {total_operaciones_cerradas}\n"
            f"Ganancias acumuladas: {ganancias_acumuladas:+.2f} USDT\n"
            f"Perdidas acumuladas: {perdidas_acumuladas:.2f} USDT\n"
            f"Balance neto total: {balance_neto_total:+.2f} USDT"
        )

        send_telegram_alert(mensaje_auditoria)

    if os.environ.get("AUDIT_HISTORICA") == "1":
        print("[AUDITORIA] Ejecutando auditoría histórica bajo demanda...")
        ejecutar_auditoria_historica()

    had_position = False
    checked_orders_for_position = False
    has_reduce_sl_tp_cached = True

    # Persistencia entre ciclos para detectar cierres reales (TP/SL) de forma inmediata.
    # Importante: debe actualizarse incluso en ramas con `continue`, por eso se gestiona
    # inmediatamente después de leer positionAmt.
    prev_position_amt = 0.0
    prev_position_amt_signed = 0.0

    # Se aplica una sola vez por posición para evitar modificaciones repetitivas.
    break_even_applied = False

    # Gate de drawdown diario: si se excede el límite, se evita abrir nuevas operaciones
    # hasta el próximo día (UTC).
    daily_drawdown_paused = False
    
    def send_daily_pnl_report() -> None:
        # Reporte para "ayer completo" en hora local del servidor.
        zona_local = pytz.timezone('America/Argentina/Buenos_Aires')
        now_local = datetime.now(zona_local)
        yesterday_date = (now_local - timedelta(days=1)).date()
        start_dt = datetime(
            yesterday_date.year,
            yesterday_date.month,
            yesterday_date.day,
            0,
            0,
            0,
            tzinfo=now_local.tzinfo,
        )
        end_dt = datetime(
            yesterday_date.year,
            yesterday_date.month,
            yesterday_date.day,
            23,
            59,
            59,
            tzinfo=now_local.tzinfo,
        )

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        # Consumimos todos los trades del rango usando paginación por timestamp.
        all_trades: list[dict] = []
        fetch_start_ms = start_ms
        while True:
            batch = client.futures_account_trades(
                symbol="BTCUSDT",
                startTime=fetch_start_ms,
                endTime=end_ms,
                limit=1000,
            )
            if not batch:
                break

            all_trades.extend(batch)

            last_time = batch[-1].get("time")
            if last_time is None:
                break

            last_time_ms = int(last_time)
            if last_time_ms >= end_ms:
                break

            # Siguiente página: avanzar 1ms para no repetir el último trade.
            fetch_start_ms = last_time_ms + 1

            # Si el batch ya vino "corto", no hay más páginas.
            if len(batch) < 1000:
                break

        realized_pnls: list[float] = []
        for t in all_trades:
            try:
                rp = float(t.get("realizedPnl", 0.0) or 0.0)
            except Exception:
                rp = 0.0
            if rp != 0.0:
                realized_pnls.append(rp)

        operaciones_cerradas = len(realized_pnls)
        ganancias_brutas = sum(p for p in realized_pnls if p > 0)
        perdidas_brutas = sum(p for p in realized_pnls if p < 0)
        net_pnl = ganancias_brutas + perdidas_brutas

        # PnL Mensual: desde el día 1 del mes actual (p.ej. 01/08/2026) hasta HOY.
        # Se calcula 100% en vivo consultando trades y filtrando por timestamp.
        month_start_dt = datetime(
            now_local.year,
            now_local.month,
            1,
            0,
            0,
            0,
            tzinfo=now_local.tzinfo,
        )
        month_start_ms = int(month_start_dt.timestamp() * 1000)
        now_ms = int(now_local.timestamp() * 1000)

        month_trades: list[dict] = []
        # Binance Futures impone un máximo de ventana temporal por request.
        # Para evitar APIError(code=-4165) "Maximum time interval is 7 days",
        # limitamos el arranque como máximo a los últimos 7 días.
        seven_days_ago_ms = int((time.time() - (7 * 24 * 60 * 60)) * 1000)
        fetch_month_start_ms = max(month_start_ms, seven_days_ago_ms)
        while True:
            month_batch = client.futures_account_trades(
                symbol="BTCUSDT",
                startTime=fetch_month_start_ms,
                endTime=now_ms,
                limit=1000,
            )
            if not month_batch:
                break

            month_trades.extend(month_batch)

            last_time = month_batch[-1].get("time")
            if last_time is None:
                break

            last_time_ms = int(last_time)
            if last_time_ms >= now_ms:
                break

            fetch_month_start_ms = last_time_ms + 1

            if len(month_batch) < 1000:
                break

        # HARD RESET (hard recompute): reiniciar PnL mensual desde 0.00
        monthly_net_pnl = 0.0
        for t in month_trades or []:
            t_time_ms = int(t.get("time", 0) or 0)
            if t_time_ms < month_start_ms or t_time_ms > now_ms:
                continue

            realized_pnl = float(t.get("realizedPnl", 0.0) or 0.0)
            commission = float(t.get("commission", 0.0) or 0.0)
            # PnL neto = realizedPnl - commission (Binance Futuros)
            net_pnl_trade = realized_pnl - commission
            monthly_net_pnl += net_pnl_trade

        fecha_ayer_str = yesterday_date.strftime("%d/%m/%Y")

        mensaje_reporte_diario = (
            "Bot Futuros: Reporte Diario de Rendimiento\n"
            f"Periodo analizado: {fecha_ayer_str}\n"
            f"Operaciones cerradas: {operaciones_cerradas}\n"
            f"Ganancias brutas: {ganancias_brutas:+.2f} USDT\n"
            f"Perdidas brutas: {perdidas_brutas:.2f} USDT\n"
            f"Resultado neto: {net_pnl:+.2f} USDT\n"
            f"PNL Mensual (Mes actual): {monthly_net_pnl:+.2f} USDT"
        )

        send_telegram_alert(mensaje_reporte_diario)

    # Intervalo mecánico fijo
    while True:
        try:
            # Validación temprana anti-Rlimit: primero consultamos posición activa.
            pos_info = client.futures_position_information(symbol=args.symbol)
            btc_pos = pos_info[0] if pos_info else {}
            raw_position_amt = float(btc_pos.get("positionAmt", "0") or 0.0)
            current_position_amt = abs(raw_position_amt)
            # Tolerancia estricta anti-dust/posiciones fantasma
            position_amt = 0.0 if current_position_amt <= 0.0001 else raw_position_amt
            position_amt = get_effective_position_amt(
                client,
                args.symbol,
                position_amt=position_amt,
                lookback_seconds=180,
            )

            valor_posicion = abs(float(position_amt))

            # --------------- Detección de cierre (persistente) ---------------
            current_position_amt = abs(float(position_amt))
            if prev_position_amt > 0.0001 and current_position_amt < 0.0001:
                pnl_realizado = 0.0
                precio_salida = None
                cantidad_total = 0.0
                comision_total = 0.0
                direccion_cerrada = "LONG" if prev_position_amt_signed > 0 else "SHORT"

                try:
                    # Dar tiempo a Binance para reflejar realizedPnl/price del cierre
                    time.sleep(2)
                    trades = client.futures_account_trades(symbol="BTCUSDT", limit=5)
                    ultimo_trade = trades[0] if trades else None
                    if ultimo_trade:
                        pnl_realizado = float(ultimo_trade.get("realizedPnl", 0.0) or 0.0)
                        # price suele representar el precio de ejecución del último fill/transaction
                        precio_salida = ultimo_trade.get("price") or ultimo_trade.get("avgPrice") or ultimo_trade.get("avg_price")
                        try:
                            precio_salida = float(precio_salida) if precio_salida is not None else None
                        except Exception:
                            precio_salida = None
                        try:
                            cantidad_total = float(
                                ultimo_trade.get("qty")
                                or ultimo_trade.get("quantity")
                                or 0.0
                            )
                        except Exception:
                            cantidad_total = 0.0
                        try:
                            comision_total_raw = float(ultimo_trade.get("commission", 0.0) or 0.0)
                            comision_total = abs(comision_total_raw)
                        except Exception:
                            comision_total = 0.0
                except Exception:
                    pnl_realizado = 0.0
                    precio_salida = None

                status_text = "Cierre de posicion"
                lado_salida = direccion_cerrada
                precio_ejecucion = float(precio_salida) if precio_salida is not None else 0.0
                resultado_neto = float(pnl_realizado) - float(comision_total)

                mensaje_cierre = (
                    "*REPORTE DE CIERRE DE POSICION*\n"
                    f"Estado: {status_text}\n"
                    f"• Direccion: {lado_salida}\n"
                    f"• Volumen: {float(cantidad_total):.3f} BTC\n"
                    f"• Precio de Ejecucion: {precio_ejecucion:,.2f} USDT\n"
                    f"• PNL Bruto: {float(pnl_realizado):+.4f} USDT\n"
                    f"• Comisiones: -{float(comision_total):.4f} USDT\n"
                    f"• Balance Neto: {float(resultado_neto):+.4f} USDT"
                )
                send_telegram_alert(mensaje_cierre)

                # Reset de estado interno tras cierre para evitar bucles/ruido.
                had_position = False
                checked_orders_for_position = None
                has_reduce_sl_tp_cached = 0
                break_even_applied = False

            # Actualizamos estado para el próximo ciclo (crucial incluso si hacemos continue).
            prev_position_amt = current_position_amt
            prev_position_amt_signed = float(position_amt)
            # ---------------------------------------------------------------
            if valor_posicion > 0.0005:
                logger.info(f"[LIVE] POSICION ya activa (positionAmt={position_amt}). No abro una nueva.")
                # Break-even protection (monitor intermedio sin tocar indicadores/régimen)
                if not break_even_applied:
                    try:
                        entry_price_val = None
                        try:
                            entry_price_val = pos_info[0].get("entryPrice") or pos_info[0].get("entry_price")
                            entry_price_val = float(entry_price_val) if entry_price_val is not None else None
                        except Exception:
                            entry_price_val = None

                        if entry_price_val is not None:
                            # Precio actual vía ticker (evita recomputar indicadores).
                            ticker = client.futures_symbol_ticker(symbol=args.symbol)
                            current_px = float(ticker.get("price"))

                            break_even_trigger_pct = 0.0030  # +0.30% a favor
                            break_even_offset = entry_price_val * 0.0007
                            stop_px = round(float(entry_price_val + break_even_offset), 1)  # tick BTCUSDT=0.10

                            should_move = (
                                (position_amt > 0 and current_px >= entry_price_val * (1 + break_even_trigger_pct))
                                or (position_amt < 0 and current_px <= entry_price_val * (1 - break_even_trigger_pct))
                            )

                            if should_move:
                                logger.info(
                                    "[BREAKEVEN] Precio objetivo alcanzado, moviendo Stop Loss a Break-Even..."
                                    f" entry={entry_price_val} current={current_px} stop_px={stop_px}"
                                )

                                # Cancelamos el STOP reduceOnly existente (STOP/STOP_MARKET).
                                try:
                                    open_orders = client.futures_get_open_orders(symbol=args.symbol)
                                except BinanceAPIException as e_open:
                                    open_orders = []
                                    logger.warning(f"[BREAKEVEN] No se pudieron listar open_orders: {e_open}", exc_info=True)

                                for o in (open_orders or []):
                                    try:
                                        if o.get("reduceOnly") and o.get("type") in ("STOP", "STOP_MARKET", "STOP_LIMIT"):
                                            oid = o.get("orderId")
                                            if oid is not None:
                                                try:
                                                    client.futures_cancel_order(symbol=args.symbol, orderId=oid)
                                                    logger.info(f"[BREAKEVEN] Stop anterior cancelado (orderId={oid}).")
                                                except BinanceAPIException as e_cancel:
                                                    logger.warning(
                                                        f"[BREAKEVEN] Falló cancel de stop (orderId={oid}): {e_cancel}",
                                                        exc_info=True,
                                                    )
                                    except Exception:
                                        pass

                                # Colocamos un nuevo STOP_MARKET en Break-Even.
                                side_close = "SELL" if position_amt > 0 else "BUY"
                                qty_abs = abs(float(position_amt))
                                try:
                                    client.futures_create_order(
                                        symbol=args.symbol,
                                        side=side_close,
                                        type="STOP_MARKET",
                                        quantity=qty_abs,
                                        stopPrice=str(stop_px),
                                        reduceOnly=True,
                                    )
                                    break_even_applied = True
                                    logger.info(f"[BREAKEVEN] Nuevo STOP_MARKET colocado en stop_px={stop_px}.")
                                except BinanceAPIException as e_mod:
                                    logger.warning(f"[BREAKEVEN] Falló colocar STOP_MARKET: {e_mod}", exc_info=True)
                    except BinanceAPIException as e_be:
                        logger.warning(f"[BREAKEVEN] Error Binance en break-even monitor: {e_be}", exc_info=True)

                time.sleep(15)
                continue

            # 1) Descargar klines recientes
            klines = client.futures_klines(
                symbol=args.symbol,
                interval=args.interval,
                limit=300,
            )

            df = pd.DataFrame(
                klines,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )

            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)

            # 2) Indicadores
            df_ind = prepare_indicators(df, cfg)
            if df_ind.empty:
                logger.warning("[LIVE] prepare_indicators devolvió DataFrame vacío. Saltando ciclo.")
                time.sleep(15)
                continue

            # 3) Régimen
            regime = detect_regime(df_ind, cfg)
            last_row = df_ind.iloc[-1]
            current_close = float(last_row["close"])
            current_rsi = float(last_row["rsi"])
            current_volume = float(last_row["volume"])
            vol_avg20 = float(last_row["vol_avg20"])
            vol_avg5 = float(last_row["vol_avg5"])

            is_alcista = regime == "ALCISTA"
            is_bajista = regime == "BAJISTA"
            is_lateral = regime == "LATERAL"
            vol_ref = vol_avg5 if is_lateral else vol_avg20
            volume_ok = np.isfinite(vol_ref) and current_volume >= (vol_ref * 0.5)
            print(
                f"[LIVE] Revisando mercado real... "
                f"Régimen detectado: ALCISTA ({is_alcista}) / BAJISTA ({is_bajista}) / LATERAL ({is_lateral}) | "
                f"VolumenOK={volume_ok} | Vol={current_volume:.6f} | VolAvg20={vol_avg20:.6f} | "
                f"Close={current_close:.6f} | RSI={current_rsi:.2f}",
                flush=True,
            )

            # 4) Verificar posición activa
            pos_info = client.futures_position_information(symbol=args.symbol)
            # Obtener el valor bruto de la API de Binance con tolerancia estricta.
            btc_pos = pos_info[0] if pos_info else {}
            raw_position_amt = float(btc_pos.get("positionAmt", "0") or 0.0)
            # Tolerancia anti-dust: <= 0.0001 BTC se considera CERO.
            position_amt = 0.0 if abs(raw_position_amt) <= 0.0001 else raw_position_amt
            position_amt = get_effective_position_amt(
                client,
                args.symbol,
                position_amt=position_amt,
                lookback_seconds=180,
            )

            # Reporte a Telegram cada 2 horas (exacto por timestamp, con re-sincronización).
            now_ts = time.time()
            # --- MODIFICAR EN EL CÓDIGO PRINCIPAL SIN ALTERAR LA POSICIÓN ---
            zona_local = pytz.timezone('America/Argentina/Buenos_Aires')
            hora_actual_local = datetime.now(zona_local)

            # Bandera de control para evitar bucles repetidos del reporte en el mismo minuto
            if not 'reporte_hoy_enviado' in locals():
                reporte_hoy_enviado = False

            # Restablecer la bandera si cambia de día
            if hora_actual_local.hour == 0 and hora_actual_local.minute == 0:
                reporte_hoy_enviado = False

            # CONDICIÓN DE DISPARO INICIAL (Para recibir el reporte de ayer AHORA MISMO al reiniciar)
            if not 'reporte_inicial_forzado' in locals():
                print("[REPORTE] Forzando reporte diario inicial de lectura...")
                send_daily_pnl_report()
                reporte_inicial_forzado = True

            # Disparo diario automático por horario local de mi país
            if hora_actual_local.hour == 6 and hora_actual_local.minute == 25 and not reporte_hoy_enviado:
                print("[REPORTE] Hora local detectada (06:25 AM). Enviando balance diario...")
                send_daily_pnl_report()
                reporte_hoy_enviado = True
            # ---------------------------------------------------------------
            if now_ts >= next_report_ts:
                try:
                    if abs(position_amt) > 0:
                        side = "Long" if position_amt > 0 else "Short"
                        size_btc = abs(position_amt)
                        entry_price_val = None
                        try:
                            entry_price_val = pos_info[0].get("entryPrice") or pos_info[0].get("entry_price")
                            entry_price_val = float(entry_price_val) if entry_price_val is not None else None
                        except Exception:
                            entry_price_val = None
                        if entry_price_val is not None:
                            position_block = f"{side} | positionAmt={size_btc:.6f} | entryPrice={entry_price_val:.6f}"
                        else:
                            position_block = f"{side} | positionAmt={size_btc:.6f}"
                    else:
                        position_block = "Sin posiciones activas"

                    # Estado de la cuenta (solo lectura) en el reporte.
                    total_usdt = None
                    available_usdt = None
                    try:
                        diag_balances = client.futures_account_balance()
                        for bal in diag_balances:
                            if str(bal.get("asset", "")).upper() == "USDT":
                                total_raw = bal.get("balance") or bal.get("walletBalance") or bal.get("totalWalletBalance")
                                avail_raw = bal.get("availableBalance") or bal.get("available_balance")
                                total_usdt = float(total_raw) if total_raw is not None else None
                                available_usdt = float(avail_raw) if avail_raw is not None else None
                                break
                    except Exception as diag_e:
                        logger.warning(f"[TELEGRAM] No se pudo obtener balances USDT para el reporte: {diag_e}", exc_info=True)

                    enviar_reporte_estado_2h(
                        position_block=position_block,
                        regime=regime,
                        current_close=current_close,
                        current_rsi=current_rsi,
                        volume_ok=volume_ok,
                        total_usdt=total_usdt,
                        available_usdt=available_usdt,
                    )
                except Exception as e:
                    logger.warning(f"[TELEGRAM] Error enviando reporte de estado: {e}", exc_info=True)
                # Avanzar en múltiplos de 2h hasta quedar en el futuro.
                while next_report_ts <= now_ts:
                    next_report_ts += report_interval_s

            # Solo entrar si no hay posición (mecánico: 1 posición a la vez)
            if abs(position_amt) > 0:
                logger.info(f"[LIVE] Posición ya activa (positionAmt={position_amt}). No abro una nueva.")
                # Salvavidas: si por reinicio/no-ejecución Binance no dejó SL/TP,
                # cerramos por MARKET cuando se alcance TP o se rompa SL.
                if not checked_orders_for_position:
                    try:
                        open_orders = client.futures_get_open_orders(symbol=args.symbol)
                    except Exception:
                        open_orders = []

                    has_reduce_sl_tp_cached = False
                    for o in (open_orders or []):
                        if o.get("reduceOnly") and o.get("type") in ("STOP_MARKET", "LIMIT"):
                            has_reduce_sl_tp_cached = True
                            break

                    checked_orders_for_position = True

                if not has_reduce_sl_tp_cached:
                    entry_price_val = None
                    try:
                        entry_price_val = pos_info[0].get("entryPrice") or pos_info[0].get("entry_price")
                        entry_price_val = float(entry_price_val) if entry_price_val is not None else None
                    except Exception:
                        entry_price_val = None

                    if entry_price_val is not None:
                        if position_amt > 0:
                            # LONG: TP = entry*(1+tp), SL = entry*(1-sl)
                            # Salida coherente con la posición LONG: usar objetivos bullish (tendencia),
                            # evitando cualquier uso de TP/SL lateral en esta emergencia.
                            tp_pct = cfg.tp_bullish_pct
                            sl_pct = cfg.sl_bullish_pct

                            tp_hit = current_close >= (entry_price_val * (1 + tp_pct))
                            sl_hit = current_close <= (entry_price_val * (1 - sl_pct))

                            if tp_hit or sl_hit:
                                logger.warning(
                                    f"[EMERGENCIA] LONG close inmediato por TP/SL. "
                                    f"entry={entry_price_val} close={current_close} tp_pct={tp_pct} sl_pct={sl_pct}"
                                )
                                try:
                                    client.futures_create_order(
                                        symbol=args.symbol,
                                        side="SELL",
                                        type="MARKET",
                                        quantity=abs(position_amt),
                                        reduceOnly=True,
                                    )
                                except Exception as e:
                                    logger.warning(f"[EMERGENCIA] Falló cierre LONG: {e}", exc_info=True)
                        else:
                            # SHORT: TP = entry*(1-tp), SL = entry*(1+sl)
                            tp_pct = cfg.tp_bearish_pct
                            sl_pct = cfg.sl_bearish_pct

                            tp_hit = current_close <= (entry_price_val * (1 - tp_pct))
                            sl_hit = current_close >= (entry_price_val * (1 + sl_pct))

                            if tp_hit or sl_hit:
                                logger.warning(
                                    f"[EMERGENCIA] SHORT close inmediato por TP/SL. "
                                    f"entry={entry_price_val} close={current_close} tp_pct={tp_pct} sl_pct={sl_pct}"
                                )
                                try:
                                    client.futures_create_order(
                                        symbol=args.symbol,
                                        side="BUY",
                                        type="MARKET",
                                        quantity=abs(position_amt),
                                        reduceOnly=True,
                                    )
                                except Exception as e:
                                    logger.warning(f"[EMERGENCIA] Falló cierre SHORT: {e}", exc_info=True)
            else:
                # Evitar órdenes huérfanas: cancelamos las existentes antes de abrir
                # --------------- DRAWdown diario UTC (anti-rachas negativas) ---------------
                # Se evalúa una vez por día UTC antes de evaluar cualquier entrada nueva.
                now_utc = datetime.now(pytz.utc)
                today_utc = now_utc.date()

                if cfg.last_pnl_check_date != today_utc:
                    cfg.last_pnl_check_date = today_utc
                    daily_drawdown_paused = False

                    # Calculamos PnL neto del día: sum(realizedPnl) - sum(commission)
                    day_start_dt = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                    start_ms = int(day_start_dt.timestamp() * 1000)
                    end_ms = int(now_utc.timestamp() * 1000)

                    daily_net_pnl = 0.0
                    fetch_start_ms = start_ms
                    try:
                        while True:
                            batch = client.futures_account_trades(
                                symbol="BTCUSDT",
                                startTime=fetch_start_ms,
                                endTime=end_ms,
                                limit=1000,
                            )
                            if not batch:
                                break

                            for t in batch:
                                rp = float(t.get("realizedPnl", 0.0) or 0.0)
                                comm = float(t.get("commission", 0.0) or 0.0)
                                daily_net_pnl += rp - comm

                            last_time = batch[-1].get("time")
                            if last_time is None:
                                break
                            last_time_ms = int(last_time)

                            if last_time_ms >= end_ms:
                                break

                            fetch_start_ms = last_time_ms + 1

                            if len(batch) < 1000:
                                break
                    except BinanceAPIException as e_pnl:
                        logger.warning(f"[DRAWDOWN] No se pudo auditar PnL diario: {e_pnl}", exc_info=True)

                    if daily_net_pnl < (-float(cfg.max_daily_loss_usd)):
                        logger.critical("CRÍTICO: Límite de Drawdown Diario Alcanzado. Bot en pausa hasta mañana.")
                        enviar_telegram(
                            "*ALERTA DE SISTEMA: LIMITE DE DRAWDOWN ALCANZADO*\n"
                            "Detalle: Las perdidas acumuladas del dia excedieron el maximo parametrizado.\n"
                            "Accion: Detencion preventiva de busqueda de entradas hasta proximo ciclo diario.",
                            proxies=requests_params.get("proxies"),
                        )
                        daily_drawdown_paused = True

                if daily_drawdown_paused:
                    # Congela aperturas nuevas el resto del día UTC
                    time.sleep(15)
                    continue

                client.futures_cancel_all_open_orders(symbol=args.symbol)

                # 5) Reglas de entrada
                if is_lateral:
                    # LATERAL deshabilitado: no abrir operaciones bajo ninguna circunstancia.
                    pass
                elif is_alcista and volume_ok and current_rsi <= cfg.bullish_rsi_entry:
                    sl_pct = cfg.sl_bullish_pct
                    tp_pct = cfg.tp_bullish_pct
                    # Position sizing dinámico (usa totalMarginBalance y cfg.risk_fraction)
                    qty = calculate_position_size(
                        client=client,
                        symbol=args.symbol,
                        entry_price=current_close,
                        sl_pct=sl_pct,
                        cfg=cfg,
                    )
                    qty = round(float(qty), 3)  # evitar rejections por decimales excesivos
                    logger.info(f"[ENTRY] ALCISTA->LONG qty={qty} sl_pct={sl_pct} tp_pct={tp_pct}")
                    if qty > 0:
                        _place_long_with_stop(
                            client,
                            args.symbol,
                            qty,
                            current_close,
                            sl_pct,
                            tp_pct,
                            symbol_info,
                            proxies=requests_params.get("proxies"),
                        )
                elif is_bajista and volume_ok and current_rsi >= cfg.bearish_rsi_entry:
                    sl_pct = cfg.sl_bearish_pct
                    tp_pct = cfg.tp_bearish_pct
                    qty = calculate_position_size(
                        client=client,
                        symbol=args.symbol,
                        entry_price=current_close,
                        sl_pct=sl_pct,
                        cfg=cfg,
                    )
                    qty = round(float(qty), 3)  # evitar rejections por rejections de decimales
                    logger.info(f"[ENTRY] BAJISTA->SHORT qty={qty} sl_pct={sl_pct} tp_pct={tp_pct}")
                    if qty > 0:
                        _place_short_with_sl_tp(
                            client,
                            args.symbol,
                            qty,
                            current_close,
                            sl_pct,
                            tp_pct,
                            symbol_info,
                            proxies=requests_params.get("proxies"),
                        )

            time.sleep(15)
        except KeyboardInterrupt:
            logger.info("Bot detenido por el usuario.")
            break
        except Exception as e:
            logger.error(f"Error en el bucle principal: {e}", exc_info=True)
            time.sleep(10)
if __name__ == '__main__':
    main()
