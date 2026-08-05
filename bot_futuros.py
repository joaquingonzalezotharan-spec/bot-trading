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
    
    lateral_rsi_entry = 32.0
    lateral_rsi_exit = 70.0
    sl_lateral_pct = 0.0030          
    tp_lateral_pct = 0.0035          
    
    bullish_rsi_entry = 65.0
    sl_bullish_pct = 0.0025          
    tp_bullish_pct = 0.0050          
    
    bearish_rsi_entry = 55.0
    bearish_rsi_exit = 35.0
    sl_bearish_pct = 0.0040          
    tp_bearish_pct = 0.0080          
    
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
        logger.info(f"[ORDEN] Abriendo posición LONG en Market. Cantidad: {qty}")
        market_order = client.futures_create_order(
            symbol=symbol,
            side="BUY",
            type="MARKET",
            quantity=qty,
        )
        exec_price = market_order.get("avgPrice") or market_order.get("avg_price") or entry_price
        exec_price = float(exec_price)

        # IMPORTANTE: esperamos a que Binance refleje la posición antes de
        # enviar órdenes reduceOnly (evita APIError "Reduce-only order failed").
        confirmed_pos_amt = 0.0
        for _ in range(10):
            try:
                pos_info = client.futures_position_information(symbol=symbol)
                pos_amt = float(pos_info[0]["positionAmt"]) if pos_info else 0.0
                if pos_amt > 0:
                    confirmed_pos_amt = pos_amt
                    break
            except Exception:
                pass
            time.sleep(0.2)
        
        if confirmed_pos_amt <= 0:
            logger.warning("[ORDEN] No se confirmó positionAmt LONG antes de SL/TP; omitiendo notificación y órdenes reduceOnly.")
            return
        
        take_profit_price = round_price(entry_price * (1 + tp_pct), symbol_info)
        stop_loss_price = round_price(entry_price * (1 - sl_pct), symbol_info)
        
        sl_order = client.futures_create_order(
            symbol=symbol, side="SELL", type="STOP_MARKET",
            stopPrice=stop_loss_price, reduceOnly=True, quantity=qty
        )
        sl_trigger_price = sl_order.get("stopPrice") or sl_order.get("stop_price") or stop_loss_price
        sl_trigger_price = float(sl_trigger_price)
        logger.info(f"[ORDEN] SL colocado en (STOP_MARKET): {sl_trigger_price}")
        
        tp_order = client.futures_create_order(
            symbol=symbol, side="SELL", type="LIMIT",
            price=take_profit_price, timeInForce="GTC", reduceOnly=True, quantity=qty
        )
        tp_trigger_price = tp_order.get("price") or take_profit_price
        tp_trigger_price = float(tp_trigger_price)
        logger.info(f"[ORDEN] TP colocado en (LIMIT Maker GTC): {tp_trigger_price}")

        # Notificación visual inmediata desde el móvil.
        msg = (
            "🚀 ¡OPERACIÓN ABIERTA Y BLINDADA!\n"
            "• Tipo: LONG\n"
            f"• Precio Entrada: {exec_price:.6f}\n"
            f"• Tamaño: {confirmed_pos_amt}\n"
            f"• 🎯 Take Profit (Nativo): {tp_trigger_price:.6f}\n"
            f"• 🛑 Stop Loss (Nativo): {sl_trigger_price:.6f}\n"
            "Nota: Asegúrate de extraer los precios de disparo reales devueltos por la respuesta de la API de Binance "
            "para garantizar que el mensaje muestre los valores exactos que quedaron guardados en el libro de órdenes"
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
        logger.info(f"[ORDEN] Abriendo posición SHORT en Market. Cantidad: {qty}")
        market_order = client.futures_create_order(
            symbol=symbol,
            side="SELL",
            type="MARKET",
            quantity=qty,
        )
        exec_price = market_order.get("avgPrice") or market_order.get("avg_price") or entry_price
        exec_price = float(exec_price)

        # IMPORTANTE: esperamos a que Binance refleje la posición antes de
        # enviar órdenes reduceOnly (evita APIError "Reduce-only order failed").
        confirmed_pos_amt = 0.0
        for _ in range(10):
            try:
                pos_info = client.futures_position_information(symbol=symbol)
                pos_amt = float(pos_info[0]["positionAmt"]) if pos_info else 0.0
                if pos_amt < 0:
                    confirmed_pos_amt = pos_amt
                    break
            except Exception:
                pass
            time.sleep(0.2)
        
        if confirmed_pos_amt >= 0:
            logger.warning("[ORDEN] No se confirmó positionAmt SHORT antes de SL/TP; omitiendo notificación y órdenes reduceOnly.")
            return
        
        take_profit_price = round_price(entry_price * (1 - tp_pct), symbol_info)
        stop_loss_price = round_price(entry_price * (1 + sl_pct), symbol_info)
        
        sl_order = client.futures_create_order(
            symbol=symbol, side="BUY", type="STOP_MARKET",
            stopPrice=stop_loss_price, reduceOnly=True, quantity=qty
        )
        sl_trigger_price = sl_order.get("stopPrice") or sl_order.get("stop_price") or stop_loss_price
        sl_trigger_price = float(sl_trigger_price)
        logger.info(f"[ORDEN] SL colocado en (STOP_MARKET): {sl_trigger_price}")
        
        tp_order = client.futures_create_order(
            symbol=symbol, side="BUY", type="LIMIT",
            price=take_profit_price, timeInForce="GTC", reduceOnly=True, quantity=qty
        )
        tp_trigger_price = tp_order.get("price") or take_profit_price
        tp_trigger_price = float(tp_trigger_price)
        logger.info(f"[ORDEN] TP colocado en (LIMIT Maker GTC): {tp_trigger_price}")

        # Notificación visual inmediata desde el móvil.
        msg = (
            "🚀 ¡OPERACIÓN ABIERTA Y BLINDADA!\n"
            "• Tipo: SHORT\n"
            f"• Precio Entrada: {exec_price:.6f}\n"
            f"• Tamaño: {confirmed_pos_amt}\n"
            f"• 🎯 Take Profit (Nativo): {tp_trigger_price:.6f}\n"
            f"• 🛑 Stop Loss (Nativo): {sl_trigger_price:.6f}\n"
            "Nota: Asegúrate de extraer los precios de disparo reales devueltos por la respuesta de la API de Binance "
            "para garantizar que el mensaje muestre los valores exactos que quedaron guardados en el libro de órdenes"
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
                    "⚠️ El tipo de margen ya está configurado o existen posiciones/órdenes abiertas. Omitiendo configuración inicial..."
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
    logger.info("===> Your service is live 🚀")
    
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

        emoji_resultado = "🟢" if balance_neto_total >= 0 else "🔴"
        mensaje_auditoria = (
            "🏆 *Bot Futuros: Balance Histórico Total*\n"
            "🚀 *Desde el inicio de operaciones hasta hoy*\n"
            f"🔄 *Total operaciones cerradas:* {total_operaciones_cerradas}\n"
            f"💰 *Ganancias acumuladas:* {ganancias_acumuladas:+.2f} USDT\n"
            f"💸 *Pérdidas acumuladas:* {perdidas_acumuladas:.2f} USDT\n"
            f"⚖️ *BALANCE NETO TOTAL:* {emoji_resultado} {balance_neto_total:+.2f} USDT"
        )

        send_telegram_alert(mensaje_auditoria)

    if os.environ.get("AUDIT_HISTORICA") == "1":
        print("[AUDITORIA] Ejecutando auditoría histórica bajo demanda...")
        ejecutar_auditoria_historica()

    had_position = False
    checked_orders_for_position = False
    has_reduce_sl_tp_cached = True
    
    def send_daily_pnl_report() -> None:
        # Reporte para "ayer completo" en hora local del servidor.
        now_local = datetime.now().astimezone()
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

        # PnL mensual (mes en curso) - Opción B
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
        fetch_month_start_ms = month_start_ms
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
            try:
                t_time_ms = int(t.get("time", 0) or 0)
                if t_time_ms < month_start_ms or t_time_ms > now_ms:
                    continue
                # PnL neto = realizedPnl - commission (Binance Futuros)
                net_pnl_trade = float(t["realizedPnl"]) - float(t["commission"])
                monthly_net_pnl += net_pnl_trade
            except Exception:
                continue

        emoji_resultado = "🟢" if net_pnl >= 0 else "🔴"
        fecha_ayer_str = yesterday_date.strftime("%d/%m/%Y")

        mensaje_reporte_diario = (
            "📊 *Bot Futuros: Reporte Diario de Rendimiento*\n"
            f"📆 *Período analizado:* {fecha_ayer_str}\n"
            f"🔄 *Operaciones cerradas:* {operaciones_cerradas}\n"
            f"🟢 *Ganancias brutas:* {ganancias_brutas:+.2f} USDT\n"
            f"🔴 *Pérdidas brutas:* {perdidas_brutas:.2f} USDT\n"
            f"🎚️ *Resultado Neto:* {emoji_resultado} {net_pnl:+.2f} USDT\n"
            f"📅 *PNL Mensual (Mes actual):* {monthly_net_pnl:+.2f} USDT"
        )

        send_telegram_alert(mensaje_reporte_diario)

    # Intervalo mecánico fijo
    while True:
        try:
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

            is_alcista = regime == "ALCISTA"
            is_bajista = regime == "BAJISTA"
            is_lateral = regime == "LATERAL"
            volume_ok = np.isfinite(vol_avg20) and current_volume >= (vol_avg20 * 0.5)
            print(
                f"[LIVE] Revisando mercado real... "
                f"Régimen detectado: ALCISTA ({is_alcista}) / BAJISTA ({is_bajista}) / LATERAL ({is_lateral}) | "
                f"VolumenOK={volume_ok} | Vol={current_volume:.6f} | VolAvg20={vol_avg20:.6f} | "
                f"Close={current_close:.6f} | RSI={current_rsi:.2f}",
                flush=True,
            )

            # 4) Verificar posición activa
            pos_info = client.futures_position_information(symbol=args.symbol)
            # Obtener el valor bruto de la API de Binance
            raw_amt = float(pos_info[0]["positionAmt"]) if pos_info else 0.0
            # Aplicar umbral de seguridad: si es menor a 0.001 BTC, forzar a 0.0
            position_amt = raw_amt if abs(raw_amt) >= 0.001 else 0.0
            position_amt = get_effective_position_amt(
                client,
                args.symbol,
                position_amt=position_amt,
                lookback_seconds=180,
            )

            # Alerta de cierre con PNL realizado (cuando pasamos de posición activa a posición=0)
            if abs(position_amt) > 0:
                had_position = True
            elif had_position and abs(position_amt) == 0:
                pnl_realizado = 0.0  # Extrae aquí el PNL del último trade cerrado de Binance
                try:
                    time.sleep(2)
                    trades = client.futures_account_trades(symbol="BTCUSDT", limit=5)
                    ultimo_trade = trades[0] if trades else None
                    if ultimo_trade:
                        pnl_realizado = float(ultimo_trade["realizedPnl"])
                except Exception:
                    pnl_realizado = 0.0

                emoji_resultado = "🟢" if pnl_realizado >= 0 else "🔴"
                signo = "+" if pnl_realizado >= 0 else ""
                mensaje_cierre = (
                    f"🏁 *Bot Futuros: Posición Cerrada*\n\n"
                    f"📊 *Resultado:* {emoji_resultado} Net PNL: {signo}{pnl_realizado:.2f} USDT\n"
                    f"🔒 *Estado de Cuenta:* Limpia y en cero, escaneando el mercado cada 15 segundos..."
                )
                send_telegram_alert(mensaje_cierre)
                had_position = False
                checked_orders_for_position = False
                has_reduce_sl_tp_cached = True

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
                            if regime == "ALCISTA":
                                tp_pct = cfg.tp_bullish_pct
                                sl_pct = cfg.sl_bullish_pct
                            else:
                                tp_pct = cfg.tp_lateral_pct
                                sl_pct = cfg.sl_lateral_pct

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
                client.futures_cancel_all_open_orders(symbol=args.symbol)

                # 5) Reglas de entrada
                if is_lateral and volume_ok and current_rsi <= cfg.lateral_rsi_entry:
                    sl_pct = cfg.sl_lateral_pct
                    tp_pct = cfg.tp_lateral_pct
                    qty = 0.016
                    logger.info(f"[ENTRY] LATERAL->LONG qty={qty} sl_pct={sl_pct} tp_pct={tp_pct}")
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
                elif is_alcista and volume_ok and current_rsi <= cfg.bullish_rsi_entry:
                    sl_pct = cfg.sl_bullish_pct
                    tp_pct = cfg.tp_bullish_pct
                    qty = 0.016
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
                    qty = 0.016
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
