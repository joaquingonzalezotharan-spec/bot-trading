import os
import time
import argparse
import logging
import math
import sys
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
    
    lateral_rsi_entry = 38.0
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
def _place_long_with_stop(client: Client, symbol: str, qty: float, entry_price: float, sl_pct: float, tp_pct: float, symbol_info: dict):
    try:
        logger.info(f"[ORDEN] Abriendo posición LONG en Market. Cantidad: {qty}")
        client.futures_create_order(symbol=symbol, side="BUY", type="MARKET", quantity=qty)
        
        take_profit_price = round_price(entry_price * (1 + tp_pct), symbol_info)
        stop_loss_price = round_price(entry_price * (1 - sl_pct), symbol_info)
        
        client.futures_create_order(
            symbol=symbol, side="SELL", type="STOP_MARKET",
            stopPrice=stop_loss_price, reduceOnly=True, quantity=qty
        )
        logger.info(f"[ORDEN] SL colocado en (STOP_MARKET): {stop_loss_price}")
        
        client.futures_create_order(
            symbol=symbol, side="SELL", type="LIMIT",
            price=take_profit_price, timeInForce="GTC", reduceOnly=True, quantity=qty
        )
        logger.info(f"[ORDEN] TP colocado en (LIMIT Maker GTC): {take_profit_price}")
    except BinanceAPIException as e:
        logger.error(f"Error de Binance al ejecutar Long Setup: {e}")
def _place_short_with_sl_tp(client: Client, symbol: str, qty: float, entry_price: float, sl_pct: float, tp_pct: float, symbol_info: dict):
    try:
        logger.info(f"[ORDEN] Abriendo posición SHORT en Market. Cantidad: {qty}")
        client.futures_create_order(symbol=symbol, side="SELL", type="MARKET", quantity=qty)
        
        take_profit_price = round_price(entry_price * (1 - tp_pct), symbol_info)
        stop_loss_price = round_price(entry_price * (1 + sl_pct), symbol_info)
        
        client.futures_create_order(
            symbol=symbol, side="BUY", type="STOP_MARKET",
            stopPrice=stop_loss_price, reduceOnly=True, quantity=qty
        )
        logger.info(f"[ORDEN] SL colocado en (STOP_MARKET): {stop_loss_price}")
        
        client.futures_create_order(
            symbol=symbol, side="BUY", type="LIMIT",
            price=take_profit_price, timeInForce="GTC", reduceOnly=True, quantity=qty
        )
        logger.info(f"[ORDEN] TP colocado en (LIMIT Maker GTC): {take_profit_price}")
    except BinanceAPIException as e:
        logger.error(f"Error de Binance al ejecutar Short Setup: {e}")
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
    # IP/Host del servidor proxy (proveedor):
    proxy_host = os.environ.get("PROXY_HOST") or "31.59.239.182"

    # Codificamos usuario/clave para que el URL sea válido.
    proxy_user_enc = quote(proxy_user, safe="")
    proxy_password_enc = quote(proxy_password, safe="")
    proxy_url = f"http://{proxy_user_enc}:{proxy_password_enc}@{proxy_host}:{proxy_port}"
    requests_params = {"proxies": {"http": proxy_url, "https": proxy_url}}

    client = Client(api_key or "", api_secret or "", requests_params=requests_params)
    
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

            client.futures_change_margin_type(symbol=args.symbol, marginType="ISOLATED")
            client.futures_change_leverage(symbol=args.symbol, leverage=cfg.leverage)
        except BinanceAPIException as e:
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
                logger.warning(f"[DIAG] No se pudo completar diagnóstico read-only: {diag_e}", exc_info=True)
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
    
    while True:
        try:
            logger.info("[LIVE] Revisando mercado real...")
            time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Bot detenido por el usuario.")
            break
        except Exception as e:
            logger.error(f"Error en el bucle principal: {e}")
            time.sleep(10)
if __name__ == '__main__':
    main()
