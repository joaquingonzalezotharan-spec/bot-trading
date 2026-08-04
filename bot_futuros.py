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
# RIESGO FIJO POR OPERACIÓN (USD)
# =====================================================================
# Pérdida máxima fija estimada en USD por operación (bajo el SL configurado).
RISK_PER_TRADE = 2.5


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
                time.sleep(60)
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
            volume_ok = np.isfinite(vol_avg20) and current_volume > (vol_avg20 * 1.2)
            print(
                f"[LIVE] Revisando mercado real... "
                f"Régimen detectado: ALCISTA ({is_alcista}) / BAJISTA ({is_bajista}) / LATERAL ({is_lateral}) | "
                f"VolumenOK={volume_ok} | Vol={current_volume:.6f} | VolAvg20={vol_avg20:.6f} | "
                f"Close={current_close:.6f} | RSI={current_rsi:.2f}",
                flush=True,
            )

            # 4) Verificar posición activa
            pos_info = client.futures_position_information(symbol=args.symbol)
            position_amt = float(pos_info[0]["positionAmt"]) if pos_info else 0.0

            # Solo entrar si no hay posición (mecánico: 1 posición a la vez)
            if abs(position_amt) > 0:
                logger.info(f"[LIVE] Posición ya activa (positionAmt={position_amt}). No abro una nueva.")
                time.sleep(60)
                continue

            # Evitar órdenes huérfanas: cancelamos las existentes antes de abrir
            client.futures_cancel_all_open_orders(symbol=args.symbol)

            # 5) Reglas de entrada
            if is_lateral and volume_ok and current_rsi <= cfg.lateral_rsi_entry:
                sl_pct = cfg.sl_lateral_pct
                tp_pct = cfg.tp_lateral_pct
                qty = calculate_qty_fixed_risk(
                    client=client,
                    symbol=args.symbol,
                    entry_price=current_close,
                    sl_pct=sl_pct,
                    cfg=cfg,
                )
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
                    )
            elif is_alcista and volume_ok and current_rsi <= cfg.bullish_rsi_entry:
                sl_pct = cfg.sl_bullish_pct
                tp_pct = cfg.tp_bullish_pct
                qty = calculate_qty_fixed_risk(
                    client=client,
                    symbol=args.symbol,
                    entry_price=current_close,
                    sl_pct=sl_pct,
                    cfg=cfg,
                )
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
                    )
            elif is_bajista and volume_ok and current_rsi >= cfg.bearish_rsi_entry:
                sl_pct = cfg.sl_bearish_pct
                tp_pct = cfg.tp_bearish_pct
                qty = calculate_qty_fixed_risk(
                    client=client,
                    symbol=args.symbol,
                    entry_price=current_close,
                    sl_pct=sl_pct,
                    cfg=cfg,
                )
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
                    )

            time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Bot detenido por el usuario.")
            break
        except Exception as e:
            logger.error(f"Error en el bucle principal: {e}", exc_info=True)
            time.sleep(10)
if __name__ == '__main__':
    main()
