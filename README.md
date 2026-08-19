# Bot Futuros — Configuración por variables de entorno

Este proyecto permite configurar los parámetros de trading mediante variables de entorno (env vars). Así podés cambiar el comportamiento del bot desde el panel de Render sin tocar el código.

Variables principales (nombre — tipo — valor por defecto)
- LEVERAGE — int — 5
- MAX_MARGIN_PER_TRADE_PCT — float — 0.30
- RISK_FRACTION — float — 0.03
- TP_BULLISH_PCT — float — 0.0140
- SL_BULLISH_PCT — float — 0.0060
- TP_BEARISH_PCT — float — 0.0140
- SL_BEARISH_PCT — float — 0.0060
- BULLISH_RSI_ENTRY — float — 62.0
- BEARISH_RSI_ENTRY — float — 45.0
- EMA_LENGTH — int — 200
- RSI_LENGTH — int — 14
- BB_LENGTH — int — 20
- BB_STD_MULT — float — 2.0
- MAX_DAILY_LOSS_USD — float — 25.0
- FEE_FRICTION — float — 0.0007

Cómo editarlas en Render (paso a paso)
1. Entrá a tu dashboard de Render y seleccioná el servicio donde corre el bot.  
2. En la sección "Environment" (o "Environment Variables") hacé click en "Add Environment Variable".  
3. Escribí el nombre de la variable (ej. `LEVERAGE`) y su valor (ej. `10`) y guardá.  
4. Repetí para todas las variables que quieras cambiar.  
5. Reiniciá el servicio (deploy) para que el cambio se aplique.

Ejemplo rápido
- Para cambiar apalancamiento a 10: crea `LEVERAGE = 10`.  
- Para bajar el TP alcista a 1%: crea `TP_BULLISH_PCT = 0.01`.

Notas y buenas prácticas
- Los valores se leen al iniciar el bot — cualquier cambio requiere redeploy.  
- Usá valores numéricos sin símbolos (ej. `0.01` para 1%).  
- Probá cambios en una cuenta de prueba antes de aplicarlos en real.  
- Si olvidás una variable, el bot usa el valor por defecto codificado en `bot_futuros.py`.

Si querés, puedo preparar un archivo de ejemplo `.env` o escribir instrucciones específicas para el panel de Render según tu servicio (nombre del servicio).  

