# Alpaca Quantitative Swing Trading Bot (1D Trend Following)

Bot modular de Swing Trading para acciones y ETFs líquidos (ej. `SPY`, `QQQ`, `AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`) ejecutándose contra la API de **Alpaca Markets** (Paper / Live) con gestión de riesgo cuantitativa estricta y órdenes de tipo **Bracket Order**.

---

## 1. Arquitectura del Sistema

El proyecto sigue una arquitectura desacoplada por capas guiada por contratos de dominio fuertemente tipados:

```
trading_bot_alpaca/
├── .env                    # Credenciales de Alpaca y variables de entorno
├── requirements.txt        # Dependencias del proyecto (alpaca-py, pandas, etc.)
├── models.py               # Contratos de dominio (TradingSignal, PositionSize, AccountState, ExecutionResult)
├── config.py               # Dataclasses de configuración (.env, riesgo, estrategia, broker)
├── data_provider.py        # Descarga y estandarización de barras 1D (HistoricalDataClient)
├── strategy.py             # Lógica matemática (EMA 20/50/200, Wilder ATR 14, Wilder RSI 14, Pullback)
├── risk_manager.py         # Dimensionamiento de posición (1% riesgo), R:R 2:1 y Kill-Switch intradía (2.5%)
├── execution.py            # Orquestación con TradingClient de Alpaca y despacho de Bracket Orders
├── main.py                 # Runner CLI y bucle de escaneo de mercado (on-demand o continuo)
└── tests/                  # Suite completa de pruebas unitarias
    ├── test_strategy.py
    ├── test_risk_manager.py
    └── test_execution.py
```

---

## 2. Especificación Cuantitativa y Matemática

### A. Estrategia (Swing Trend Following en 1D)

1. **Filtro de Tendencia Macro**:
   $$\text{Precio} > \text{EMA}(200) \quad \land \quad \text{EMA}(50) > \text{EMA}(200)$$
   Asegura alineación con la tendencia alcista institucional.

2. **Condición de Retroceso (Pullback)**:
   - En las últimas 3 velas, el mínimo testeó la zona de soporte dinámico de la media corta:
     $$\min(\text{Low}_{t-2:t}) \le \text{EMA}(20) \times (1 + \text{tolerancia})$$

3. **Gatillo de Entrada (Price Action Breakout)**:
   - Ruptura del máximo de la vela previa:
     $$\text{Close}_t > \text{High}_{t-1}$$

4. **Momentum RSI**:
   - $\text{RSI}(14)$ rebotando desde la zona de soporte institucional de 40:
     $$\text{RSI}_t \ge 40.0 \quad \land \quad \text{RSI}_t > \text{RSI}_{t-1} \quad \land \quad \min(\text{RSI}_{t-3:t}) \le 45.0$$

---

### B. Gestión de Riesgo Estricta (Core)

1. **Riesgo Máximo por Operación**:
   $$\text{Riesgo en Dólares} = \text{Capital (Equity)} \times 0.01 \quad (1.0\%)$$

2. **Niveles de Salida (Bracket Order)**:
   - **Stop Loss**: $1.5 \times \text{ATR}(14)$ por debajo de la entrada:
     $$\text{Stop Loss Price} = \text{Entry Price} - (1.5 \times \text{ATR}_{14})$$
   - **Take Profit**: Ratio Riesgo-Beneficio $2:1$:
     $$\text{Take Profit Price} = \text{Entry Price} + 2.0 \times (\text{Entry Price} - \text{Stop Loss Price})$$

3. **Cálculo de Tamaño de Posición (Shares)**:
   $$\text{Shares} = \frac{\text{Capital} \times 0.01}{|\text{Entry Price} - \text{Stop Loss Price}|}$$
   > **Nota Arquitectónica sobre Alpaca API**: Las órdenes de tipo Bracket en Alpaca exigen números enteros de acciones (`fractional orders must be simple orders`). Por lo tanto, el sistema aplica redondeo hacia abajo (`math.floor(shares)`). Si la cuenta no puede asumir al menos 1 acción entera sin sobrepasar el 1.0% de riesgo máximo, la orden es rechazada preventivamente para proteger la cuenta.

4. **Kill-Switch Intradía (2.5%)**:
   $$\text{Drawdown Intradía} = \frac{\text{Last Equity} - \text{Equity}}{\text{Last Equity}}$$
   Si el drawdown intradía $\ge 2.5\%$, el Kill-Switch se activa automáticamente bloqueando la apertura de nuevas posiciones y emitiendo una alerta crítica.

5. **Protección contra Sobreconcentración y Límite de Buying Power**:
   - Límite de exposición máxima por activo: máximo 30% del portafolio en una sola acción.
   - Validación contra `buying_power` en tiempo real.

---

## 3. Uso y Comandos

### Ejecución de Pruebas Unitarias
```bash
./venv/bin/python -m unittest discover tests
```

### Escaneo a Demanda (Dry-Run / Simulado)
```bash
./venv/bin/python main.py --dry-run
```

### Escaneo a Demanda con Tickers Personalizados
```bash
./venv/bin/python main.py --dry-run --tickers SPY,AAPL,NVDA
```

### Ejecución Real contra Alpaca Paper Trading
```bash
./venv/bin/python main.py
```

### Modo Continuo (Escaneo cada hora o al cierre)
```bash
./venv/bin/python main.py --continuous --interval 3600
```

---

## 4. Interfaz Web & Dashboard en Tiempo Real (FastAPI)

El bot incluye una interfaz web moderna en **Dark Mode** (estilo Bloomberg/TradingView) con actualización en vivo y planificador en segundo plano con **APScheduler**.

### Iniciar el Dashboard Web
Con un solo comando puedes iniciar el servidor y acceder desde tu navegador a **`http://localhost:8000`**:

```bash
# Opción 1: Mediante el script bash
./run_dashboard.sh

# Opción 2: Directamente con python
./venv/bin/python run_web.py
```

### Características del Dashboard:
1. **Monitor de Cuenta**: Tarjetas con Equity en tiempo real, Efectivo, Buying Power, P&L intradía y estado del Kill-Switch.
2. **Posiciones Abiertas**: Visualización de activos en cartera con precio de entrada, precio actual, P&L no realizado y niveles activos de **Stop Loss** y **Take Profit**.
3. **Radar Cuantitativo de Mercado**: Tabla con los tickers analizados, precio 1D, RSI(14), EMAs (20, 50, 200), ATR(14), señales (BUY / HOLD) y diagnóstico matemático detallado.
4. **Control del Bot**: Botón **"Escanear Ahora"** para disparar un escaneo manual en tiempo real y switch de encendido/pausa (**ONLINE / PAUSED**).
5. **Terminal de Logs en Vivo**: Visor de logs con actualización automática cada 3 segundos y auto-scroll.
6. **Planificador en Segundo Plano (APScheduler)**:
   * Escaneo automático programado de **Lunes a Viernes a las 3:50 PM EST** (10 minutos antes del cierre de NYSE).

### Endpoints REST API:
* `GET /`: Dashboard web interactivo.
* `GET /api/status`: Estado del bot, balances de Alpaca, drawdown intradía y posiciones abiertas.
* `GET /api/signals`: Resultados técnicos del último escaneo.
* `POST /api/scan-now`: Ejecución manual e inmediata de escaneo de mercado.
* `POST /api/toggle-bot`: Alternar bot entre activo y pausado.
* `GET /api/logs`: Registros en memoria para la terminal embebida.
* `GET /api/jobs`: Tareas programadas y próxima hora de ejecución.

---

## 5. Variables de Entorno (`.env`)

| Variable | Descripción | Valor por Defecto |
| :--- | :--- | :--- |
| `ALPACA_API_KEY` | Clave API de Alpaca | *(Requerido)* |
| `ALPACA_SECRET_KEY` | Clave Secreta de Alpaca | *(Requerido)* |
| `ALPACA_PAPER` | Modo Paper Trading (`True` / `False`) | `True` |
| `ALPACA_DATA_FEED` | Feed de datos de mercado (`iex` o `sip`) | `iex` |
| `MAX_RISK_PER_TRADE_PCT` | Riesgo máximo por operación | `0.01` (1.0%) |
| `MAX_INTRADAY_DRAWDOWN_PCT` | Umbral Kill-Switch de Drawdown intradía | `0.025` (2.5%) |
| `RISK_REWARD_RATIO` | Ratio beneficio:riesgo | `2.0` (2:1) |
| `ATR_SL_MULTIPLIER` | Multiplicador de ATR para Stop Loss | `1.5` |
| `SYMBOLS` | Lista de tickers a escanear | `SPY,QQQ,AAPL,MSFT,NVDA,AMZN,GOOGL` |
| `SCHEDULER_INTERVAL_MINUTES` | Intervalo opcional de escaneo periódico en minutos | *(Opcional)* |
