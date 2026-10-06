#!/usr/bin/env bash
# ==============================================================================
# Script de inicio rápido para el Dashboard Web del Trading Bot con Alpaca
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Activar entorno virtual si existe
if [ -d "$DIR/venv" ]; then
    source "$DIR/venv/bin/activate"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "=================================================================="
echo " Iniciando Alpaca Swing Trading Bot Dashboard en http://$HOST:$PORT"
echo "=================================================================="

exec python "$DIR/run_web.py" --host "$HOST" --port "$PORT" "$@"
