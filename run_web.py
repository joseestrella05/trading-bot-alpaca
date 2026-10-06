"""Launcher script for the Alpaca Swing Trading Bot Web Dashboard.

Runs the FastAPI application under Uvicorn ASGI server.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

import argparse
import sys
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="Alpaca Swing Trading Bot - Web Dashboard Runner")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("  ALPACA QUANTITATIVE SWING TRADING BOT - WEB DASHBOARD")
    print("=" * 65)
    print(f"  Interface Web URL : http://{args.host}:{args.port}")
    print(f"  Local Browser Link: http://localhost:{args.port}")
    print(f"  Scheduler Policy  : Mon-Fri at 3:50 PM EST (NYSE Pre-Close)")
    print("=" * 65)
    print("  Presiona Ctrl+C para detener el servidor.\n")

    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
