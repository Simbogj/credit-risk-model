"""
Bati Bank Credit Risk Intelligence Platform
=============================================
Launcher script to run the FastAPI model backend and serve the Banking Enterprise React frontend.

Usage:
    python run_react_dashboard.py [--port 8000] [--host 127.0.0.1]
"""

import argparse
import sys
import webbrowser
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "api"))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run Credit Risk React Platform")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    print("=" * 70)
    print("  BATI BANK CREDIT RISK INTELLIGENCE PLATFORM (REACT VIEW)")
    print("=" * 70)
    print(f"  ● React Frontend & REST API running at: {url}")
    print(f"  ● Interactive Underwriting Terminal:   {url}/dashboard")
    print(f"  ● Interactive API Swagger Docs:        {url}/docs")
    print("=" * 70)
    print("  Press Ctrl+C to terminate the server.\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run("main:app", host=args.host, port=args.port, app_dir=str(PROJECT_ROOT / "src" / "api"))


if __name__ == "__main__":
    main()
