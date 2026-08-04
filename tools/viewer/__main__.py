"""python -m tools.viewer [--host 127.0.0.1] [--port 8080] [--no-browser]"""

from __future__ import annotations

import argparse
import threading
import webbrowser


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.viewer",
        description="Serve the prompt/experiment config viewer.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--no-browser", action="store_true", help="do not auto-open the browser"
    )
    args = parser.parse_args(argv)

    import uvicorn

    from tools.viewer.server import create_app

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}/prompts"
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
