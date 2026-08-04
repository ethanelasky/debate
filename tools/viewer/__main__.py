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

    from tools.viewer.server import create_app, is_loopback_host

    if not is_loopback_host(args.host):
        print(
            f"\n*** WARNING: binding to {args.host} exposes an UNAUTHENTICATED "
            "read+write config editor to the network.\n"
            "*** Anyone who can reach this port can rewrite your prompt and "
            "experiment YAML. Use 127.0.0.1 unless you mean it.\n"
        )

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}/prompts"
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    # the bound host is same-origin by construction; every other Host is a 400
    uvicorn.run(create_app(allowed_hosts=[args.host]), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
