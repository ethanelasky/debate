"""FastAPI app factory for the config viewer.

The two config directories are parameters (tests point them at tmp-dir
fixtures); the defaults are the repo's real config families. The viewer reads
and writes ONLY yaml files inside these two directories.

This is a read+WRITE API with no authentication, so it also validates the Host
header: without that, any page the user visits could point a hostname it
controls at 127.0.0.1 (DNS rebinding) and become same-origin with the viewer.
Only loopback names — plus whatever host the server was explicitly bound to —
are accepted; anything else is a 400 before a route ever runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[1]

#: Host names that are always accepted (any port). IPv6 literals are compared
#: unbracketed — host_name() strips the brackets a Host header carries.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def host_name(header: str) -> str:
    """The host part of a Host header, lowercased, port and brackets stripped.

    Three shapes to tell apart: a bracketed IPv6 literal (`[::1]:8080`, port
    after the bracket), a bare IPv6 literal (`::1`, which cannot carry a port
    at all), and everything else (`localhost:8080`).
    """
    header = header.strip().lower()
    if header.startswith("["):
        end = header.find("]")
        return header[1:end] if end != -1 else header[1:]
    if header.count(":") > 1:
        return header  # bare IPv6 literal: no port to strip
    return header.split(":", 1)[0]


def is_loopback_host(host: str) -> bool:
    """Whether `--host <host>` binds to loopback only."""
    return host_name(host) in LOOPBACK_HOSTS


def create_app(
    configs_dir: Optional[str | Path] = None,
    prompts_dir: Optional[str | Path] = None,
    allowed_hosts: Optional[Iterable[str]] = None,
) -> FastAPI:
    from tools.viewer.routes_experiments import router as experiments_router
    from tools.viewer.routes_prompts import router as prompts_router

    app = FastAPI(title="Config Viewer", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.configs_dir = Path(configs_dir) if configs_dir else _REPO_ROOT / "configs"
    app.state.prompts_dir = (
        Path(prompts_dir)
        if prompts_dir
        else _REPO_ROOT / "infra" / "envs" / "debate" / "prompt_configs"
    )
    app.state.templates = Jinja2Templates(directory=str(_PKG_DIR / "templates"))

    allowed = LOOPBACK_HOSTS | {host_name(h) for h in (allowed_hosts or ())}
    app.state.allowed_hosts = allowed

    @app.middleware("http")
    async def reject_foreign_hosts(request, call_next):
        # A missing Host header is rejected too: HTTP/1.1 requires one, and an
        # absent one cannot be checked against the allowlist.
        if host_name(request.headers.get("host", "")) not in allowed:
            return JSONResponse(
                status_code=400,
                content={"detail": "invalid Host header — the viewer only serves loopback hosts"},
            )
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=str(_PKG_DIR / "static")), name="static")
    app.include_router(prompts_router)
    app.include_router(experiments_router)

    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse(url="/prompts")

    return app
