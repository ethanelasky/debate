"""FastAPI app factory for the config viewer.

The two config directories are parameters (tests point them at tmp-dir
fixtures); the defaults are the repo's real config families. The viewer reads
and writes ONLY yaml files inside these two directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[1]


def create_app(
    configs_dir: Optional[str | Path] = None,
    prompts_dir: Optional[str | Path] = None,
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
    app.mount("/static", StaticFiles(directory=str(_PKG_DIR / "static")), name="static")
    app.include_router(prompts_router)
    app.include_router(experiments_router)

    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse(url="/prompts")

    return app
