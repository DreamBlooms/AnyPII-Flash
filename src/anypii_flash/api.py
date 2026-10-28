from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .cache import ResultCache
from .config import Settings
from .decider import DecideClassifier, Decider
from .errors import AnyPiiError, ModelUnavailableError
from .extractor import Extractor, GlinerExtractor
from .models import AnyPiiRequest, AnyPiiResponse
from .service import process_request

log = logging.getLogger(__name__)


def _error(message: str, error_type: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": error_type}}


def create_app(
    settings: Settings | None = None,
    extractor: Extractor | None = None,
    decider: Decider | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        if app.state.extractor is None:
            model = GlinerExtractor(settings)
            model.load()
            if settings.warmup:
                model.warmup()
            app.state.extractor = model
        app.state.ready = True
        yield

    app = FastAPI(
        title="AnyPII Flash",
        version=__version__,
        description="Schema-driven PII redaction with GLiNER2.5.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.extractor = extractor
    app.state.decider = decider or DecideClassifier(settings)
    app.state.cache = ResultCache(settings.result_cache_entries, settings.result_cache_bytes)
    app.state.ready = False
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    def require_api_key(authorization: str | None = Header(default=None)) -> None:
        if settings.api_key and authorization != f"Bearer {settings.api_key}":
            raise AnyPiiError(
                "missing or invalid API key",
                status_code=401,
                error_type="authentication_error",
            )

    @app.exception_handler(AnyPiiError)
    async def anypii_error_handler(request: Request, exc: AnyPiiError):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error(exc.message, exc.error_type),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {"msg": "invalid request"}
        location = ".".join(str(part) for part in first.get("loc", ()))
        message = first.get("msg", "invalid request")
        if location:
            message = f"{location}: {message}"
        return JSONResponse(status_code=400, content=_error(message))

    @app.exception_handler(ModelUnavailableError)
    async def model_error_handler(request: Request, exc: ModelUnavailableError):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error(exc.message, exc.error_type),
        )

    @app.middleware("http")
    async def payload_limit(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
            except ValueError:
                length = 0
            if length > settings.max_payload_bytes:
                return JSONResponse(
                    status_code=413,
                    content=_error(
                        f"request body is too large (max {settings.max_payload_bytes} bytes)"
                    ),
                )
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict:
        current = app.state.extractor
        return {
            "ok": bool(app.state.ready),
            "status": "healthy" if app.state.ready else "loading",
            "ready": bool(app.state.ready),
            "model": current.model_name if current else settings.model_label,
            "backend": current.backend_name if current else settings.backend_label,
            "architecture": "boundary",
            "plus_model": settings.plus_model_id,
            "plus_ready": bool(getattr(app.state.decider, "ready", False)),
            "version": __version__,
        }

    @app.get("/v1/models")
    async def models() -> dict:
        current = app.state.extractor
        return {
            "data": [
                {
                    "id": current.model_name if current else settings.model_label,
                    "object": "model",
                    "owned_by": "anypii-flash",
                }
            ]
        }

    def handle(request: AnyPiiRequest) -> AnyPiiResponse:
        return process_request(
            request,
            app.state.extractor,
            app.state.decider,
            app.state.cache,
            settings,
        )

    @app.post("/v1/anypii", response_model=AnyPiiResponse, response_model_exclude_none=True)
    def anypii(
        request: AnyPiiRequest,
        _: None = Depends(require_api_key),
    ) -> AnyPiiResponse:
        return handle(request)

    @app.post("/v1/flash", response_model=AnyPiiResponse, response_model_exclude_none=True)
    def flash(
        request: AnyPiiRequest,
        _: None = Depends(require_api_key),
    ) -> AnyPiiResponse:
        return handle(request)

    return app
