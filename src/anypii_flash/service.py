from __future__ import annotations

import json
import time

from .cache import ResultCache
from .config import Settings
from .errors import AnyPiiError, UnsupportedFeatureError
from .extractor import Extractor, TypeSpec
from .masking import apply_mask
from .models import AnyPiiRequest, AnyPiiResponse, FlashUsage, Usage
from .spans import group_spans, normalize_spans


def _cache_key(request: AnyPiiRequest, model: str) -> str:
    payload = {
        "cache_version": "anypii-flash-v1",
        "model": model,
        "text": request.state.text,
        "policy": request.state.policy or "",
        "threshold": request.threshold,
        "mask_style": request.mask_style,
        "mode": request.mode,
        "skip_keep_types": request.skip_keep_types,
        "omit_keep_spans": request.omit_keep_spans,
        "types": [
            {
                "id": type_id,
                "description": definition.description,
                "redact": definition.redact,
            }
            for type_id, definition in request.types.items()
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate(request: AnyPiiRequest, settings: Settings) -> list[TypeSpec]:
    if request.mode != "flash":
        raise UnsupportedFeatureError(
            f'mode "{request.mode}" is not supported; AnyPII Flash only supports "flash"'
        )
    if request.parrot_typed:
        raise UnsupportedFeatureError("parrot_typed is not supported by AnyPII Flash")
    if not 0.0 <= request.threshold <= 1.0:
        raise AnyPiiError("threshold must be in [0, 1]")
    if len(request.state.text) > settings.max_text_cp:
        raise AnyPiiError(f"state.text is too long (max {settings.max_text_cp} code points)")
    if len(request.types) > settings.max_types:
        raise AnyPiiError(f"at most {settings.max_types} types are supported")

    specs: list[TypeSpec] = []
    for type_id, definition in request.types.items():
        if not type_id:
            raise AnyPiiError("type ids must be non-empty")
        if definition.redact is None:
            raise UnsupportedFeatureError(
                f"types.{type_id}.redact is required; "
                "explicit true/false is required in the Flash MVP"
            )
        if definition.redact == "auto":
            raise UnsupportedFeatureError(
                f'types.{type_id}.redact="auto" is not implemented yet; '
                "use explicit true/false until the Decide classifier is integrated"
            )
        specs.append(
            TypeSpec(
                id=type_id,
                description=definition.description,
                redact=bool(definition.redact),
            )
        )
    return specs


def process_request(
    request: AnyPiiRequest,
    extractor: Extractor,
    cache: ResultCache,
    settings: Settings,
) -> AnyPiiResponse:
    started = time.perf_counter()
    specs = _validate(request, settings)
    key = _cache_key(request, extractor.model_name)

    cached = cache.get(key)
    if cached is not None:
        cached.usage.flash.cache_status = "hit"
        cached.usage.flash.elapsed_ms = 0.0
        cached.usage.flash.chunks = 0
        cached.usage.flash.forward_passes = 0
        cached.usage.flash.cache_entries = len(cache)
        return cached

    if not request.state.text or not specs:
        response = _empty_response(request, extractor, settings)
        cache.put(key, response)
        response.usage.flash.cache_status = "miss"
        response.usage.flash.cache_entries = len(cache)
        return response

    if all(not spec.redact for spec in specs) and (
        request.skip_keep_types or request.omit_keep_spans
    ):
        response = _empty_response(request, extractor, settings)
        cache.put(key, response)
        response.usage.flash.cache_status = "miss"
        response.usage.flash.cache_entries = len(cache)
        return response

    extracted, stats = extractor.extract(request.state.text, specs, request.threshold)
    hits = normalize_spans(
        extracted,
        request.state.text,
        specs,
        request.threshold,
        skip_keep_types=request.skip_keep_types,
        omit_keep_spans=request.omit_keep_spans,
    )
    groups, group_index = group_spans(hits)
    redacted_text = apply_mask(request.state.text, hits, group_index, request.mask_style)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    response = AnyPiiResponse(
        redacted_text=redacted_text,
        spans=groups,
        model=extractor.model_name,
        backend=extractor.backend_name,
        usage=Usage(
            decodes=0,
            branches=0,
            generated_tokens=0,
            fallback=False,
            flash=FlashUsage(
                splitter=settings.splitter,
                chunks=stats.chunks,
                word_tokens=stats.word_tokens,
                forward_passes=stats.forward_passes,
                elapsed_ms=round(elapsed_ms, 3),
                cache_status="miss",
                cache_entries=0,
            ),
        ),
    )
    cache.put(key, response)
    response.usage.flash.cache_entries = len(cache)
    return response


def _empty_response(
    request: AnyPiiRequest, extractor: Extractor, settings: Settings
) -> AnyPiiResponse:
    return AnyPiiResponse(
        redacted_text=request.state.text,
        spans=[],
        model=extractor.model_name,
        backend=extractor.backend_name,
        usage=Usage(
            decodes=0,
            branches=0,
            generated_tokens=0,
            fallback=False,
            flash=FlashUsage(
                splitter=settings.splitter,
                chunks=0,
                word_tokens=0,
                forward_passes=0,
                elapsed_ms=0.0,
                cache_status="miss",
                cache_entries=0,
            ),
        ),
    )
