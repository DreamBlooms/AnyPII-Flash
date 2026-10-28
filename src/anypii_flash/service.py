from __future__ import annotations

import json
import re
import time
from dataclasses import replace

from .cache import ResultCache
from .config import Settings
from .decider import GLOBAL_KEEP, GLOBAL_REDACT, Decider, DecisionItem
from .errors import AnyPiiError, UnsupportedFeatureError
from .extractor import Extractor, TypeSpec
from .masking import apply_mask
from .models import AnyPiiRequest, AnyPiiResponse, FlashUsage, PlusUsage, Usage
from .spans import group_spans, normalize_spans

CONTEXT_RADIUS = 40


def _type_signature(spec: TypeSpec) -> set[str]:
    text = (spec.id + " " + spec.description).lower()
    signature = {token for token in re.findall(r"[a-z0-9_]{2,}", text)}
    cjk = "".join(ch for ch in spec.description if "\u3400" <= ch <= "\u9fff")
    for index in range(len(cjk) - 1):
        signature.add(cjk[index : index + 2])
    return signature


def _candidate_ids(type_id: str, specs: list[TypeSpec]) -> tuple[str, ...]:
    """Keep the extracted type plus types with overlapping descriptions/ids."""
    by_id = {spec.id: spec for spec in specs}
    original = by_id[type_id]
    signature = _type_signature(original)
    out = [type_id]
    for spec in specs:
        if spec.id == type_id:
            continue
        if signature & _type_signature(spec):
            out.append(spec.id)
    return tuple(out[:5])


def _unique_items(
    hits,
    descriptions: dict[str, str],
    text: str,
    *,
    with_candidates: bool,
    specs: list[TypeSpec],
) -> tuple[list[DecisionItem], dict[int, list[int]]]:
    """Deduplicate repeated (type, text) spans before batching."""
    items: list[DecisionItem] = []
    groups: dict[int, list[int]] = {}
    representative: dict[tuple[str, str], int] = {}
    for index, hit in enumerate(hits):
        if not hit.auto:
            continue
        key = (hit.type_id, hit.text)
        if key not in representative:
            representative[key] = index
            groups[index] = []
            items.append(
                DecisionItem(
                    index=index,
                    type_id=hit.type_id,
                    description=descriptions.get(hit.type_id, hit.type_id),
                    text=hit.text,
                    context=text[
                        max(0, hit.start - CONTEXT_RADIUS) : min(
                            len(text), hit.end + CONTEXT_RADIUS
                        )
                    ],
                    candidate_ids=(_candidate_ids(hit.type_id, specs) if with_candidates else ()),
                )
            )
        groups[representative[key]].append(index)
    return items, groups


def _cache_key(
    request: AnyPiiRequest,
    model: str,
    decider_model: str,
    policy_threshold: float,
    type_refine_threshold: float,
) -> str:
    payload = {
        "cache_version": "anypii-flash-v2",
        "model": model,
        "decider_model": decider_model,
        "text": request.state.text,
        "policy": request.state.policy or "",
        "threshold": request.threshold,
        "redact_threshold": request.redact_threshold,
        "type_refine_threshold": (
            request.type_refine_threshold
            if request.type_refine_threshold is not None
            else type_refine_threshold
        ),
        "policy_threshold": policy_threshold,
        "mask_style": request.mask_style,
        "mode": request.mode,
        "type_refine": request.type_refine,
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


def _validate(request: AnyPiiRequest, settings: Settings) -> tuple[list[TypeSpec], bool]:
    if request.mode not in ("flash", "flash-plus"):
        raise UnsupportedFeatureError(
            f'mode "{request.mode}" is not supported; use "flash" or "flash-plus"'
        )
    if request.parrot_typed:
        raise UnsupportedFeatureError("parrot_typed is not supported by AnyPII Flash")
    if not 0.0 <= request.threshold <= 1.0:
        raise AnyPiiError("threshold must be in [0, 1]")
    redact_threshold = request.redact_threshold
    if redact_threshold is not None and not 0.0 <= redact_threshold <= 1.0:
        raise AnyPiiError("redact_threshold must be in [0, 1]")
    type_refine_threshold = request.type_refine_threshold
    if type_refine_threshold is not None and not 0.0 <= type_refine_threshold <= 1.0:
        raise AnyPiiError("type_refine_threshold must be in [0, 1]")
    if len(request.state.text) > settings.max_text_cp:
        raise AnyPiiError(f"state.text is too long (max {settings.max_text_cp} code points)")
    if len(request.types) > settings.max_types:
        raise AnyPiiError(f"at most {settings.max_types} types are supported")

    specs: list[TypeSpec] = []
    has_auto = False
    for type_id, definition in request.types.items():
        if not type_id:
            raise AnyPiiError("type ids must be non-empty")
        auto = definition.redact is None or definition.redact == "auto"
        if request.mode == "flash" and auto:
            raise UnsupportedFeatureError(
                f"types.{type_id}.redact is required in flash mode; "
                'use "flash-plus" for auto policy decisions'
            )
        if auto:
            has_auto = True
            redact: bool | None = None
        else:
            redact = bool(definition.redact)
        specs.append(
            TypeSpec(
                id=type_id,
                description=definition.description,
                redact=redact,
            )
        )
    if has_auto and not (request.state.policy or "").strip():
        raise UnsupportedFeatureError("flash-plus auto redaction requires a non-empty state.policy")
    return specs, has_auto


def process_request(
    request: AnyPiiRequest,
    extractor: Extractor,
    decider: Decider,
    cache: ResultCache,
    settings: Settings,
) -> AnyPiiResponse:
    started = time.perf_counter()
    specs, has_auto = _validate(request, settings)
    key = _cache_key(
        request,
        extractor.model_name,
        decider.model_name,
        settings.policy_threshold,
        settings.type_refine_threshold,
    )

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

    if all(spec.redact is False for spec in specs) and (
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

    plus_usage: PlusUsage | None = None
    type_refined = 0
    type_refine_forward_passes = 0
    type_refine_elapsed_ms = 0.0
    action_elapsed_ms = 0.0
    action_forward_passes = 0
    action = None
    action_global = False
    descriptions = {spec.id: spec.description for spec in specs}
    by_id = {spec.id: spec for spec in specs}
    should_refine = request.mode == "flash-plus" and request.type_refine is not False
    refine_threshold = (
        request.type_refine_threshold
        if request.type_refine_threshold is not None
        else settings.type_refine_threshold
    )
    threshold = (
        request.redact_threshold
        if request.redact_threshold is not None
        else settings.policy_threshold
    )

    initial_auto_hits = [hit for hit in hits if hit.auto]
    if initial_auto_hits and has_auto:
        action = decider.classify_action(
            request.state.text,
            request.state.policy or "",
        )
        action_elapsed_ms = action.elapsed_ms
        action_forward_passes = action.forward_passes
        action_global = (
            action.action in (GLOBAL_KEEP, GLOBAL_REDACT)
            and action.confidence >= settings.policy_action_threshold
        )

    if should_refine and not action_global:
        refine_items, refine_groups = _unique_items(
            hits,
            descriptions,
            request.state.text,
            with_candidates=True,
            specs=specs,
        )
        if refine_items:
            refinements = decider.refine_types(
                request.state.text,
                refine_items,
                specs,
                threshold=refine_threshold,
            )
            type_refine_elapsed_ms = refinements.elapsed_ms
            type_refine_forward_passes = refinements.forward_passes
            refined_hits = list(hits)
            for representative, (new_type, _) in refinements.types.items():
                if new_type not in by_id:
                    continue
                spec = by_id[new_type]
                for index in refine_groups.get(representative, ()):
                    hit = refined_hits[index]
                    if new_type == hit.type_id:
                        continue
                    refined_hits[index] = replace(
                        hit,
                        type_id=new_type,
                        auto=spec.redact is None,
                        redacted=bool(spec.redact),
                        redact_score=-1.0,
                    )
                    type_refined += 1
            hits = refined_hits

    if action_global:
        decided: list = []
        for hit in hits:
            if not hit.auto:
                decided.append(hit)
                continue
            score = 1.0 if action.action == GLOBAL_REDACT else 0.0
            decided.append(replace(hit, redacted=score >= threshold, redact_score=score))
        hits = decided
        plus_usage = PlusUsage(
            model=decider.model_name,
            backend=decider.backend_name,
            decisions=0,
            forward_passes=action_forward_passes,
            elapsed_ms=round(action_elapsed_ms, 3),
            policy_threshold=threshold,
            policy_action=action.action,
            action_confidence=round(action.confidence, 6),
            type_refined=0,
            type_refine_forward_passes=0,
            type_refine_threshold=refine_threshold,
            action_elapsed_ms=round(action_elapsed_ms, 3),
        )
    else:
        auto_hits = [hit for hit in hits if hit.auto]
        if auto_hits and has_auto:
            decision_items, decision_groups = _unique_items(
                hits,
                descriptions,
                request.state.text,
                with_candidates=False,
                specs=specs,
            )
            decisions = decider.decide(
                request.state.text,
                request.state.policy or "",
                decision_items,
                action=action,
            )
            index_to_representative = {
                index: representative
                for representative, members in decision_groups.items()
                for index in members
            }
            decided = []
            for index, hit in enumerate(hits):
                if not hit.auto:
                    decided.append(hit)
                    continue
                representative = index_to_representative.get(index, index)
                score = decisions.scores.get(representative, 0.0)
                decided.append(
                    replace(
                        hit,
                        redacted=score >= threshold,
                        redact_score=score,
                    )
                )
            hits = decided
            plus_usage = PlusUsage(
                model=decider.model_name,
                backend=decider.backend_name,
                decisions=len(decision_items),
                forward_passes=(
                    action_forward_passes + type_refine_forward_passes + decisions.forward_passes
                ),
                elapsed_ms=round(
                    action_elapsed_ms + type_refine_elapsed_ms + decisions.elapsed_ms,
                    3,
                ),
                policy_threshold=threshold,
                policy_action=action.action if action else None,
                action_confidence=round(action.confidence if action else 0.0, 6),
                type_refined=type_refined,
                type_refine_forward_passes=type_refine_forward_passes,
                type_refine_threshold=refine_threshold,
                action_elapsed_ms=round(action_elapsed_ms, 3),
                type_refine_elapsed_ms=round(type_refine_elapsed_ms, 3),
                decision_elapsed_ms=round(decisions.elapsed_ms, 3),
            )
        elif type_refined:
            plus_usage = PlusUsage(
                model=decider.model_name,
                backend=decider.backend_name,
                decisions=0,
                forward_passes=action_forward_passes + type_refine_forward_passes,
                elapsed_ms=round(action_elapsed_ms + type_refine_elapsed_ms, 3),
                policy_threshold=threshold,
                policy_action=action.action if action else None,
                action_confidence=round(action.confidence if action else 0.0, 6),
                type_refined=type_refined,
                type_refine_forward_passes=type_refine_forward_passes,
                type_refine_threshold=refine_threshold,
                action_elapsed_ms=round(action_elapsed_ms, 3),
                type_refine_elapsed_ms=round(type_refine_elapsed_ms, 3),
            )

    if request.skip_keep_types or request.omit_keep_spans:
        hits = [hit for hit in hits if hit.redacted]

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
            plus=plus_usage,
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
