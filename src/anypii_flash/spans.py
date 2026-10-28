from __future__ import annotations

from dataclasses import dataclass

from .extractor import ExtractedSpan, TypeSpec
from .models import EntitySpan, Occurrence

_TRIM = set(".,;:!?，。；：！？、")


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and (text[start].isspace() or text[start] in _TRIM):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in _TRIM):
        end -= 1
    return start, end


@dataclass(frozen=True)
class SpanHit:
    start: int
    end: int
    text: str
    type_id: str
    score: float
    redacted: bool


def normalize_spans(
    extracted: list[ExtractedSpan],
    text: str,
    types: list[TypeSpec],
    threshold: float,
    *,
    skip_keep_types: bool = False,
    omit_keep_spans: bool = False,
) -> list[SpanHit]:
    """Trim, filter, deduplicate, and make spans non-overlapping."""
    by_id = {spec.id: spec for spec in types}
    order = {spec.id: index for index, spec in enumerate(types)}
    candidates: list[SpanHit] = []
    seen: set[tuple[int, int, str]] = set()

    for item in extracted:
        spec = by_id.get(item.type_id)
        if spec is None or item.score < threshold:
            continue
        if not (0 <= item.start < item.end <= len(text)):
            continue
        start, end = _trim_span(text, item.start, item.end)
        if start >= end:
            continue
        key = (start, end, item.type_id)
        if key in seen:
            continue
        seen.add(key)
        redacted = bool(spec.redact)
        if (omit_keep_spans or skip_keep_types) and not redacted:
            continue
        candidates.append(
            SpanHit(
                start=start,
                end=end,
                text=text[start:end],
                type_id=item.type_id,
                score=item.score,
                redacted=redacted,
            )
        )

    candidates.sort(
        key=lambda hit: (-hit.score, -(hit.end - hit.start), hit.start, order[hit.type_id])
    )
    kept: list[SpanHit] = []
    for hit in candidates:
        if any(hit.start < other.end and other.start < hit.end for other in kept):
            continue
        kept.append(hit)

    kept.sort(key=lambda hit: hit.start)
    return kept


def group_spans(hits: list[SpanHit]) -> tuple[list[EntitySpan], list[int]]:
    """Group occurrences by (type, text, redacted) and return per-hit group ids."""
    groups: list[EntitySpan] = []
    group_index: list[int] = []
    lookup: dict[tuple[str, str, bool], int] = {}

    for hit in hits:
        key = (hit.type_id, hit.text, hit.redacted)
        index = lookup.get(key)
        if index is None:
            index = len(groups)
            lookup[key] = index
            groups.append(
                EntitySpan(
                    type=hit.type_id,
                    text=hit.text,
                    redacted=hit.redacted,
                    occurrences=[],
                )
            )
        groups[index].occurrences.append(Occurrence(start=hit.start, end=hit.end, score=hit.score))
        group_index.append(index)

    return groups, group_index
