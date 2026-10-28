from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings
from .errors import ModelUnavailableError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TypeSpec:
    id: str
    description: str
    redact: bool


@dataclass(frozen=True)
class ExtractedSpan:
    start: int
    end: int
    type_id: str
    score: float
    text: str


@dataclass(frozen=True)
class ExtractionStats:
    chunks: int = 1
    word_tokens: int = 0
    forward_passes: int = 1


class Extractor(Protocol):
    model_name: str
    backend_name: str

    def extract(
        self, text: str, types: list[TypeSpec], threshold: float
    ) -> tuple[list[ExtractedSpan], ExtractionStats]: ...


def estimate_word_tokens(text: str) -> int:
    """Approximate the GLiNER word count for both CJK and space-delimited text."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff")
    latin = len(re.findall(r"[A-Za-z0-9_]+", text))
    other = sum(
        1
        for ch in text
        if not ch.isspace()
        and not ("\u3400" <= ch <= "\u9fff")
        and not ("\uf900" <= ch <= "\ufaff")
        and not re.match(r"[A-Za-z0-9_]", ch)
    )
    return max(1, cjk + latin + other)


class GlinerExtractor:
    """Thin adapter around GLiNER2.5's AutoExtractor boundary model."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.model_label
        self.backend_name = settings.backend_label
        self._lock = threading.Lock()
        self.model: Any = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from gliner2 import AutoExtractor
        except Exception as exc:
            raise ModelUnavailableError(
                "gliner2 is not installed. Install the model extra with "
                '`pip install "gliner2[local]"`.'
            ) from exc

        try:
            self.model = AutoExtractor.from_pretrained(
                self.settings.model_id,
                word_splitter=self.settings.splitter,
                map_location=self.settings.device,
            )
            self.model.eval()
        except Exception as exc:
            raise ModelUnavailableError(
                f"failed to load GLiNER2 model {self.settings.model_id!r}: {exc}"
            ) from exc

    def warmup(self) -> None:
        self.load()
        try:
            self.extract("warmup", [TypeSpec("warmup", "warmup text", True)], 0.5)
        except Exception:
            log.exception("warmup failed")
            raise

    def extract(
        self, text: str, types: list[TypeSpec], threshold: float
    ) -> tuple[list[ExtractedSpan], ExtractionStats]:
        if not text or not types:
            return [], ExtractionStats(chunks=0, word_tokens=0, forward_passes=0)
        self.load()
        type_ids = {f"t{index}": spec.id for index, spec in enumerate(types)}
        entity_types = {key: (spec.description or spec.id) for key, spec in zip(type_ids, types)}
        units = estimate_word_tokens(text)
        chunks = (
            1
            if units <= self.settings.chunk_size
            else max(
                1,
                math.ceil(
                    (units - self.settings.chunk_size)
                    / max(1, self.settings.chunk_size - self.settings.chunk_overlap)
                )
                + 1,
            )
        )

        with self._lock:
            try:
                if units <= self.settings.chunk_size:
                    result = self.model.extract_entities(
                        text,
                        entity_types,
                        threshold=threshold,
                        include_spans=True,
                        include_confidence=True,
                        overlap_policy="flat",
                    )
                else:
                    result = self.model.extract_entities_long(
                        text,
                        entity_types,
                        threshold=threshold,
                        chunk_size=self.settings.chunk_size,
                        chunk_overlap=self.settings.chunk_overlap,
                        include_spans=True,
                        include_confidence=True,
                        overlap_policy="flat",
                    )
            except Exception as exc:
                raise ModelUnavailableError(f"GLiNER2 extraction failed: {exc}") from exc

        spans = self._parse_result(result, text, type_ids, threshold)
        return spans, ExtractionStats(chunks=chunks, word_tokens=units, forward_passes=chunks)

    @staticmethod
    def _parse_result(
        result: dict[str, Any],
        text: str,
        type_ids: dict[str, str],
        threshold: float,
    ) -> list[ExtractedSpan]:
        entities = result.get("entities", {})
        if isinstance(entities, list):
            merged: dict[str, Any] = {}
            for item in entities:
                if isinstance(item, dict):
                    merged.update(item)
            entities = merged
        if not isinstance(entities, dict):
            return []

        out: list[ExtractedSpan] = []
        seen: set[tuple[int, int, str]] = set()
        for safe_type, values in entities.items():
            type_id = type_ids.get(safe_type, safe_type)
            if isinstance(values, dict):
                values = [values]
            if not isinstance(values, list):
                continue
            for value in values:
                start: int
                end: int
                score = 1.0
                if isinstance(value, dict):
                    if "start" not in value or "end" not in value:
                        continue
                    start = int(value["start"])
                    end = int(value["end"])
                    score = float(value.get("confidence", 1.0))
                elif isinstance(value, str):
                    start = text.find(value)
                    if start < 0:
                        continue
                    end = start + len(value)
                else:
                    continue
                if score < threshold:
                    continue
                if not (0 <= start < end <= len(text)):
                    continue
                span_text = text[start:end]
                if not span_text.strip():
                    continue
                key = (start, end, type_id)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    ExtractedSpan(
                        start=start,
                        end=end,
                        type_id=type_id,
                        score=score,
                        text=span_text,
                    )
                )
        return out
