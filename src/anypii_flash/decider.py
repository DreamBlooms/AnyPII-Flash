from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import Settings
from .errors import ModelUnavailableError
from .extractor import TypeSpec

# These labels are the classifier's prompt vocabulary, not PII rules.
# They stay in Chinese: A/B tests showed English redact/keep labels collapsed
# generic-drug and brand decisions more often on the same examples.
GLOBAL_KEEP = "全部保留"
GLOBAL_REDACT = "全部打码"
GLOBAL_PARTIAL_REDACT = "部分打码"
GLOBAL_PARTIAL_KEEP = "部分保留"
SPAN_REDACT = "打码"
SPAN_KEEP = "保留"


@dataclass(frozen=True)
class DecisionItem:
    """One extracted span waiting for a policy decision."""

    index: int
    type_id: str
    description: str
    text: str
    context: str
    candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionDecision:
    action: str | None
    confidence: float
    elapsed_ms: float
    forward_passes: int = 1


@dataclass(frozen=True)
class TypeRefinementBatch:
    # Maps the original hit index to (new type id, confidence).
    types: dict[int, tuple[str, float]]
    elapsed_ms: float
    forward_passes: int


@dataclass(frozen=True)
class DecisionBatch:
    # Maps the original hit index to P(redact).
    scores: dict[int, float]
    elapsed_ms: float
    forward_passes: int
    action: str | None = None
    action_confidence: float = 0.0


class Decider(Protocol):
    model_name: str
    backend_name: str
    ready: bool

    def classify_action(self, text: str, policy: str) -> ActionDecision: ...

    def decide(
        self,
        text: str,
        policy: str,
        items: list[DecisionItem],
        action: ActionDecision | None = None,
    ) -> DecisionBatch: ...

    def refine_types(
        self,
        text: str,
        items: list[DecisionItem],
        types: list[TypeSpec],
        threshold: float | None = None,
    ) -> TypeRefinementBatch: ...


class DecideClassifier:
    """Classify each extracted span with the original text and policy context.

    The pipeline is:

    1. classify the policy globally as all-keep, all-redact, or mixed;
    2. for a mixed policy, classify every extracted span independently.

    Every per-span call receives the complete original text as the classifier
    input, plus the policy, mention, local context, and type description as the
    task prompt. This keeps decisions tied to the actual extracted result rather
    than only to its type.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = Path(settings.plus_model_id).name or settings.plus_model_id
        self.backend_name = settings.backend_label
        self._lock = threading.Lock()
        self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

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
                self.settings.plus_model_id,
                map_location=self.settings.device,
            )
            self.model.eval()
        except Exception as exc:
            raise ModelUnavailableError(
                f"failed to load Decide model {self.settings.plus_model_id!r}: {exc}"
            ) from exc

    def _classify(self, text: str, tasks: dict) -> dict:
        with self._lock:
            try:
                return self.model.classify_text(
                    text,
                    tasks,
                    include_confidence=True,
                )
            except Exception as exc:
                raise ModelUnavailableError(f"Decide classification failed: {exc}") from exc

    def refine_types(
        self,
        text: str,
        items: list[DecisionItem],
        types: list[TypeSpec],
        threshold: float | None = None,
    ) -> TypeRefinementBatch:
        if not text or not items or not types:
            return TypeRefinementBatch(types={}, elapsed_ms=0.0, forward_passes=0)
        self.load()
        threshold = threshold if threshold is not None else self.settings.type_refine_threshold
        by_id = {spec.id: spec for spec in types}
        groups: dict[tuple[str, ...], list[DecisionItem]] = {}
        for item in items:
            candidate_ids = item.candidate_ids or tuple(by_id)
            groups.setdefault(candidate_ids, []).append(item)

        started = time.perf_counter()
        refined: dict[int, tuple[str, float]] = {}
        forward_passes = 0
        for candidate_ids, group in groups.items():
            candidate_specs = [by_id[type_id] for type_id in candidate_ids if type_id in by_id]
            if len(candidate_specs) < 2:
                continue
            labels = {spec.id: (spec.description or spec.id) for spec in candidate_specs}
            texts = [
                f"{text}\n\nMention:\n{item.text}\n\nContext:\n{item.context}\n\n"
                "Which type best describes this mention?"
                for item in group
            ]
            results = self._batch_classify(
                texts,
                {"type": {"labels": labels}},
            )
            forward_passes += (
                len(group) + self.settings.plus_batch_size - 1
            ) // self.settings.plus_batch_size
            for item, result in zip(group, results):
                value = result.get("type", {})
                if not isinstance(value, dict):
                    continue
                label = str(value.get("label", ""))
                confidence = float(value.get("confidence", 0.0))
                if label in labels and confidence >= threshold:
                    refined[item.index] = (label, confidence)
        return TypeRefinementBatch(
            types=refined,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            forward_passes=forward_passes,
        )

    def _batch_classify(self, texts: list[str], tasks: dict) -> list[dict]:
        with self._lock:
            try:
                return self.model.batch_classify_text(
                    texts,
                    tasks,
                    batch_size=self.settings.plus_batch_size,
                    include_confidence=True,
                )
            except Exception as exc:
                raise ModelUnavailableError(f"Decide classification failed: {exc}") from exc

    def classify_action(self, text: str, policy: str) -> ActionDecision:
        self.load()
        started = time.perf_counter()
        decision_text = text
        if policy:
            decision_text += "\n\nPolicy:\n" + policy
        result = self._classify(
            decision_text,
            {
                "action": {
                    "labels": [
                        GLOBAL_KEEP,
                        GLOBAL_REDACT,
                        GLOBAL_PARTIAL_REDACT,
                        GLOBAL_PARTIAL_KEEP,
                    ]
                }
            },
        )
        value = result.get("action", {})
        if isinstance(value, dict):
            action = str(value.get("label", ""))
            confidence = float(value.get("confidence", 0.0))
        else:
            action = str(value)
            confidence = 0.0
        return ActionDecision(
            action=action,
            confidence=confidence,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def decide(
        self,
        text: str,
        policy: str,
        items: list[DecisionItem],
        action: ActionDecision | None = None,
    ) -> DecisionBatch:
        if not text or not items:
            return DecisionBatch(scores={}, elapsed_ms=0.0, forward_passes=0)
        self.load()
        started = time.perf_counter()
        if action is None:
            action = self.classify_action(text, policy)
        action_name = action.action
        action_confidence = action.confidence

        if action_name in (GLOBAL_KEEP, GLOBAL_REDACT) and (
            action_confidence >= self.settings.policy_action_threshold
        ):
            score = 1.0 if action_name == GLOBAL_REDACT else 0.0
            return DecisionBatch(
                scores={item.index: score for item in items},
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                forward_passes=0,
                action=action_name,
                action_confidence=action_confidence,
            )

        texts = [
            f"{text}\n\nPolicy:\n{policy or '(none)'}\n\nMention:\n{item.text}\n\n"
            f"Context:\n{item.context}\n\nType:\n{item.type_id}: {item.description}\n"
            "Should this mention be redacted under the policy?"
            for item in items
        ]
        results = self._batch_classify(
            texts,
            {"decision": {"labels": [SPAN_REDACT, SPAN_KEEP]}},
        )
        scores: dict[int, float] = {}
        for item, result in zip(items, results):
            value = result.get("decision", {})
            if isinstance(value, dict):
                label = str(value.get("label", ""))
                confidence = float(value.get("confidence", 0.0))
            else:
                label = str(value)
                confidence = 0.0
            scores[item.index] = confidence if label == SPAN_REDACT else 1.0 - confidence

        return DecisionBatch(
            scores=scores,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            forward_passes=1
            + (len(items) + self.settings.plus_batch_size - 1) // self.settings.plus_batch_size,
            action=action_name,
            action_confidence=action_confidence,
        )
