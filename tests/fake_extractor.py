from __future__ import annotations

from dataclasses import dataclass

from anypii_flash.extractor import ExtractedSpan, ExtractionStats, TypeSpec


@dataclass
class FakeExtractor:
    model_name: str = "fake-gliner2.5-multi-v1"
    backend_name: str = "Fake"
    calls: int = 0
    spans: dict[str, list[tuple[str, float]]] | None = None

    def __post_init__(self) -> None:
        if self.spans is None:
            self.spans = {
                "patient_name": [("陈思远", 0.98)],
                "clinician_name": [("主治医师林婉清", 0.96), ("责任护士周雅琴", 0.94)],
                "pharma_company": [("辉瑞制药", 0.90)],
                "drug_brand": [("阿莫仙", 0.90)],
                "drug_generic": [("阿莫西林", 0.90)],
                "person": [("Alice Johnson", 0.95)],
                "company": [("Acme Corp", 0.92)],
                "phone": [("13800138000", 0.99)],
                "email": [("alice@example.com", 0.99)],
            }

    def extract(
        self, text: str, types: list[TypeSpec], threshold: float
    ) -> tuple[list[ExtractedSpan], ExtractionStats]:
        self.calls += 1
        out: list[ExtractedSpan] = []
        for spec in types:
            for needle, score in self.spans.get(spec.id, []):
                start = text.find(needle)
                if start >= 0 and score >= threshold:
                    out.append(
                        ExtractedSpan(
                            start=start,
                            end=start + len(needle),
                            type_id=spec.id,
                            score=score,
                            text=needle,
                        )
                    )
        return out, ExtractionStats(chunks=1, word_tokens=len(text), forward_passes=1)


@dataclass
class FakeDecider:
    model_name: str = "fake-gliner2.5-multi-Decide"
    backend_name: str = "Fake"
    ready: bool = True
    scores: dict[str, float] | None = None
    calls: int = 0
    last_text: str = ""
    last_policy: str = ""
    last_items: list = None
    type_refinements: dict[str, tuple[str, float]] | None = None
    action: str | None = None
    action_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.scores is None:
            self.scores = {
                "patient_name": 0.9,
                "clinician_name": 0.1,
                "drug_generic": 0.2,
                "drug_brand": 0.8,
                "pharma_company": 0.4,
            }

    def refine_types(self, text: str, items, types, threshold: float | None = None):
        from anypii_flash.decider import TypeRefinementBatch

        threshold = 0.55 if threshold is None else threshold
        out = {}
        for item in items:
            value = (self.type_refinements or {}).get(item.type_id)
            if value is not None and value[1] >= threshold:
                out[item.index] = value
        return TypeRefinementBatch(
            types=out,
            elapsed_ms=5.0,
            forward_passes=len(items),
        )

    def classify_action(self, text: str, policy: str):
        from anypii_flash.decider import ActionDecision

        return ActionDecision(
            action=self.action,
            confidence=self.action_confidence,
            elapsed_ms=5.0,
            forward_passes=1,
        )

    def decide(self, text: str, policy: str, items, action=None):
        from anypii_flash.decider import DecisionBatch

        self.calls += 1
        self.last_text = text
        self.last_policy = policy
        self.last_items = list(items)
        return DecisionBatch(
            scores={item.index: self.scores.get(item.type_id, 0.0) for item in items},
            elapsed_ms=12.5,
            forward_passes=1,
            action=self.action,
            action_confidence=self.action_confidence,
        )
