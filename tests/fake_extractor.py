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
