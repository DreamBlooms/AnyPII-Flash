from __future__ import annotations

from anypii_flash.extractor import ExtractedSpan, TypeSpec
from anypii_flash.masking import apply_mask
from anypii_flash.spans import group_spans, normalize_spans


def span(start: int, end: int, type_id: str, score: float = 0.9) -> ExtractedSpan:
    return ExtractedSpan(
        start=start,
        end=end,
        type_id=type_id,
        score=score,
        text="",
    )


def test_trims_sentence_punctuation():
    text = "Alice works at Acme Corp."
    types = [TypeSpec("company", "company", True)]
    hits = normalize_spans([span(15, 25, "company")], text, types, 0.5)
    assert len(hits) == 1
    assert hits[0].text == "Acme Corp"
    assert (hits[0].start, hits[0].end) == (15, 24)


def test_skip_keep_drops_keep_before_overlap():
    text = "主治医师林婉清"
    types = [
        TypeSpec("patient_name", "patient", True),
        TypeSpec("clinician_name", "clinician", False),
    ]
    extracted = [
        span(0, 7, "patient_name", 0.99),
        span(0, 7, "clinician_name", 0.98),
    ]
    hits = normalize_spans(
        extracted,
        text,
        types,
        0.5,
        skip_keep_types=True,
    )
    assert [(hit.type_id, hit.text) for hit in hits] == [("patient_name", "主治医师林婉清")]


def test_groups_repeated_occurrences():
    text = "陈思远来访。陈思远离开。"
    types = [TypeSpec("patient_name", "patient", True)]
    hits = normalize_spans(
        [span(0, 3, "patient_name"), span(6, 9, "patient_name")],
        text,
        types,
        0.5,
    )
    groups, index = group_spans(hits)
    assert len(groups) == 1
    assert index == [0, 0]
    assert [(occ.start, occ.end) for occ in groups[0].occurrences] == [(0, 3), (6, 9)]


def test_masking_length_and_typed():
    text = "ABC"
    types = [TypeSpec("x", "x", True)]
    hits = normalize_spans([span(0, 3, "x")], text, types, 0.5)
    groups, index = group_spans(hits)
    assert apply_mask(text, hits, index, "token") == "[REDACTED]"
    assert apply_mask(text, hits, index, "length") == "███"
    assert apply_mask(text, hits, index, "type") == "[x]"
    assert apply_mask(text, hits, index, "typed") == "[x:0]"
    assert len(groups) == 1
