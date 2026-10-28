from __future__ import annotations

import pytest
from fake_extractor import FakeExtractor
from fastapi.testclient import TestClient

from anypii_flash.api import create_app
from anypii_flash.config import Settings

MEDICAL = "患者陈思远因肺炎入院。主治医师林婉清、责任护士周雅琴。予阿莫西林（商品名阿莫仙，辉瑞制药）抗感染。"
TYPES = {
    "patient_name": {"description": "患者姓名", "redact": True},
    "clinician_name": {"description": "医务人员姓名", "redact": False},
    "pharma_company": {"description": "药厂名称", "redact": True},
    "drug_brand": {"description": "药品品牌名", "redact": True},
    "drug_generic": {"description": "药品通用名", "redact": False},
}


@pytest.fixture
def client():
    settings = Settings(result_cache_entries=16, result_cache_bytes=1024 * 1024)
    fake = FakeExtractor()
    app = create_app(settings, extractor=fake)
    with TestClient(app) as test_client:
        yield test_client, fake


def payload(**overrides):
    body = {
        "state": {"text": MEDICAL, "policy": "打码患者姓名。"},
        "types": TYPES,
        "threshold": 0.55,
        "mask_style": "token",
    }
    body.update(overrides)
    return body


def test_health(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["architecture"] == "boundary"


def test_token_redaction(client):
    test_client, fake = client
    response = test_client.post("/v1/anypii", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert body["redacted_text"] == (
        "患者[REDACTED]因肺炎入院。主治医师林婉清、责任护士周雅琴。"
        "予阿莫西林（商品名[REDACTED]，[REDACTED]）抗感染。"
    )
    assert body["usage"]["generated_tokens"] == 0
    assert body["usage"]["flash"]["cache_status"] == "miss"
    assert fake.calls == 1
    assert [span["type"] for span in body["spans"]] == [
        "patient_name",
        "clinician_name",
        "clinician_name",
        "drug_generic",
        "drug_brand",
        "pharma_company",
    ]


def test_mask_styles(client):
    test_client, _ = client
    length = test_client.post("/v1/anypii", json=payload(mask_style="length")).json()
    assert "█" * 3 in length["redacted_text"]

    typed = test_client.post("/v1/anypii", json=payload(mask_style="typed")).json()
    assert "[patient_name:0]" in typed["redacted_text"]
    assert "[drug_brand:4]" in typed["redacted_text"]

    by_type = test_client.post("/v1/anypii", json=payload(mask_style="type")).json()
    assert "[patient_name]" in by_type["redacted_text"]


def test_skip_and_omit_keep(client):
    test_client, _ = client
    skipped = test_client.post("/v1/anypii", json=payload(skip_keep_types=True)).json()
    assert all(span["redacted"] for span in skipped["spans"])
    assert "林婉清" in skipped["redacted_text"]
    assert "阿莫西林" in skipped["redacted_text"]

    omitted = test_client.post("/v1/anypii", json=payload(omit_keep_spans=True)).json()
    assert all(span["redacted"] for span in omitted["spans"])
    assert "林婉清" in omitted["redacted_text"]


def test_all_keep_skips_model(client):
    test_client, fake = client
    types = {
        key: {"description": value["description"], "redact": False} for key, value in TYPES.items()
    }
    before = fake.calls
    body = test_client.post("/v1/anypii", json=payload(types=types, skip_keep_types=True)).json()
    assert body["redacted_text"] == MEDICAL
    assert body["spans"] == []
    assert fake.calls == before


def test_empty_text(client):
    test_client, fake = client
    before = fake.calls
    body = test_client.post("/v1/anypii", json=payload(state={"text": ""})).json()
    assert body["redacted_text"] == ""
    assert body["spans"] == []
    assert fake.calls == before


def test_result_cache(client):
    test_client, fake = client
    first = test_client.post("/v1/anypii", json=payload()).json()
    second = test_client.post("/v1/anypii", json=payload()).json()
    assert fake.calls == 1
    assert second["usage"]["flash"]["cache_status"] == "hit"
    assert second["usage"]["flash"]["elapsed_ms"] == 0.0
    assert second["spans"] == first["spans"]


def test_rejects_auto_and_unsupported_mode(client):
    test_client, _ = client
    auto_types = {
        **TYPES,
        "patient_name": {"description": "患者姓名", "redact": "auto"},
    }
    response = test_client.post("/v1/anypii", json=payload(types=auto_types))
    assert response.status_code == 400
    assert "not implemented yet" in response.json()["error"]["message"]

    response = test_client.post("/v1/anypii", json=payload(mode="readout"))
    assert response.status_code == 400
    assert "only supports" in response.json()["error"]["message"]


def test_rejects_missing_redact_and_bad_threshold(client):
    test_client, _ = client
    missing = {**TYPES, "patient_name": {"description": "患者姓名"}}
    response = test_client.post("/v1/anypii", json=payload(types=missing))
    assert response.status_code == 400
    assert "redact is required" in response.json()["error"]["message"]

    response = test_client.post("/v1/anypii", json=payload(threshold=2.0))
    assert response.status_code == 400
    assert "threshold" in response.json()["error"]["message"]


def test_too_many_types():
    settings = Settings(max_types=1)
    with TestClient(create_app(settings, extractor=FakeExtractor())) as test_client:
        response = test_client.post("/v1/anypii", json=payload())
    assert response.status_code == 400
    assert "at most 1 types" in response.json()["error"]["message"]


def test_api_key():
    settings = Settings(api_key="secret")
    with TestClient(create_app(settings, extractor=FakeExtractor())) as test_client:
        unauthorized = test_client.post("/v1/anypii", json=payload())
        authorized = test_client.post(
            "/v1/anypii",
            json=payload(),
            headers={"Authorization": "Bearer secret"},
        )
    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_flash_alias(client):
    test_client, _ = client
    response = test_client.post("/v1/flash", json=payload())
    assert response.status_code == 200
    assert "redacted_text" in response.json()


def test_validation_is_400(client):
    test_client, _ = client
    response = test_client.post("/v1/anypii", json={"types": {}})
    assert response.status_code == 400
    assert "error" in response.json()
