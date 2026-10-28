from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from anypii_flash.api import create_app
from anypii_flash.config import Settings


@pytest.mark.skipif(
    not os.getenv("ANYPPI_FLASH_TEST_MODEL"),
    reason="set ANYPPI_FLASH_TEST_MODEL to run the real-model smoke test",
)
def test_real_model_smoke():
    settings = Settings(
        model_id=os.environ["ANYPPI_FLASH_TEST_MODEL"],
        warmup=False,
        result_cache_entries=0,
    )
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        assert health["ready"] is True
        response = client.post(
            "/v1/anypii",
            json={
                "state": {"text": "患者陈思远因肺炎入院。主治医师林婉清、责任护士周雅琴。"},
                "types": {
                    "patient_name": {
                        "description": "患者或患儿的姓名，不是医务人员。",
                        "redact": True,
                    },
                    "clinician_name": {
                        "description": "医生、护士、药师等医务人员姓名。",
                        "redact": False,
                    },
                },
                "threshold": 0.55,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["usage"]["flash"]["forward_passes"] >= 1
    assert any(span["type"] == "patient_name" for span in body["spans"])
    assert "陈思远" not in body["redacted_text"]
    assert "林婉清" in body["redacted_text"]
