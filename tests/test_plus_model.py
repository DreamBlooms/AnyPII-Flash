from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from anypii_flash.api import create_app
from anypii_flash.config import Settings

MODEL = os.getenv("ANYPPI_FLASH_TEST_MODEL")
PLUS_MODEL = os.getenv("ANYPPI_FLASH_TEST_PLUS_MODEL")


@pytest.mark.skipif(
    not MODEL or not PLUS_MODEL,
    reason="set ANYPPI_FLASH_TEST_MODEL and ANYPPI_FLASH_TEST_PLUS_MODEL",
)
def test_real_flash_plus_smoke():
    settings = Settings(
        model_id=MODEL,
        plus_model_id=PLUS_MODEL,
        warmup=False,
        result_cache_entries=0,
        policy_threshold=0.5,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/anypii",
            json={
                "state": {
                    "text": "患者陈思远因肺炎入院。主治医师林婉清。",
                    "policy": "打码患者姓名；医生姓名保留。",
                },
                "types": {
                    "patient_name": {
                        "description": "患者姓名",
                        "redact": "auto",
                    },
                    "clinician_name": {
                        "description": "医务人员姓名",
                        "redact": "auto",
                    },
                },
                "mode": "flash-plus",
                "threshold": 0.55,
                "redact_threshold": 0.5,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["usage"]["plus"]["forward_passes"] >= 1
    assert body["usage"]["plus"]["decisions"] == 2
    assert body["spans"]
    assert any("redact_score" in span["occurrences"][0] for span in body["spans"])
