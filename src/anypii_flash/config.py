from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    model_id: str = "fastino/gliner2.5-multi-v1"
    plus_model_id: str = "fastino/GLiNER2.5-multi-Decide"
    device: str = "cpu"
    splitter: str = "char"
    chunk_size: int = 384
    chunk_overlap: int = 64
    max_types: int = 25
    max_text_cp: int = 16000
    max_payload_bytes: int = 2 * 1024 * 1024
    threshold: float = 0.55
    policy_threshold: float = 0.5
    policy_action_threshold: float = 0.8
    type_refine_threshold: float = 0.7
    plus_batch_size: int = 8
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = ""
    result_cache_entries: int = 64
    result_cache_bytes: int = 64 * 1024 * 1024
    warmup: bool = True
    log_level: str = "info"

    @property
    def model_label(self) -> str:
        name = Path(self.model_id).name
        return name or self.model_id

    @property
    def backend_label(self) -> str:
        device = self.device.lower()
        if device.startswith("cuda"):
            return f"PyTorch CUDA ({self.device})"
        if device == "mps":
            return "PyTorch MPS"
        return "PyTorch CPU"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_id=os.getenv("ANYPPI_FLASH_MODEL", cls.model_id),
            plus_model_id=os.getenv("ANYPPI_FLASH_PLUS_MODEL", cls.plus_model_id),
            device=os.getenv("ANYPPI_FLASH_DEVICE", cls.device),
            splitter=os.getenv("ANYPPI_FLASH_SPLITTER", cls.splitter),
            chunk_size=_env_int("ANYPPI_FLASH_CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("ANYPPI_FLASH_CHUNK_OVERLAP", cls.chunk_overlap),
            max_types=_env_int("ANYPPI_FLASH_MAX_TYPES", cls.max_types),
            max_text_cp=_env_int("ANYPPI_FLASH_MAX_TEXT_CP", cls.max_text_cp),
            max_payload_bytes=_env_int("ANYPPI_FLASH_MAX_PAYLOAD_BYTES", cls.max_payload_bytes),
            threshold=_env_float("ANYPPI_FLASH_THRESHOLD", cls.threshold),
            policy_threshold=_env_float("ANYPPI_FLASH_POLICY_THRESHOLD", cls.policy_threshold),
            host=os.getenv("ANYPPI_FLASH_HOST", cls.host),
            port=_env_int("ANYPPI_FLASH_PORT", cls.port),
            api_key=os.getenv("ANYPPI_FLASH_API_KEY", cls.api_key),
            result_cache_entries=_env_int(
                "ANYPPI_FLASH_RESULT_CACHE_ENTRIES", cls.result_cache_entries
            ),
            result_cache_bytes=_env_int("ANYPPI_FLASH_RESULT_CACHE_BYTES", cls.result_cache_bytes),
            warmup=os.getenv("ANYPPI_FLASH_WARMUP", "1").lower() not in ("0", "false", "no"),
            log_level=os.getenv("ANYPPI_FLASH_LOG_LEVEL", cls.log_level),
        )
