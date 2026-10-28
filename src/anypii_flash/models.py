from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MaskStyle = Literal["token", "length", "mask", "type", "typed"]
RedactValue = bool | Literal["auto"]


class State(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    policy: str | None = ""


class TypeDef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str = ""
    redact: RedactValue | None = None


class AnyPiiRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: State
    types: dict[str, TypeDef] = Field(default_factory=dict)
    threshold: float = 0.55
    mask_style: MaskStyle = "token"
    mode: str = "flash"
    skip_keep_types: bool = False
    omit_keep_spans: bool = False
    parrot_typed: bool = False
    fallback: bool = True


class Occurrence(BaseModel):
    start: int
    end: int
    score: float


class EntitySpan(BaseModel):
    type: str
    text: str
    redacted: bool
    occurrences: list[Occurrence]


class FlashUsage(BaseModel):
    splitter: str
    chunks: int
    word_tokens: int
    forward_passes: int
    elapsed_ms: float
    cache_status: str
    cache_entries: int = 0


class Usage(BaseModel):
    decodes: int = 0
    branches: int = 0
    generated_tokens: int = 0
    fallback: bool = False
    flash: FlashUsage


class AnyPiiResponse(BaseModel):
    redacted_text: str
    spans: list[EntitySpan]
    model: str
    backend: str
    usage: Usage
