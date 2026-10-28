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
    redact_threshold: float | None = None
    mask_style: MaskStyle = "token"
    mode: str = "flash"
    type_refine: bool | None = None
    type_refine_threshold: float | None = None
    skip_keep_types: bool = False
    omit_keep_spans: bool = False
    parrot_typed: bool = False
    fallback: bool = True


class Occurrence(BaseModel):
    start: int
    end: int
    score: float
    redact_score: float | None = None


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


class PlusUsage(BaseModel):
    model: str
    backend: str
    decisions: int
    forward_passes: int
    elapsed_ms: float
    policy_threshold: float
    policy_action: str | None = None
    action_confidence: float = 0.0
    type_refined: int = 0
    type_refine_forward_passes: int = 0
    type_refine_threshold: float | None = None


class Usage(BaseModel):
    decodes: int = 0
    branches: int = 0
    generated_tokens: int = 0
    fallback: bool = False
    flash: FlashUsage
    plus: PlusUsage | None = None


class AnyPiiResponse(BaseModel):
    redacted_text: str
    spans: list[EntitySpan]
    model: str
    backend: str
    usage: Usage
