from __future__ import annotations

from .spans import SpanHit


def apply_mask(text: str, hits: list[SpanHit], group_index: list[int], style: str) -> str:
    if style not in ("token", "length", "mask", "type", "typed"):
        raise ValueError(f"unsupported mask_style: {style}")
    if style == "typed" and len(group_index) != len(hits):
        raise ValueError("typed mask requires one group index per hit")

    out: list[str] = []
    cursor = 0
    for index, hit in enumerate(hits):
        start = max(cursor, hit.start)
        end = min(len(text), hit.end)
        if start >= end:
            continue
        out.append(text[cursor:start])
        if hit.redacted:
            if style == "token":
                out.append("[REDACTED]")
            elif style == "type":
                out.append(f"[{hit.type_id}]")
            elif style == "typed":
                out.append(f"[{hit.type_id}:{group_index[index]}]")
            else:
                out.append("\u2588" * (end - start))
        else:
            out.append(text[start:end])
        cursor = end
    out.append(text[cursor:])
    return "".join(out)
