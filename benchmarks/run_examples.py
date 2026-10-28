#!/usr/bin/env python3
"""Run AnyPII JSON examples against a running AnyPII Flash server."""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import urllib.request


def post(url: str, payload: dict) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=os.getenv("ANYPPI_FLASH_URL", "http://127.0.0.1:8080/v1/anypii"),
    )
    parser.add_argument(
        "--examples",
        default=os.path.join(os.path.dirname(__file__), "..", "examples"),
    )
    args = parser.parse_args()

    for path in sorted(glob.glob(os.path.join(args.examples, "*.json"))):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        started = time.perf_counter()
        result = post(args.url, payload)
        elapsed = time.perf_counter() - started
        print("=" * 72)
        print(os.path.basename(path), f"{elapsed:.2f}s", result.get("usage"))
        print(result["redacted_text"])
        for span in result["spans"]:
            print(
                f"  {span['type']:<18} {span['text']!r:<28} "
                f"redacted={span['redacted']} occurrences={len(span['occurrences'])}"
            )


if __name__ == "__main__":
    main()
