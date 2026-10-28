#!/usr/bin/env python3
"""Download the GLiNER2.5 checkpoint into a local directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fastino/gliner2.5-multi-v1")
    parser.add_argument("--out", default="models/gliner2.5-multi-v1")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--xet",
        action="store_true",
        help="Allow the Xet transfer backend; disabled by default for mirror compatibility.",
    )
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    if not args.xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import snapshot_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=str(out),
    )
    print(out)


if __name__ == "__main__":
    main()
