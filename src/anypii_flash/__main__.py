from __future__ import annotations

import argparse
import dataclasses

import uvicorn

from .api import create_app
from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="AnyPII Flash")
    parser.add_argument("--model", default=settings.model_id)
    parser.add_argument("--plus-model", default=settings.plus_model_id)
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--splitter", default=settings.splitter)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--threshold", type=float, default=settings.threshold)
    parser.add_argument("--policy-threshold", type=float, default=settings.policy_threshold)
    parser.add_argument(
        "--policy-action-threshold",
        type=float,
        default=settings.policy_action_threshold,
    )
    parser.add_argument(
        "--type-refine-threshold", type=float, default=settings.type_refine_threshold
    )
    parser.add_argument("--plus-batch-size", type=int, default=settings.plus_batch_size)
    parser.add_argument("--api-key", default=settings.api_key)
    parser.add_argument("--result-cache-entries", type=int, default=settings.result_cache_entries)
    parser.add_argument("--result-cache-bytes", type=int, default=settings.result_cache_bytes)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--log-level", default=settings.log_level)
    args = parser.parse_args()

    settings = dataclasses.replace(
        settings,
        model_id=args.model,
        plus_model_id=args.plus_model,
        host=args.host,
        port=args.port,
        device=args.device,
        splitter=args.splitter,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        threshold=args.threshold,
        policy_threshold=args.policy_threshold,
        policy_action_threshold=args.policy_action_threshold,
        type_refine_threshold=args.type_refine_threshold,
        plus_batch_size=args.plus_batch_size,
        api_key=args.api_key,
        result_cache_entries=args.result_cache_entries,
        result_cache_bytes=args.result_cache_bytes,
        warmup=False if args.no_warmup else settings.warmup,
        log_level=args.log_level,
    )

    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)


if __name__ == "__main__":
    main()
