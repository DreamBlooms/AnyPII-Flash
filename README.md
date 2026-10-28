# AnyPII Flash

**Schema-driven PII redaction with GLiNER2.5. Readout-free. No generated text.**

AnyPII Flash is an independent Python service for local text redaction. It uses
[`fastino/gliner2.5-multi-v1`](https://huggingface.co/fastino/gliner2.5-multi-v1)
to extract schema-defined entities, then applies deterministic masking.

This is the Flash v1 scope:

- extraction, masking, HTTP API, and result cache;
- explicit `redact: true/false` only;
- no llama.cpp, Qwen, logit readout, PCD, or regex extraction engine;
- `redact: "auto"` will be enabled later with
  [`fastino/GLiNER2.5-multi-Decide`](https://huggingface.co/fastino/GLiNER2.5-multi-Decide).

## Install

Python 3.12 is recommended. GLiNER2.5, PyTorch, and Transformers are required
for local inference.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e ".[model,dev]"
```

Download the checkpoint once:

```sh
.venv/bin/python scripts/download_model.py \
  --endpoint https://hf-mirror.com \
  --out models/gliner2.5-multi-v1
```

Use `--endpoint` only when a mirror is needed. Without it, the standard Hugging
Face endpoint is used.

## Run

```sh
ANYPPI_FLASH_MODEL=models/gliner2.5-multi-v1 \
  .venv/bin/anypii-flash --port 8080
```

The model is loaded during startup and warmed up before `/health` reports
`ready: true`.

```sh
curl -s http://127.0.0.1:8080/v1/anypii \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {
      "text": "患者陈思远因肺炎入院。主治医师林婉清、责任护士周雅琴。"
    },
    "types": {
      "patient_name": {"description": "患者或患儿的姓名，不是医务人员。", "redact": true},
      "clinician_name": {"description": "医生、护士、药师等医务人员姓名。", "redact": false}
    },
    "threshold": 0.55,
    "mask_style": "token"
  }'
```

Both `/v1/anypii` and `/v1/flash` use the same handler.

The `examples/` directory contains the same seven requests as AnyPII. With a
server running:

```sh
python benchmarks/run_examples.py   --url http://127.0.0.1:8080/v1/anypii   --examples examples
```

## API

Request:

| field | meaning |
| --- | --- |
| `state.text` | input text |
| `state.policy` | accepted for compatibility; not used in v1 |
| `types` | map of type id to definition |
| `types.*.description` | semantic description passed to GLiNER2 |
| `types.*.redact` | required explicit `true` or `false` |
| `threshold` | GLiNER confidence threshold, default `0.55` |
| `mask_style` | `token`, `length`, `mask`, `type`, or `typed` |
| `skip_keep_types` | omit keep spans and skip model work when every type is keep |
| `omit_keep_spans` | omit `redacted:false` spans from the response |

Supported response fields:

```json
{
  "redacted_text": "患者[REDACTED]因肺炎入院。",
  "spans": [
    {
      "type": "patient_name",
      "text": "陈思远",
      "redacted": true,
      "occurrences": [{"start": 2, "end": 5, "score": 0.99}]
    }
  ],
  "model": "gliner2.5-multi-v1",
  "backend": "PyTorch CPU",
  "usage": {
    "decodes": 0,
    "branches": 0,
    "generated_tokens": 0,
    "fallback": false,
    "flash": {
      "splitter": "char",
      "chunks": 1,
      "word_tokens": 24,
      "forward_passes": 1,
      "elapsed_ms": 90.0,
      "cache_status": "miss",
      "cache_entries": 1
    }
  }
}
```

`usage.flash.chunks` and `forward_passes` are estimates based on the configured
word splitter and chunking parameters.

### Differences from AnyPII

- `mode` defaults to `"flash"`; other modes return `400`.
- `redact:"auto"` returns `400` until the Decide classifier is integrated.
- `policy` is accepted but ignored in v1.
- `parrot_typed` is rejected.
- No `readout`, `pcd`, or `fallback` inference paths are exposed.
- No separate regex/NIST fast path: all extraction comes from GLiNER2.5.

## Configuration

Environment variables:

```text
ANYPPI_FLASH_MODEL                 model id or local directory
ANYPPI_FLASH_DEVICE                cpu, cuda, cuda:0, mps
ANYPPI_FLASH_SPLITTER              char (default) or whitespace
ANYPPI_FLASH_CHUNK_SIZE            384 words
ANYPPI_FLASH_CHUNK_OVERLAP         64 words
ANYPPI_FLASH_MAX_TYPES             25
ANYPPI_FLASH_MAX_TEXT_CP           16000
ANYPPI_FLASH_MAX_PAYLOAD_BYTES     2097152
ANYPPI_FLASH_THRESHOLD             0.55
ANYPPI_FLASH_HOST                  127.0.0.1
ANYPPI_FLASH_PORT                  8080
ANYPPI_FLASH_API_KEY               optional bearer token
ANYPPI_FLASH_RESULT_CACHE_ENTRIES  64
ANYPPI_FLASH_RESULT_CACHE_BYTES    67108864
ANYPPI_FLASH_WARMUP                1
ANYPPI_FLASH_LOG_LEVEL             info
```

CLI flags override the corresponding environment variables. Run
`anypii-flash --help` for the full list.

## How it works

```text
text + types
  -> GLiNER2 schema
  -> AutoExtractor
       boundary architecture
       char word splitter
       short text: extract_entities
       long text: extract_entities_long
  -> normalize spans
       trim punctuation
       remove overlaps
       group repeated occurrences
  -> apply redact flags
  -> token / length / type / typed mask
  -> response
```

The model is loaded once per process. Requests are serialized around the
PyTorch model; the result cache remains safe for concurrent HTTP requests.

## Tests

```sh
PYTHONPATH=src .venv/bin/pytest
```

The default suite uses a fake extractor and does not download the model. It
covers API compatibility, masking, caching, errors, and skip/omit behavior.

Optional model smoke test:

```sh
ANYPPI_FLASH_TEST_MODEL=models/gliner2.5-multi-v1 \
  PYTHONPATH=src .venv/bin/pytest tests/test_model.py
```

## Limits

- GLiNER2.5 may return shorter spans than AnyPII. For example, it can return
  `林婉清` instead of `主治医师林婉清`.
- Without a regex layer, structured PII such as phone numbers, national IDs,
  bank cards, and API keys relies entirely on GLiNER2.5.
- Long Chinese documents use character-level chunking; latency grows with the
  number of chunks.
- The v1 API does not implement policy-based auto redaction.
- Model weights are a separate artifact. Verify the model card and license
  before redistribution.

## License

MIT for this project. GLiNER2 is Apache-2.0; model weights keep their own terms.
