# AnyPII Flash

English | [简体中文](README.zh-CN.md)

**Faster, arbitrary custom PII redaction with schema-driven GLiNER2.5.**

AnyPII Flash is an independent Python service for local text redaction. It uses
[`fastino/gliner2.5-multi-v1`](https://huggingface.co/fastino/gliner2.5-multi-v1)
to extract schema-defined entities, then applies deterministic masking.

Two modes are available:

- **`flash`**: explicit `redact: true/false`; fastest and deterministic.
- **`flash-plus`**: optional policy mode. After extraction, each span with
  `redact:"auto"` is optionally type-refined and classified independently by
  [`fastino/GLiNER2.5-multi-Decide`](https://huggingface.co/fastino/GLiNER2.5-multi-Decide)
  using the original text, local context, and natural-language policy.
  Type refinement is on by default in this mode.

## Install

Python 3.12 is recommended. GLiNER2.5, PyTorch, and Transformers are required
for local inference.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e ".[model,dev]"
```

Download the checkpoints once:

```sh
.venv/bin/python scripts/download_model.py \
  --endpoint https://hf-mirror.com \
  --out models/gliner2.5-multi-v1

.venv/bin/python scripts/download_model.py \
  --endpoint https://hf-mirror.com \
  --model fastino/GLiNER2.5-multi-Decide \
  --out models/gliner2.5-multi-Decide
```

Use `--endpoint` only when a mirror is needed. Without it, the standard Hugging
Face endpoint is used.

## Run

```sh
ANYPPI_FLASH_MODEL=models/gliner2.5-multi-v1 \
ANYPPI_FLASH_PLUS_MODEL=models/gliner2.5-multi-Decide \
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

### Flash-plus example

```sh
curl -s http://127.0.0.1:8080/v1/anypii \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {
      "text": "患者陈思远因肺炎入院。主治医师林婉清。",
      "policy": "打码患者姓名；医生姓名保留。"
    },
    "types": {
      "patient_name": {"description": "患者姓名", "redact": "auto"},
      "clinician_name": {"description": "医务人员姓名", "redact": "auto"}
    },
    "mode": "flash-plus",
    "redact_threshold": 0.5
  }'
```

The Decide input contains both the original text and the policy. With type
refinement enabled (default in `flash-plus`), each auto span first gets an
independent type classification against the request's types. The decision stage
then checks whether the policy is global (`全部保留` / `全部打码`); for mixed
policies it runs one redact/keep classification per extracted span. Global
all-keep/all-redact is accepted only above `policy_action_threshold` (default
`0.8`). The output adds `usage.plus` and `redact_score` on auto occurrences.

The `examples/` directory contains the same seven requests as AnyPII. With a
server running:

```sh
python benchmarks/run_examples.py \
  --url http://127.0.0.1:8080/v1/anypii \
  --examples examples
```

## API

Request:

| field | meaning |
| --- | --- |
| `state.text` | input text |
| `state.policy` | natural-language policy; required for `redact:"auto"` in `flash-plus` |
| `types` | map of type id to definition |
| `types.*.description` | semantic description passed to GLiNER2 |
| `types.*.redact` | `true`/`false`; `auto` requires `mode:"flash-plus"` |
| `threshold` | GLiNER extraction threshold, default `0.7` |
| `redact_threshold` | flash-plus policy threshold; defaults to the server setting (`0.5`) |
| `mode` | `flash` (default) or `flash-plus` |
| `type_refine` | optional; type refinement for auto spans in `flash-plus` (default `true`) |
| `type_refine_threshold` | optional; per-request type-refinement threshold (default server value `0.7`) |
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
    },
    "plus": {
      "model": "gliner2.5-multi-Decide",
      "backend": "PyTorch CPU",
      "decisions": 1,
      "forward_passes": 1,
      "elapsed_ms": 65.0,
      "policy_threshold": 0.5,
      "policy_action": "全部打码",
      "action_confidence": 0.99,
      "type_refined": 1,
      "type_refine_forward_passes": 1,
      "type_refine_threshold": 0.7
    }
  }
}
```

`usage.flash.chunks` and `forward_passes` are estimates based on the configured
word splitter and chunking parameters. `usage.plus.type_refined` is the number
of auto spans whose type changed; `type_refine_forward_passes` is the number of
type-classification forwards. `type_refine_threshold` reports the effective
per-request threshold.

### Differences from AnyPII

- `mode` is `"flash"` or `"flash-plus"`; other modes return `400`.
- `flash` requires explicit `redact:true/false`.
- `flash-plus` allows `redact:"auto"` or omitted `redact`; a non-empty
  `state.policy` is required for auto types. Type refinement is on by default;
  set `type_refine:false` to disable it, or override
  `type_refine_threshold` per request.
- `policy` is ignored for extraction; it is used only by the flash-plus
  decision stage.
- `parrot_typed` is rejected.
- Only the `flash` and `flash-plus` inference paths are exposed.

## Configuration

Environment variables:

```text
ANYPPI_FLASH_MODEL                 extraction model id or local directory
ANYPPI_FLASH_PLUS_MODEL            Decide model id or local directory
ANYPPI_FLASH_DEVICE                cpu, cuda, cuda:0, mps
ANYPPI_FLASH_SPLITTER              char (default) or whitespace
ANYPPI_FLASH_CHUNK_SIZE            384 words
ANYPPI_FLASH_CHUNK_OVERLAP         64 words
ANYPPI_FLASH_MAX_TYPES             25
ANYPPI_FLASH_MAX_TEXT_CP           16000
ANYPPI_FLASH_MAX_PAYLOAD_BYTES     2097152
ANYPPI_FLASH_THRESHOLD             0.55
ANYPPI_FLASH_POLICY_THRESHOLD      0.5
ANYPPI_FLASH_POLICY_ACTION_THRESHOLD 0.8
ANYPPI_FLASH_TYPE_REFINE_THRESHOLD 0.7
ANYPPI_FLASH_PLUS_BATCH_SIZE      8
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
       flash: explicit only
       flash-plus: per-span type refine + global action + per-span Decide
  -> token / length / type / typed mask
  -> response
```

For `flash-plus`, auto spans first go through optional type refinement against
all request types. The decision stage then receives the original text plus the
policy, checks for a global all-keep/all-redact instruction above
`policy_action_threshold`, and otherwise classifies each auto span separately
using the span text, local context, type description, and policy. The span is redacted when its score clears
`redact_threshold`. Extracted spans are never replaced by generated text.

The extraction model is loaded once per process. The Decide model is loaded
lazily on the first flash-plus request that contains an auto type. Requests are
serialized around each PyTorch model; the result cache remains safe for
concurrent HTTP requests. In a mixed flash-plus request, each auto span costs a
type-classification forward and a redact/keep forward, so long documents are
slower than `flash`.


## Performance

Reference machine: **AMD Ryzen 9 7940HX with Radeon Graphics**, 16 physical
cores (32 threads; this environment allocates 16 logical CPUs), 48 GiB RAM. Both
services used 12 CPU threads. Both models were fully loaded before timing; all
entries are steady state. Result caches were disabled, prefix caches were
enabled, and timings are the shortest client wall-clock time observed across
repeated requests.

| request | AnyPII + Qwen3.5-0.8B Q8_0 | AnyPII Flash, `flash` | AnyPII Flash, `flash-plus` |
| --- | ---: | ---: | ---: |
| `medical.json` | 2.87s | 0.10s | 0.96s |
| `medical-long.json` | 8.32s | 1.71s | 7.76s |
| `contact.json` | <1ms | 0.08s | 0.29s |

`flash` is 28.4x faster on the short medical request and 4.9x faster on the long
one. `flash-plus` remains faster than AnyPII on the long request, but its
type-refinement and per-span decision stages add cost. `contact.json` is an
exception: AnyPII resolves it through its format fast path in under 1 ms.

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

Optional flash-plus smoke test:

```sh
ANYPPI_FLASH_TEST_MODEL=models/gliner2.5-multi-v1 \
ANYPPI_FLASH_TEST_PLUS_MODEL=models/gliner2.5-multi-Decide \
  PYTHONPATH=src .venv/bin/pytest tests/test_plus_model.py
```

## Limits

- GLiNER2.5 may return shorter spans than AnyPII. For example, it can return
  `林婉清` instead of `主治医师林婉清`.
- Structured PII such as phone numbers, national IDs, bank cards, and API keys
  relies entirely on GLiNER2.5.
- Long Chinese documents use character-level chunking; latency grows with the
  number of chunks.
- `flash-plus` is experimental. Decide classifies each extracted span
  independently, but zero-shot policy decisions remain less reliable than the
  explicit `flash` mode. A mixed-policy request adds type-refinement and
  redact/keep forwards per auto span.
- Type refinement is heuristic: a new type is accepted only above
  `type_refine_threshold`; it can still change a correct extraction.
- Policy classification is zero-shot. Check `redact_threshold` and your policy
  wording on your own data before relying on auto decisions.
- Model weights are separate artifacts. Verify the model cards and licenses
  before redistribution.

## License

MIT for this project. GLiNER2 is Apache-2.0; model weights keep their own terms.
