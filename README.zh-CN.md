# AnyPII Flash

[English](README.md) | 简体中文

**基于 GLiNER2.5 的 Schema 驱动的更快的任意自定义 PII 脱敏引擎。**

AnyPII Flash 是一个独立的 Python 本地脱敏服务。它使用
[`fastino/gliner2.5-multi-v1`](https://huggingface.co/fastino/gliner2.5-multi-v1)
按请求中的类型定义抽取实体，再用确定性的打码逻辑返回脱敏文本和结构化片段。

提供两种模式：

- **`flash`**：使用显式 `redact: true/false`，速度最快，行为确定。
- **`flash-plus`**：可选策略模式。抽取完成后，对每个 `redact:"auto"` 的片段
  先用
  [`fastino/GLiNER2.5-multi-Decide`](https://huggingface.co/fastino/GLiNER2.5-multi-Decide)
  做可选的类型精修，再结合原文、局部上下文和自然语言策略做独立打码判断。
  该模式下类型精修默认开启。

## 安装

建议使用 Python 3.12。本地推理需要 GLiNER2.5、PyTorch 和 Transformers。

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e ".[model,dev]"
```

下载模型：

```sh
.venv/bin/python scripts/download_model.py \
  --endpoint https://hf-mirror.com \
  --out models/gliner2.5-multi-v1

.venv/bin/python scripts/download_model.py \
  --endpoint https://hf-mirror.com \
  --model fastino/GLiNER2.5-multi-Decide \
  --out models/gliner2.5-multi-Decide
```

只有在需要镜像时使用 `--endpoint`；不传时使用 Hugging Face 默认地址。模型权重
目录已加入 `.gitignore`，不会进入版本库。

## 启动

```sh
ANYPPI_FLASH_MODEL=models/gliner2.5-multi-v1 \
ANYPPI_FLASH_PLUS_MODEL=models/gliner2.5-multi-Decide \
  .venv/bin/anypii-flash --port 8080
```

抽取模型在启动阶段加载并 warmup；Decide 模型在第一次需要 `flash-plus` 自动判定时
加载。`/health` 在抽取模型就绪后返回 `ready: true`，并报告 `plus_ready`。

### flash 示例

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

`POST /v1/anypii` 与 `POST /v1/flash` 使用同一处理逻辑。

### flash-plus 示例

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

Decide 的输入同时包含原文和策略。默认开启类型精修时，每个 auto 片段先针对请求
中的类型做一次独立分类；随后打码阶段判断策略是否为全局的“全部保留”或“全部打码”。
如果是混合策略，再对每个片段分别做打码/保留分类。全局“全部保留/全部打码”需要
达到 `policy_action_threshold`（默认 `0.8`）。响应中的 `usage.plus` 和每个 auto
片段的 `redact_score` 可查看判定结果。

`examples/` 目录包含与 AnyPII 对应的七个示例。服务启动后可运行：

```sh
python benchmarks/run_examples.py \
  --url http://127.0.0.1:8080/v1/anypii \
  --examples examples
```

## API

请求字段：

| 字段 | 含义 |
| --- | --- |
| `state.text` | 输入文本 |
| `state.policy` | 自然语言策略；`flash-plus` 的 auto 类型必须提供 |
| `types` | 类型 id 到类型定义的映射 |
| `types.*.description` | 传给 GLiNER2 的语义说明 |
| `types.*.redact` | `true`/`false`；`auto` 需要 `mode:"flash-plus"` |
| `threshold` | GLiNER 抽取置信度阈值，默认 `0.7` |
| `redact_threshold` | `flash-plus` 打码决策阈值，默认使用服务端 `0.5` |
| `mode` | `flash`（默认）或 `flash-plus` |
| `type_refine` | 可选；`flash-plus` 下是否对 auto 片段做类型精修，默认 `true` |
| `type_refine_threshold` | 可选；按请求覆盖类型精修阈值，默认使用服务端 `0.7` |
| `mask_style` | `token`、`length`、`mask`、`type` 或 `typed` |
| `skip_keep_types` | 省略保留片段；所有类型都是保留类型时不调用模型 |
| `omit_keep_spans` | 从响应中省略 `redacted:false` 片段，原文不变 |

响应字段：

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

`usage.flash.chunks` 和 `forward_passes` 是按 splitter 和分块参数估算的值。
`usage.plus.decisions` 是实际进入打码决策的片段数；`type_refined` 是类型发生变化的
auto 片段数；`type_refine_forward_passes` 是类型精修的前向次数；
`type_refine_threshold` 是本次请求实际使用的阈值。

### 与 AnyPII 的差异

- `mode` 只支持 `flash` 和 `flash-plus`。
- `flash` 必须显式给出 `redact:true/false`。
- `flash-plus` 允许 `redact:"auto"` 或省略 `redact`，但 auto 类型必须提供非空
  `state.policy`。类型精修默认开启；可用 `type_refine:false` 关闭，或用
  `type_refine_threshold` 按请求覆盖阈值。
- `policy` 不参与抽取，只参与 `flash-plus` 的决策阶段。
- `parrot_typed` 会被拒绝。
- 只提供 `flash` 与 `flash-plus` 两条推理路径。

## 配置

环境变量：

```text
ANYPPI_FLASH_MODEL                 抽取模型 id 或本地目录
ANYPPI_FLASH_PLUS_MODEL            Decide 模型 id 或本地目录
ANYPPI_FLASH_DEVICE                cpu、cuda、cuda:0、mps
ANYPPI_FLASH_SPLITTER              char（默认）或 whitespace
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
ANYPPI_FLASH_API_KEY               可选 Bearer token
ANYPPI_FLASH_RESULT_CACHE_ENTRIES  64
ANYPPI_FLASH_RESULT_CACHE_BYTES    67108864
ANYPPI_FLASH_WARMUP                1
ANYPPI_FLASH_LOG_LEVEL             info
```

命令行参数会覆盖对应环境变量。完整列表见 `anypii-flash --help`。

## 实现

```text
text + types
  -> GLiNER2 schema
  -> AutoExtractor
       boundary 架构
       char word splitter
       短文本：extract_entities
       长文本：extract_entities_long
  -> 归一化片段
       裁剪标点
       消除重叠
       合并重复出现
  -> 打码判定
       flash：只使用显式 redact
       flash-plus：类型精修 + 全局动作 + 每个 auto 片段独立 Decide
  -> token / length / type / typed 遮罩
  -> 响应
```

`flash-plus` 的决策输入包含原文；auto 片段先按请求中的全部类型做可选类型精修，
再由 Decide 检查全局的“全部保留/全部打码”（需高于 `policy_action_threshold`），
并对混合策略逐个片段做打码/保留判定。
策略、mention、局部上下文和类型说明放在 classification prompt 中。片段分数达到
`redact_threshold` 时打码。抽取片段不会由模型重写。

抽取模型每个进程只加载一次；Decide 模型按需加载。模型推理按进程串行，结果缓存
对并发 HTTP 请求安全。混合 `flash-plus` 请求会为每个 auto 片段增加类型精修和
打码/保留两次分类前向，因此长文本明显慢于 `flash`。


## 性能

测试机器：**AMD Ryzen 9 7940HX with Radeon Graphics**，16 个物理核心（32 线程；
当前环境分配 16 个逻辑 CPU），48 GiB 内存。两个服务都限制为 12 个 CPU 线程。
两个模型都在计时前完成加载，表中全部为稳态结果；结果缓存关闭，前缀缓存开启，
耗时为客户端 wall time 的重复请求最短用时。

| 请求 | AnyPII + Qwen3.5-0.8B Q8_0 | AnyPII Flash，`flash` | AnyPII Flash，`flash-plus` |
| --- | ---: | ---: | ---: |
| `medical.json` | 2.87s | 0.10s | 0.96s |
| `medical-long.json` | 8.32s | 1.71s | 7.76s |
| `contact.json` | <1ms | 0.08s | 0.29s |

`flash` 在短医疗请求上快 28.4 倍，在长医疗请求上快 4.9 倍。`flash-plus` 在长
请求上仍快于 AnyPII，但类型精修和逐片段决策会增加开销。`contact.json` 是例外：
AnyPII 通过格式快速路径在 1ms 内完成。

## 测试

```sh
PYTHONPATH=src .venv/bin/pytest
```

默认测试使用 fake extractor，不下载模型，覆盖 API、mask、缓存、错误和
skip/omit 行为。

可选的真实抽取模型测试：

```sh
ANYPPI_FLASH_TEST_MODEL=models/gliner2.5-multi-v1 \
  PYTHONPATH=src .venv/bin/pytest tests/test_model.py
```

可选的 flash-plus 测试：

```sh
ANYPPI_FLASH_TEST_MODEL=models/gliner2.5-multi-v1 \
ANYPPI_FLASH_TEST_PLUS_MODEL=models/gliner2.5-multi-Decide \
  PYTHONPATH=src .venv/bin/pytest tests/test_plus_model.py
```

## 限制

- GLiNER2.5 可能返回比 AnyPII 更短的片段，例如 `林婉清` 而不是
  `主治医师林婉清`。
- 电话、身份证、银行卡和 API key 等结构化 PII 完全依赖 GLiNER2.5。
- 长中文文档按字符分块，延迟随 chunk 数增长。
- `flash-plus` 是实验性功能。Decide 会逐个片段判断，但零样本自然语言策略仍不如
  显式 `flash` 稳定；混合策略下每个 auto 片段增加类型精修和打码/保留两次前向。
- 类型精修是启发式的：只有超过 `type_refine_threshold` 才接受新类型，仍可能把原本
  正确的类型改错。
- 策略分类是零样本能力。上线前应在自己的数据上检查 `redact_threshold` 和策略写法。
- 模型权重是独立产物，分发前请确认模型卡的许可条款。

## 许可证

本项目使用 MIT。GLiNER2 使用 Apache-2.0；模型权重保留其各自的许可条款。
