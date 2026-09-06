# Memory

ForgeCoder targets **6 GB total RAM** on Windows 11 + VS Code. These are hard
architecture rules, not suggestions.

## The ten rules

| # | Rule                                                       |
| - | ---------------------------------------------------------- |
| 1 | Only one model ever loaded.                                |
| 2 | No local embedding model initially.                        |
| 3 | No vector database initially.                              |
| 4 | SQLite only.                                               |
| 5 | 4096-token maximum chat context.                           |
| 6 | 1024-token maximum completion context.                     |
| 7 | No background model inference.                             |
| 8 | Index incrementally.                                       |
| 9 | Don't index generated/binary/vendor dirs.                  |
|10 | Unload/release unnecessary caches.                         |

## Budget breakdown (chat)

| Section                | Tokens |
| ---------------------- | ------ |
| System instructions    | ~300   |
| User request           | ~200   |
| Repository context     | ~1,500 |
| Relevant code          | ~1,000 |
| Conversation history   | ~500   |
| Safety / tool rules    | ~300   |
| **Target**             | ~3,800 |

Completion is even tighter: **512–1024 tokens** of context, generated
`max_tokens = 64`, `temperature = 0.1`.

## Hardware profile

`runtime/scripts/detect_hardware.py` samples RAM / CPU cores / GPU VRAM and
writes `%LOCALAPPDATA%\ForgeCoder\config.json`:

```json
{ "ram_gb": 6.0, "threads": 4, "context": 4096, "batch": 128,
  "gpu_layers": 0, "completion_context": 1024 }
```

Thread selection keeps a couple of cores free for the editor:
`threads = max(1, cores - 2)` (halved below 4 GB RAM). `gpu_layers` is
nonzero only when VRAM ≥ 4 GB.

## Memory monitor

`core/memory/monitor.py` samples:

- system memory pressure (`psutil`, with a Windows `GlobalMemoryStatusEx`
  fallback),
- model RSS (`llama-server` PID),
- server RSS.

Escalation ladder under pressure:

```
clear retrieval cache
  → disable background indexing
  → reduce batch
  → reduce context
```

Never let indexing compete aggressively with inference: indexing is paused
while a generation is in flight (`MemoryMonitor.should_throttle`).

## Indexing memory

- Files are read one at a time; hashes are compared to the DB to skip
  unchanged files.
- Chunks and symbols are inserted transactionally per file.
- Only supported source extensions are indexed; `node_modules`, `.git`,
  `dist`, `build`, `target`, `.venv`, binaries, and generated files are
  skipped.

## Measuring

`training/scripts/benchmark.py` reports tokens/sec per request shape. Do not
promise a fixed tokens/sec number — benchmark on the target hardware.