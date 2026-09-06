# Security

ForgeCoder is a **local-first** tool. The default posture is that nothing ever
leaves your machine unless you explicitly enable it.

## Design guarantees

- **Loopback only.** The Forge API (`127.0.0.1:8787`) and llama.cpp server
  (`127.0.0.1:8080`) bind to localhost by default. Never `0.0.0.0`.
- **No telemetry.** ForgeCoder does not phone home.
- **No cloud model.** Inference runs entirely through local `llama-server`.
- **No automatic uploads.** Prompts, files, and repository context are never
  sent anywhere.
- **No arbitrary shell execution.** The server never executes commands from
  model output.

## Permission model

Capabilities are gated through an explicit permission set. v0.1 ships with
read-only retrieval and prompts for any file-mutating action.

| Permission   | v0.1       |
| ------------ | ---------- |
| `READ_FILE`  | allowed    |
| `SEARCH`     | allowed    |
| `GIT_READ`   | allowed    |
| `WRITE_FILE` | require confirmation |
| `BUILD`      | require confirmation |
| `TEST`       | require confirmation |
| `SHELL`      | require confirmation |

Patches are always presented in a **diff preview** first. ForgeCoder never
silently overwrites a source file.

## Reporting a vulnerability

Please report security issues privately to the maintainers by opening a
GitHub Security Advisory, or by creating an issue with a `security` label
that **does not contain exploit details**. Do not open public issues with
working exploits.