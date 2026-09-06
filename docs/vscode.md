# VS Code extension

## Features

- **Chat sidebar** — the ForgeCoder icon in the activity bar opens a persistent
  "do anything" chat view that streams tokens from the Forge server. Buttons
  appear for `Apply Fix` / `View Diff` when the server returns a structured
  patch; `ForgeCoder: Clear Chat` in the view title resets the conversation.
  `ForgeCoder: Chat` (or right-click → chat about a selection) focuses it.
- **Inline completion (autocomplete)** — separate, conservative pipeline.
  Uses Qwen FIM (`<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`),
  40 lines of context before, 20 after, default `max_tokens 64`,
  `temperature 0.1`.
- **Code actions** — `Explain`, `Fix`, `Refactor`, `Generate Tests` from the
  editor context menu.
- **Explain Error** — gathers current diagnostics + file + selection and asks
  the server for a diagnosis.
- **Search Workspace** — FTS5 repository search with jump-to-result.
- **Patch workflow** — everything that mutates code goes through
  `Original | Proposed` diff first; applying requires an explicit confirm
  (setting `forgecoder.allowWriteFile` skips the prompt for the current
  session's patches).
- **Status bar** — shows whether the Forge API and llama.cpp are reachable.

## Commands

| Command                       | ID                      |
| ----------------------------- | ----------------------- |
| ForgeCoder: Chat              | `forgecoder.chat`       |
| ForgeCoder: Explain Selection | `forgecoder.explain`    |
| ForgeCoder: Fix Selection     | `forgecoder.fix`        |
| ForgeCoder: Generate Tests    | `forgecoder.generateTests` |
| ForgeCoder: Refactor Selection| `forgecoder.refactor`   |
| ForgeCoder: Search Workspace  | `forgecoder.search`     |
| ForgeCoder: Explain Error     | `forgecoder.explainError` |
| ForgeCoder: Apply Patch       | `forgecoder.applyPatch` |
| ForgeCoder: Restart Server    | `forgecoder.restartServer` |

## Settings

| Key                                  | Default            |
| ------------------------------------ | ------------------ |
| `forgecoder.serverUrl`               | `http://127.0.0.1:8787` |
| `forgecoder.enableAutocomplete`      | `true`             |
| `forgecoder.autocompleteDelayMs`     | `400`              |
| `forgecoder.autocompleteMaxTokens`   | `64`               |
| `forgecoder.contextTokens`           | `4096`             |
| `forgecoder.allowWriteFile`          | `false`            |
| `forgecoder.autoIndexOnOpen`         | `true`             |

## Development

```powershell
cd apps\vscode
npm install
npm run compile      # builds out/ (src + test)
npm test             # launches VS Code test host
npm run package      # builds forgecoder-0.1.0.vsix
```

Press F5 in VS Code to launch an Extension Development Host.

## Architecture

```
media/chat.js  ←→  src/chat/SidebarProvider.ts  ←→  ForgeApi (client/api.ts)
                                                  │
                        HttpClient (client/httpClient.ts)
                                                  │  SSE / JSON
                                        Forge server :8787
```

- The WebView only posts typed messages (`send`, `clear`, `viewPatches`,
  `applyPatch`) and receives rendered diffs back from the host — the
  WebView never touches the filesystem.
- The completion provider aborts on cancellation and stays silent on any
  error (offline server = no UI noise).
- Indexing runs in the background when a folder opens and is throttled away
  during generation.