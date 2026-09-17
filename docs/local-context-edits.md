# Local context and direct edits

## Research basis

Reviewed https://github.com/supermemoryai/supermemory (README and CLAUDE.md)
and https://supermemory.ai/docs/concepts/memory-vs-rag.
Supermemory separates retrieved documents from evolving memory and emphasizes
compact, relevant context. Its published benchmark claims were not reproduced here.
ForgeCoder borrows the context-selection principle, not its hosted service or SDK.
No persistent fact extraction, memory graph, model training, or cloud dependency
was added by this change.

## Prompt context

The current saved file is placed before indexed search results. Selections retain
1-based original-file line numbers. Indexed copies of that active file are omitted
to avoid mixing stale and fresh evidence. Context fits whole source lines into an
estimated token allowance; token counts remain heuristic, not tokenizer-exact.
Other repository results remain ranked and deduplicated by the existing retriever.

## Direct local file application

In a trusted VS Code workspace, save the target file, optionally select source,
then run **ForgeCoder: Edit and Apply to Active File** from the Command Palette.
Enter the requested change, review the proposed diff, and choose **Apply**.
The command uses the existing local edit, preview, and patch-apply endpoints.
Only one patch targeting the active file is accepted. Unsaved/changed editors,
changed disk contents, other target files, and unconfirmed writes are refused.
Multi-file changes continue through the existing patch/Plan-Act workflow.

**Fix** and **Refactor** now take the same path when their structured patch
targets the open file: the suggestion is injected directly after the preview is
confirmed. When it targets another file, spans several files, or the context
does not identify an open file, the patch is offered to the sidebar review
workflow instead. The decision and the anchor checks live in
`apps/vscode/src/patch/inject.ts` (`planInjection`, `checkAnchors`), which is
pure logic with no VS Code imports and is unit-tested in
`apps/vscode/test/suite/inject.test.ts`:

- `planInjection` writes only for a single patch whose `path` equals the active
  file; everything else is handed to the review flow with a reason.
- `checkAnchors` requires the preview's `original` to equal the editor text
  (LF-normalized), so a file that moved on since generation is never written to
  by guesswork.
- The editor version is re-checked after the user confirms and before the
  write, and `expected_original` carries the same anchor to the server.

`POST /v1/patch/apply` accepts optional `expected_original` (UTF-8 text with
normalized LF newlines). A mismatch rejects the write. This is an optimistic
stale-file check, not an OS-level lock; legacy callers omitting it retain their
previous behavior. Confirmation still requires `X-Forge-Confirm: true`.
The extension now forwards that header correctly.

## Research basis — anchored suggestions

Reviewed https://github.com/alibaba/open-code-review (README, `src/` and the
`suggestdiff` path). It converts a model suggestion into a unified diff with
real file/line coordinates and then validates those coordinates before showing
them, which is what ForgeCoder now does in three places: `core/git/review.py`
(only added lines from the hunk headers can be referenced), the
`EDIT_SCHEMA`/`FIX_SCHEMA` grammar constraint (`core/patching/contracts.py`)
and `apps/vscode/src/patch/inject.ts` (anchored, confirmed, stale-guarded
local writes). OCR's `suggestdiff` only renders diffs — it does not apply them —
so ForgeCoder's existing preview → confirm → `PATCH /v1/patch/apply` path
remains the only writer, and no hosted service, API key, or network dependency
was added.

Use local Forge/llama.cpp endpoints; no Supermemory account or network service
is used at runtime by these additions. Tests mock inference and do not establish
real-model quality gains or validate a live VS Code interaction.
