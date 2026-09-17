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

`POST /v1/patch/apply` accepts optional `expected_original` (UTF-8 text with
normalized LF newlines). A mismatch rejects the write. This is an optimistic
stale-file check, not an OS-level lock; legacy callers omitting it retain their
previous behavior. Confirmation still requires `X-Forge-Confirm: true`.
The extension now forwards that header correctly.

Use local Forge/llama.cpp endpoints; no Supermemory account or network service
is used at runtime by these additions. Tests mock inference and do not establish
real-model quality gains or validate a live VS Code interaction.
