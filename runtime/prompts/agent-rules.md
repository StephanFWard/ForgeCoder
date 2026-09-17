# ForgeCoder agent rules

The router for the rules ForgeCoder's coding assistant is held to. Each rule is
one heading; the body carries `category`, `severity`, `check`, and a one-line
description. `check` names a function in `core/agent/rules.py` that the server
runs against a reply before the reply is shown, injected, or applied.

Rules are operational, not aspirational: a rule with no registered check is
reported as a stale rule and must be deleted or upgraded. Categories are
Startup, Forbidden, Definition of done, Uncertainty, and Approval; a rule that
fits none of them usually wants to be two rules.

Severity: `block` halts the action, `warn` is surfaced to the user, `info` is
recorded only.

The full rule set stays here. System prompts carry only the rules that apply to
the behavior being invoked (progressive disclosure), never this whole file.

### fresh-context-before-edit
category: Startup
severity: block
check: context_present
Source with line numbers must have been supplied for a target file; without it the model returns `files: []` and names the missing context.
A structured edit produced from no evidence is a guess, not an edit.

### allowed-paths-only
category: Forbidden
severity: block
check: scope_paths
Every patched path must match the task's allowed paths and match none of its forbidden paths.
The forbidden list always exists; `.git`, dependency trees, build output, lockfiles, and secrets are never patch targets.

### line-ranges-in-file
category: Forbidden
severity: block
check: lines_within_file
Every operation's line range must exist in the file that was supplied for it.
Line numbers refer to the original file, 1-based and inclusive, before any operation is applied.

### no-unrequested-files
category: Forbidden
severity: block
check: existing_target
A patch may only target a file that was verified to exist in the workspace.
New-file creation follows the dedicated create schema, not the edit schema.

### operations-well-formed
category: Definition of done
severity: block
check: ops_well_formed
`replace`/`insert` carry real code; `delete` carries an empty string; an `insert` uses `end_line == start_line`.
Content holds code only, never line numbers, diff prefixes, or placeholders for omitted code.

### operations-non-overlapping
category: Definition of done
severity: block
check: ops_non_overlapping
Operations in one file must not overlap; changes at the same location are combined into one operation.

### no-unverified-test-claims
category: Definition of done
severity: warn
check: no_unverified_claims
Never state that tests, linters, or builds passed unless they were executed and their output is quoted.
A proposal is described as a proposal; application and verification are reported only after they happen.

### ask-dont-guess
category: Uncertainty
severity: warn
check: declared_unknowns
Unresolved unknowns are named instead of being silently resolved by a plausible choice.
If the context cannot support the edit, reply with the specific missing information and an empty `files` list.

### writes-need-confirmation
category: Approval
severity: block
check: confirmed_write
No file is written without a preview, an explicit confirmation, and a matching stale-file anchor.
The model never applies its own patch; ForgeCoder's local endpoints are the only writer.