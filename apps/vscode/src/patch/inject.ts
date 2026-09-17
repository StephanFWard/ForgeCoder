/**
 * Decide whether a model suggestion may be injected directly into the file it
 * pertains to, and whether that file is still the one the suggestion was built
 * against.
 *
 * Discipline borrowed from alibaba/open-code-review's suggestion -> unified-diff
 * path: a suggestion is only written when it resolves to a real target whose
 * coordinates (path + line ranges) still match the file on disk. Everything
 * here is pure logic with no VS Code imports so it can be unit-tested in the
 * existing suite; the caller performs the actual write through
 * /v1/patch/apply with X-Forge-Confirm, which re-checks the stale guard
 * server-side.
 */
import { FilePatch } from '../client/api';

export type InjectionPlan =
  | { kind: 'inject'; patch: FilePatch }
  | { kind: 'offer'; reason: string };

export interface InjectionInput {
  patches?: FilePatch[];
  /** Workspace-relative path of the open file, when there is one. */
  activeFile?: string;
  /** Workspace root; without it there is nothing to write into. */
  workspace?: string;
}

export function normalizeLf(text: string): string {
  return text.replace(/\r\n/g, '\n');
}

export function planInjection(input: InjectionInput): InjectionPlan {
  const patches = input.patches ?? [];
  if (!patches.length) {
    return { kind: 'offer', reason: 'No structured patch was produced.' };
  }
  if (!input.workspace) {
    return { kind: 'offer', reason: 'Open a workspace folder to apply an edit to a file.' };
  }
  if (!input.activeFile) {
    return { kind: 'offer', reason: 'Open the file this suggestion targets to apply it directly.' };
  }
  if (patches.length > 1) {
    return { kind: 'offer', reason: 'This change spans several files; review it in the patch preview workflow.' };
  }
  const patch = patches[0];
  if (patch.path !== input.activeFile) {
    return {
      kind: 'offer',
      reason: `The suggestion targets ${patch.path}, not the open file; review it in the patch preview workflow.`,
    };
  }
  return { kind: 'inject', patch };
}

export interface AnchorCheck {
  ok: boolean;
  reason?: string;
}

/**
 * Verify the suggestion still applies to the open, unmodified file.
 *
 * `previewOriginal` is what the server read for the patch as it will be
 * applied; if that differs from the editor's own text, the file moved on
 * between generation and now and the edit must be regenerated rather than
 * guessed at.
 */
export function checkAnchors(input: {
  patchPath: string;
  activeFile?: string;
  previewOriginal?: string;
  documentText: string;
}): AnchorCheck {
  if (!input.activeFile || input.patchPath !== input.activeFile) {
    return { ok: false, reason: 'The suggestion no longer targets the open file; regenerate the edit.' };
  }
  if (input.previewOriginal === undefined) {
    return { ok: false, reason: 'The preview did not return the original text; regenerate the edit.' };
  }
  if (input.previewOriginal !== normalizeLf(input.documentText)) {
    return { ok: false, reason: 'File changed since the suggestion was generated; regenerate the edit.' };
  }
  return { ok: true };
}