/** Diff preview — always show Original | Proposed before any write. */
import * as vscode from 'vscode';
import { ForgeApi, FilePatch, PatchPreviewResponse } from '../client/api';

export async function showPatchPreview(workspace: string, patch: FilePatch, api: ForgeApi): Promise<void> {
  const preview = await api.patchPreview(workspace, patch);

  const left = await openDoc(patch.path, preview, true);
  const right = await openDoc(patch.path, preview, false);
  if (!left || !right) {
    void vscode.window.showErrorMessage('Could not open the diff editor.');
    return;
  }
  await vscode.commands.executeCommand('vscode.diff', left.uri, right.uri, `ForgeCoder: ${patch.path}`, {
    preview: true,
  });
}

async function openDoc(
  path: string,
  preview: PatchPreviewResponse,
  isOriginal: boolean,
): Promise<vscode.TextDocument | undefined> {
  const label = `${path} ${isOriginal ? '(original)' : '(proposed)'}`;
  const doc = await vscode.workspace.openTextDocument({
    language: detectLanguage(path),
    content: isOriginal ? preview.original ?? '' : preview.proposed ?? '',
  });
  // Rename the in-memory document so the diff header reads nicely.
  void doc;
  return doc;
}

function detectLanguage(path: string): string {
  const ext = path.split('.').pop()?.toLowerCase() ?? '';
  const map: Record<string, string> = {
    ts: 'typescript', tsx: 'typescriptreact', js: 'javascript', jsx: 'javascriptreact',
    py: 'python', java: 'java', rs: 'rust', go: 'go', cs: 'csharp',
    cpp: 'cpp', cc: 'cpp', h: 'c', c: 'c', sql: 'sql', kt: 'kotlin',
  };
  return map[ext] ?? 'plaintext';
}