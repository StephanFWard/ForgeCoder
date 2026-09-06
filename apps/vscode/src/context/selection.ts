/** Gather editor selection & workspace context for API calls. */
import * as vscode from 'vscode';

export interface ActiveContext {
  workspace?: string;
  file?: string;
  selection?: { start: number; end: number };
  code: string;
}

export function activeFileContext(): ActiveContext {
  const editor = vscode.window.activeTextEditor;
  const empty: ActiveContext = { code: '' };
  if (!editor) {
    return empty;
  }
  const document = editor.document;
  const workspace = vscode.workspace.getWorkspaceFolder(document.uri);
  const fullText = document.getText();

  if (editor.selection.isEmpty) {
    return {
      workspace: workspace?.uri.fsPath,
      file: workspace ? getRelativePath(workspace, document.uri) : document.uri.fsPath,
      code: fullText.slice(0, 4000),
    };
  }

  const sel = editor.selection;
  // 1-based inclusive lines, matching the server's coordinate convention.
  const startLine = sel.start.line + 1;
  const endLine = sel.end.line + 1;
  let snippet = '';
  for (let i = sel.start.line; i <= sel.end.line; i += 1) {
    snippet += document.lineAt(i).text + '\n';
  }
  return {
    workspace: workspace?.uri.fsPath,
    file: workspace ? getRelativePath(workspace, document.uri) : document.uri.fsPath,
    selection: { start: startLine, end: endLine },
    code: snippet.trim(),
  };
}

function getRelativePath(workspace: vscode.WorkspaceFolder, uri: vscode.Uri): string {
  const rel = vscode.workspace.asRelativePath(uri, false).replace(/\\/g, '/');
  return rel;
}

export function currentWorkspace(): string | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

export function diagnosticSummary(): string {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return '';
  }
  const diags = vscode.languages.getDiagnostics(editor.document.uri);
  const lines = diags
    .slice(0, 20)
    .map((d) => `${d.severity === vscode.DiagnosticSeverity.Error ? 'error' : 'warning'}: ${d.message}`)
    .join('\n');
  return lines;
}