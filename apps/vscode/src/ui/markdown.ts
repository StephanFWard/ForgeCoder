/** Simple Markdown output panel for explanations / generated tests. */
import * as vscode from 'vscode';

export function showMarkdown(title: string, markdown: string): void {
  const panel = vscode.window.createWebviewPanel(
    'forgecoder.markdown',
    title,
    vscode.ViewColumn.One,
    { enableScripts: false },
  );
  const safe = markdown.replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>body{font-family:var(--vscode-font-family);padding:0 16px;white-space:pre-wrap;line-height:1.5}</style>
</head><body><pre>${safe}</pre></body></html>`;
  panel.webview.html = html;
}