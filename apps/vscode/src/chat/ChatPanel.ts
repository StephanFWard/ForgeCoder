/** Chat panel — a VS Code WebView that streams tokens from the Forge server. */
import * as vscode from 'vscode';
import { ForgeApi, FilePatch, ChatTurn } from '../client/api';
import { showPatchPreview } from '../ui/diff';

interface PendingPatch {
  workspace: string;
  patch: FilePatch;
}

export class ChatPanel {
  private static instance: ChatPanel | undefined;
  private readonly panel: vscode.WebviewPanel;
  private readonly disposables: vscode.Disposable[] = [];
  private pendingPatch: PendingPatch | undefined;
  private history: Array<{ role: string; content: string }> = [];
  private streaming = false;

  static open(extensionUri: vscode.Uri, api: ForgeApi): ChatPanel {
    if (ChatPanel.instance) {
      ChatPanel.instance.panel.reveal(vscode.ViewColumn.Beside);
      return ChatPanel.instance;
    }
    ChatPanel.instance = new ChatPanel(extensionUri, api);
    return ChatPanel.instance;
  }

  private constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly api: ForgeApi,
  ) {
    this.panel = vscode.window.createWebviewPanel(
      'forgecoder.chat',
      'ForgeCoder',
      vscode.ViewColumn.Beside,
      { enableScripts: true, retainContextWhenHidden: true },
    );
    this.panel.webview.html = this.renderHtml();
    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
    this.panel.webview.onDidReceiveMessage(
      (msg) => void this.onMessage(msg),
      null,
      this.disposables,
    );
  }

  private async onMessage(msg: Record<string, unknown>): Promise<void> {
    switch (msg.command) {
      case 'send':
        await this.send(String(msg.message ?? ''));
        break;
      case 'applyPatch':
        await this.confirmApply();
        break;
      case 'viewPatches':
        await this.previewPatches();
        break;
      case 'clear':
        this.history = [];
        this.post('cleared', {});
        break;
    }
  }

  private async send(message: string): Promise<void> {
    if (this.streaming || !message.trim()) {
      return;
    }
    this.streaming = true;
    this.post('userMessage', { content: message });
    this.post('assistantStart', {});

    const editor = vscode.window.activeTextEditor;
    let turn: ChatTurn = { message, history: this.history };
    if (editor) {
      const workspace = vscode.workspace.getWorkspaceFolder(editor.document.uri)?.uri.fsPath;
      const file = workspace
        ? vscode.workspace.asRelativePath(editor.document.uri, false).replace(/\\/g, '/')
        : editor.document.uri.fsPath;
      let selection: { start: number; end: number } | undefined;
      if (!editor.selection.isEmpty) {
        selection = { start: editor.selection.start.line + 1, end: editor.selection.end.line + 1 };
      }
      turn = { message, workspace, file, selection, history: this.history };
    }

    let full = '';
    try {
      await this.api.chatStream(turn, (ev) => {
        if (ev.type === 'delta') {
          const delta = String(ev.content ?? '');
          full += delta;
          this.post('delta', { content: delta });
        } else if (ev.type === 'error') {
          this.post('error', { message: String(ev.message) });
        } else if (ev.type === 'done') {
          this.post('assistantDone', {});
        }
      });
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      this.post('error', {
        message: `Cannot reach the Forge server. Start it with:\npython -m uvicorn forge_server.main:app --host 127.0.0.1 --port 8787\n\n(${detail})`,
      });
      this.post('assistantDone', {});
    }

    this.history.push({ role: 'user', content: message });
    if (full) {
      this.history.push({ role: 'assistant', content: full });
    }
    this.history = this.history.slice(-12);
    this.streaming = false;
  }

  /** Store a pending patch produced by a code action and surface it to the user. */
  offerPatch(workspace: string, patches: FilePatch[]): void {
    if (patches.length > 0) {
      this.pendingPatch = { workspace, patch: patches[0] };
      this.post('pendingPatch', { path: patches[0].path });
    }
  }

  private async previewPatches(): Promise<void> {
    const pending = this.pendingPatch;
    if (!pending) {
      return;
    }
    await showPatchPreview(pending.workspace, pending.patch, this.api);
  }

  private async confirmApply(): Promise<void> {
    const pending = this.pendingPatch;
    if (!pending) {
      return;
    }
    const allow = vscode.workspace.getConfiguration('forgecoder').get<boolean>('allowWriteFile', false);
    if (!allow) {
      const answer = await vscode.window.showWarningMessage(
        `Apply changes to ${pending.patch.path}?`,
        { modal: true },
        'Apply',
        'Cancel',
      );
      if (answer !== 'Apply') {
        return;
      }
    }
    const result = await this.api.patchApply(pending.workspace, pending.patch, true);
    if (!result.ok) {
      void vscode.window.showErrorMessage(String(result.error ?? 'Apply failed'));
      return;
    }
    void vscode.window.showInformationMessage(`Applied ${pending.patch.path}`);
    this.post('applied', { path: pending.patch.path });
  }

  private post(command: string, payload: Record<string, unknown>): void {
    void this.panel.webview.postMessage({ command, ...payload });
  }

  private renderHtml(): string {
    const css = this.panel.webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'chat.css'));
    const js = this.panel.webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'chat.js'));
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="${css}">
<title>ForgeCoder</title>
</head>
<body>
<header><span class="logo">◆</span> ForgeCoder <button id="clear" title="Clear conversation">⌫</button></header>
<main id="messages"></main>
<footer>
  <input id="input" type="text" placeholder="Ask ForgeCoder..." autocomplete="off" spellcheck="false">
  <button id="send" title="Send">➤</button>
  <div id="patchBar" class="hidden">
    <span id="patchInfo"></span>
    <button id="viewDiff">View Diff</button>
    <button id="apply">Apply Fix</button>
  </div>
</footer>
<script src="${js}"></script>
</body>
</html>`;
  }

  private dispose(): void {
    ChatPanel.instance = undefined;
    this.disposables.forEach((d) => d.dispose());
  }
}