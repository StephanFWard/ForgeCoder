/**
 * Chat sidebar — the ForgeCoder activity-bar view ("do anything" chat).
 *
 * Implements vscode.WebviewViewProvider so ForgeCoder appears as a normal
 * sidebar panel: an icon in the activity bar, a persistent chat view with
 * streaming responses, and Apply Fix / View Diff patch actions.
 */
import * as vscode from 'vscode';
import { ForgeApi, FilePatch, ChatTurn } from '../client/api';
import { showPatchPreview } from '../ui/diff';

interface PendingPatch {
  workspace: string;
  patch: FilePatch;
}

export class SidebarProvider implements vscode.WebviewViewProvider {
  public static readonly viewId = 'forgecoder.sidebar';

  private view: vscode.WebviewView | undefined;
  private readonly disposables: vscode.Disposable[] = [];
  private pendingPatch: PendingPatch | undefined;
  private history: Array<{ role: string; content: string }> = [];
  private streaming = false;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly api: ForgeApi,
  ) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = { enableScripts: true, localResourceRoots: [this.extensionUri] };
    view.webview.html = this.renderHtml();

    view.onDidDispose(() => {
      this.view = undefined;
    }, null, this.disposables);

    view.webview.onDidReceiveMessage(
      (msg: Record<string, unknown>) => void this.onMessage(msg),
      null,
      this.disposables,
    );
  }

  /** Bring the sidebar to the front (used when a patch or message arrives). */
  reveal(): void {
    void this.view?.show?.(true);
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
        message: `Cannot reach the Forge server. Start it with:\npython runtime\\scripts\\start_forge.py\n\n(${detail})`,
      });
      this.post('assistantDone', {});
    }

    this.history.push({ role: 'user', content: message });
    if (full) {
      // Repetition guard: 1.5B models sometimes anchor on the previous turn.
      const previousAssistant = [...this.history].reverse().find((m) => m.role === 'assistant');
      if (previousAssistant && full.trim() === previousAssistant.content.trim()) {
        this.post('notice', {
          message: 'The model repeated its previous answer. Use Clear Chat or rephrase with more detail.',
        });
      }
      this.history.push({ role: 'assistant', content: full });
    }
    this.history = this.history.slice(-12);
    this.streaming = false;
  }
  /** Store a pending patch produced by a code action and surface it in the sidebar. */
  offerPatch(workspace: string, patches: FilePatch[]): void {
    if (patches.length > 0) {
      this.pendingPatch = { workspace, patch: patches[0] };
      this.reveal();
      this.post('pendingPatch', { path: patches[0].path });
    }
  }

  hasPendingPatch(): boolean {
    return this.pendingPatch !== undefined;
  }

  private async previewPatches(): Promise<void> {
    const pending = this.pendingPatch;
    if (!pending) {
      return;
    }
    await showPatchPreview(pending.workspace, pending.patch, this.api);
  }

  /** Public entry so the "Apply Patch" command can reuse the sidebar flow. */
  async applyPendingPatch(): Promise<void> {
    await this.confirmApply();
  }

  /** Clear history and reset the webview DOM (title-bar broom icon). */
  clearConversation(): void {
    this.history = [];
    this.post('cleared', {});
  }

  private async confirmApply(): Promise<void> {
    const pending = this.pendingPatch;
    if (!pending) {
      void vscode.window.showWarningMessage('No pending patch — ask ForgeCoder for a fix first.');
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
    void this.view?.webview.postMessage({ command, ...payload });
  }

  private renderHtml(): string {
    const webview = this.view!.webview;
    const css = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'chat.css'));
    const js = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'chat.js'));
    const nonce = Math.random().toString(36).slice(2);
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
<link rel="stylesheet" href="${css}">
<title>ForgeCoder</title>
</head>
<body>
<header><span class="logo">&#9670;</span> ForgeCoder <button id="clear" title="Clear conversation">&#9000;</button></header>
<main id="messages"></main>
<footer>
  <input id="input" type="text" placeholder="Ask ForgeCoder anything..." autocomplete="off" spellcheck="false">
  <button id="send" title="Send">&#10148;</button>
  <div id="patchBar" class="hidden">
    <span id="patchInfo"></span>
    <button id="viewDiff">View Diff</button>
    <button id="apply">Apply Fix</button>
  </div>
</footer>
<script nonce="${nonce}" src="${js}"></script>
</body>
</html>`;
  }

  dispose(): void {
    this.disposables.forEach((d) => d.dispose());
  }
}
