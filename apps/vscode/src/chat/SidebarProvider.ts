/**
 * Chat sidebar — the ForgeCoder activity-bar view ("do anything" chat).
 *
 * Implements vscode.WebviewViewProvider so ForgeCoder appears as a normal
 * sidebar panel: an icon in the activity bar, a persistent chat view with
 * streaming responses, and Apply Fix / View Diff patch actions.
 */
import * as vscode from 'vscode';
import { ForgeApi, FilePatch, ChatTurn, ActResponse, GitChangesResponse, PlanResponse } from '../client/api';
import { showPatchPreview } from '../ui/diff';
import { currentWorkspace } from '../context/selection';

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

      // ---- git powers ----
      case 'reviewChanges':
        await this.reviewChanges();
        break;
      case 'commit':
        await this.commitWithForge();
        break;
      case 'push':
        await this.pushToOrigin();
        break;

      // ---- plan -> act events ----
      case 'plan':
        await this.makePlan(String(msg.message ?? ''));
        break;
      case 'act':
        await this.runStep(String(msg.planId ?? ''), Number(msg.index ?? -1),
                           msg.action === 'test' || msg.action === 'commit');
        break;
    }
  }

  private async reviewChanges(): Promise<void> {
    const workspace = currentWorkspace();
    if (!workspace) {
      this.post('error', { message: 'Open a workspace folder first.' });
      return;
    }
    this.reveal();
    this.post('assistantStart', {});
    this.post('delta', { content: 'Reviewing working changes…' });
    const res = await this.api.gitChanges(workspace).catch((err) => ({
      ok: false, error: err instanceof Error ? err.message : String(err),
    } as GitChangesResponse));
    this.post('assistantDone', {});
    if (!res.ok) {
      this.post('error', { message: res.error ?? 'Review failed' });
      return;
    }
    this.post('changesReview', {
      branch: res.branch,
      clean: res.clean,
      count: res.entries?.length ?? 0,
      review: res.review,
      diff: res.diff,
    });
  }

  private async commitWithForge(): Promise<void> {
    const workspace = currentWorkspace();
    if (!workspace) {
      void vscode.window.showWarningMessage('Open a workspace folder first.');
      return;
    }
    const message = await vscode.window.showInputBox({
      prompt: 'Commit message',
      placeHolder: 'ForgeCoder: describe the change',
      ignoreFocusOut: true,
    });
    if (!message) {
      return;
    }
    const res = await this.api.gitCommit(workspace, message, true);
    if (!res.ok) {
      void vscode.window.showErrorMessage(`Commit failed: ${res.error}`);
      this.post('error', { message: `Commit failed: ${res.error}` });
      return;
    }
    void vscode.window.showInformationMessage(`Committed ${res.hash}: ${res.subject}`);
    this.post('notice', { message: `Committed ${res.hash}: ${res.subject} (${res.staged} files)` });
  }

  private async pushToOrigin(): Promise<void> {
    const workspace = currentWorkspace();
    if (!workspace) {
      void vscode.window.showWarningMessage('Open a workspace folder first.');
      return;
    }
    const answer = await vscode.window.showWarningMessage(
      'Push the current branch to origin?', { modal: true }, 'Push',
    );
    if (answer !== 'Push') {
      return;
    }
    const res = await this.api.gitPush(workspace, true);
    if (!res.ok) {
      void vscode.window.showErrorMessage(`Push failed: ${res.error}`);
      this.post('error', { message: `Push failed: ${res.error}` });
      return;
    }
    this.post('notice', { message: `Pushed ${res.branch} to ${res.remote}` });
  }

  // ------------------------------------------------------- plan -> act events
  private async makePlan(message: string): Promise<void> {
    if (!message.trim()) {
      return;
    }
    const workspace = currentWorkspace();
    this.reveal();
    this.post('assistantStart', {});
    this.post('delta', { content: 'Planning…' });
    const res = await this.api.plan(message, workspace).catch((err) => ({
      ok: false, error: err instanceof Error ? err.message : String(err),
    } as PlanResponse));
    this.post('assistantDone', {});
    if (!res.ok) {
      this.post('error', { message: res.error ?? 'Planning failed' });
      return;
    }
    this.post('plan', { planId: res.plan_id, summary: res.summary, steps: res.steps,
                        fallback: res.fallback === true });
  }

  private async runStep(planId: string, index: number, mutating: boolean): Promise<void> {
    if (index < 0) {
      return;
    }
    const workspace = currentWorkspace();
    this.reveal();
    this.post('stepStart', { index });
    const res = await this.api.act(planId, index, workspace, mutating)
      .catch((err) => ({ ok: false, error: err instanceof Error ? err.message : String(err),
                         step_index: index, title: `Step ${index + 1}`, action: 'unknown',
                         output: '', next_index: null } as ActResponse));
    this.post('stepDone', {
      index: res.step_index ?? index,
      ok: res.ok === true,
      title: res.title ?? `Step ${index + 1}`,
      output: res.output ?? res.error ?? '',
      hasPatch: Boolean(res.patch),
    });
    if (res.patch) {
      this.offerPatch(workspace ?? '', [res.patch]);
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

  /** Public entry points for the command palette / view title menus. */
  async startReview(): Promise<void> {
    await this.reviewChanges();
  }

  async startCommit(): Promise<void> {
    await this.commitWithForge();
  }

  async startPush(): Promise<void> {
    await this.pushToOrigin();
  }

  enablePlanMode(): void {
    this.reveal();
    this.post('planMode', {});
    this.post('notice', { message: 'Plan & Act mode ON — type a goal to get runnable steps.' });
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
<header><span class="logo">&#9670;</span> ForgeCoder
  <button id="review" title="Review working changes (git diff + AI review)">&#8635;</button>
  <button id="commitBtn" title="Commit all changes">&#10003;</button>
  <button id="planBtn" title="Plan &amp; Act mode: turn your next message into executable steps">&#9654;</button>
  <button id="clear" title="Clear conversation">&#9000;</button>
</header>
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
