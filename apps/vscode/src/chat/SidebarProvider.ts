/**
 * Chat sidebar — the ForgeCoder activity-bar view ("do anything" chat).
 *
 * Implements vscode.WebviewViewProvider so ForgeCoder appears as a normal
 * sidebar panel: an icon in the activity bar, a persistent chat view with
 * streaming responses, and Apply Fix / View Diff patch actions.
 */
import * as vscode from 'vscode';
import { ForgeApi, FilePatch, MultiPatchFile, ChatTurn, ActResponse, GitChangesResponse,
  PlanResponse } from '../client/api';
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
  private pendingMultiPatch: { workspace: string; patches: FilePatch[]; creates: Record<string, string>; message: string } | undefined;
  private currentPlan: { planId: string; steps: Array<{ title: string; action: string; done?: boolean }> } | undefined;
  private history: Array<{ role: string; content: string }> = [];
  private streaming = false;
  private planMode = false;
  private lastMessage: string = '';

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
      case 'applyMultiPatch':
        await this.confirmApplyMulti();
        break;
      case 'viewPatches':
        await this.previewPatches();
        break;
      case 'viewMultiPatches':
        await this.previewMultiPatches();
        break;
      case 'createFile':
        await this.createFile(String(msg.path ?? ''), String(msg.content ?? ''));
        break;
      case 'clear':
        this.history = [];
        this.planMode = false;
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
                           msg.action === 'test' || msg.action === 'commit',
                           Boolean(msg.confirmed));
        break;
      case 'actAll':
        await this.runAllSteps(String(msg.planId ?? ''), Number(msg.count ?? 0));
        break;
      case 'togglePlanMode':
        this.togglePlanMode();
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
    this.currentPlan = { planId: res.plan_id,
                         steps: res.steps.map((s) => ({ title: s.title, action: s.action })) };
  }

  private async runAllSteps(planId: string, count: number): Promise<void> {
    for (let i = 0; i < count; i++) {
      const step = this.currentPlan?.steps[i];
      if (!step || step.done) { continue; }
      const mutating = step.action === 'test' || step.action === 'commit';
      let confirmed = false;
      if (mutating) {
        const answer = await vscode.window.showWarningMessage(
          `Plan step ${i + 1} (${step.action}) wants to ${step.action === 'commit' ? 'modify git state' : 'run code'}. Allow once?`,
          { modal: true }, 'Allow once', 'Skip step',
        );
        if (answer !== 'Allow once') { continue; }
        confirmed = true;
      }
      await this.runStep(planId, i, mutating, confirmed);
      if (this.pendingMultiPatch) {
        await this.confirmApplyMulti(true);
      }
    }
  }

  private async runStep(planId: string, index: number, mutating: boolean, confirmed = false): Promise<void> {
    if (index < 0) {
      return;
    }
    const workspace = currentWorkspace();
    this.reveal();
    this.post('stepStart', { index });
    const res = await this.api.act(planId, index, workspace, mutating && !confirmed)
      .catch((err) => ({ ok: false, error: err instanceof Error ? err.message : String(err),
                         step_index: index, title: `Step ${index + 1}`, action: 'unknown',
                         output: '', next_index: null } as ActResponse));
    this.post('stepDone', {
      index: res.step_index ?? index,
      ok: res.ok === true,
      title: res.title ?? `Step ${index + 1}`,
      output: res.output ?? res.error ?? '',
      hasPatch: Boolean(res.patch || res.multiPatch || (res.creates && Object.keys(res.creates).length)),
    });
    if (this.currentPlan?.steps[index]) { this.currentPlan.steps[index].done = true; }
    if (res.patch) {
      this.offerPatch(workspace ?? '', [res.patch]);
    } else if ((res.multiPatch && res.multiPatch.length) || (res.creates && Object.keys(res.creates).length)) {
      this.pendingMultiPatch = { workspace: workspace ?? '', patches: (res.multiPatch ?? []).map((f) => ({
        path: f.path,
        operations: f.operations ?? [],
      })), creates: res.creates ?? {}, message: '' };
      this.pendingPatch = undefined;
      this.reveal();
      const createdPaths = Object.keys(res.creates ?? {});
      const editedPaths = (res.multiPatch ?? []).map((f) => f.path);
      this.post('pendingMultiPatch', {
        count: editedPaths.length + createdPaths.length,
        paths: editedPaths.concat(createdPaths),
      });
    }
  }

  private async send(message: string): Promise<void> {
    if (!message.trim()) {
      return;
    }
    this.history.push({ role: 'user', content: message });
    this.post('userMessage', { content: message });
    this.lastMessage = message;

    // Use the agent run endpoint - each function is its own agent
    // The agent returns actions that drive UI behavior (view, diff, apply, etc.)
    try {
      const rec = await this.api.runAgent(message);
      if (!rec?.ok) {
        this.post('error', { message: rec?.error ?? 'Unknown error' });
        return;
      }
      // Agent-driven actions: the agent decides what UI actions to offer
      await this.handleAgentResponse(rec);
    } catch (err) {
      this.post('error', { message: String(err) });
    }
  }

  /** Handle agent response - UI behavior driven by Recommendation.actions */
  private async handleAgentResponse(rec: any): Promise<void> {
    const actions = rec?.actions || [];
    const patches = rec?.patches || [];
    const multiPatches = rec?.multi_patches || null;
    const creates = rec?.creates || {};
    const steps = rec?.steps || null;

    // Stream the response text if present
    if (rec?.text || rec?.answer) {
      this.post('assistantStart', {});
      this.post('delta', { content: rec.text || rec.answer });
      this.post('assistantDone', {});
    }

    // Handle plan mode: agent returned steps
    if (actions.includes('plan') || (steps && steps.length > 0)) {
      if (steps && steps.length > 0) {
        this.currentPlan = {
          planId: 'agent-' + Date.now(),
          steps: steps.map((s: any, i: number) => ({
            title: s.title || s.action || `Step ${i + 1}`,
            action: s.action || 'edit',
            done: false
          }))
        };
        this.planMode = true;
        this.post('planMode', {});
        this.post('plan', this.currentPlan);
      }
      return;
    }

    // Handle multi-file patch action
    if (actions.includes('apply_multi') && multiPatches && multiPatches.length > 0) {
      this.pendingMultiPatch = {
        workspace: rec.workspace || currentWorkspace(),
        patches: multiPatches,
        creates,
        message: this.lastMessage || ''
      };
      this.post('pendingMultiPatch', { count: multiPatches.length });
      return;
    }

    // Handle single-file patch action
    if (actions.includes('apply') && patches.length > 0) {
      this.pendingPatch = {
        workspace: rec.workspace || currentWorkspace(),
        patch: patches[0]
      };
      this.post('pendingPatch', { path: patches[0].path });
      return;
    }

    // Handle view/diff action (show patches)
    if (actions.includes('view') || actions.includes('diff')) {
      if (patches.length > 0) {
        this.post('viewPatches', { patches });
        return;
      }
    }

    // Handle review action
    if (actions.includes('review')) {
      await this.reviewChanges();
      return;
    }

    // Handle test action
    if (actions.includes('test')) {
      try {
        const result = await this.api.runTests(rec.workspace || currentWorkspace());
        if (result.ok) {
          this.post('testResult', { output: result.output });
        }
      } catch (e) {
        // Fall through to normal response handling
      }
    }

    // Default: just show the response
    this.post('assistantDone', {});
  }

  /** Store a pending patch produced by a code action and surface it in the sidebar. */
  offerPatch(workspace: string, patches: FilePatch[]): void {
    if (patches.length > 0) {
      if (patches.length === 1) {
        this.pendingPatch = { workspace, patch: patches[0] };
        this.reveal();
        this.post('pendingPatch', { path: patches[0].path });
      } else {
        // Multi-file patch: store for bulk preview/apply.
        this.pendingMultiPatch = { workspace, patches, creates: {}, message: '' };
        this.pendingPatch = undefined;
        this.reveal();
        this.post('pendingMultiPatch', { count: patches.length, paths: patches.map((p) => p.path) });
      }
    }
  }

  hasPendingPatch(): boolean {
    return this.pendingPatch !== undefined || this.pendingMultiPatch !== undefined;
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

  togglePlanMode(): void {
    this.planMode = !this.planMode;
    this.post('planMode', {});
    this.post('notice', {
      message: this.planMode
        ? 'Plan & Act mode ON — your next message becomes a plan with runnable steps.'
        : 'Plan & Act mode off.',
    });
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
    this.planMode = false;
    this.post('cleared', {});
  }


  private async confirmApply(autoConfirmed = false): Promise<void> {
    const pending = this.pendingPatch;
    if (!pending) {
      void vscode.window.showWarningMessage('No pending patch — ask ForgeCoder for a fix first.');
      return;
    }
    const allowWrite = vscode.workspace.getConfiguration('forgecoder').get<boolean>('allowWriteFile', false);
    if (!autoConfirmed && !allowWrite) {
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

  /** Preview a multi-file patch set in VS Code's diff editor. */
  private async previewMultiPatches(): Promise<void> {
    const pending = this.pendingMultiPatch;
    if (!pending) {
      return;
    }
    const res = await this.api.multiPatchPreview(pending.workspace, pending.patches, pending.creates, pending.message);
    if (!res.ok || !res.files?.length) {
      void vscode.window.showErrorMessage(String(res.error ?? 'Preview failed'));
      return;
    }
    // Show the first changed file in the diff editor, then reveal remaining via notification.
    const first = res.files.find((f) => f.changed && f.diff);
    if (first && first.original != null && first.proposed != null) {
      await showPatchPreview(pending.workspace, {
        path: first.path,
        operations: (first.operations ?? []).filter((o) => o.type !== 'delete' || o.content),
      }, this.api);
    } else {
      void vscode.window.showInformationMessage(`ForgeCoder: ${res.files.length} file(s) changed.`);
    }
    void vscode.window.showInformationMessage(`Previewed ${res.files.length} file(s) — ${res.files.filter((f) => f.created).length} new.`);
  }

  /** Apply a multi-file patch set atomically (rollback on failure). */
  private async confirmApplyMulti(autoConfirmed = false): Promise<void> {
    const pending = this.pendingMultiPatch;
    if (!pending) {
      void vscode.window.showWarningMessage('No pending multi-file patch.');
      return;
    }
    const allowMulti = autoConfirmed ||
      vscode.workspace.getConfiguration('forgecoder').get<boolean>('allowWriteFile', false);
    if (!allowMulti) {
      const changed = pending.patches.filter((p) => p.operations.some((o) => o.type !== 'delete'));
      const created = Object.keys(pending.creates);
      const summary = [...changed.map((p) => p.path), ...created]
        .map((p) => `  • ${p}`)
        .join('\n');
      const answer = await vscode.window.showWarningMessage(
        `Apply ${changed.length} edit(s) and ${created.length} new file(s)?\n\n${summary}`,
        { modal: true },
        'Apply All',
        'Cancel',
      );
      if (answer !== 'Apply All') {
        return;
      }
    }
    const result = await this.api.multiPatchApply(pending.workspace, pending.patches, pending.creates, pending.message, true);
    if (!result.ok) {
      void vscode.window.showErrorMessage(String(result.error ?? 'Multi-file apply failed'));
      return;
    }
    void vscode.window.showInformationMessage(`Applied ${result.applied?.length ?? 0} file(s).`);
    this.post('multiApplied', { files: result.applied ?? [] });
  }

  /** Create a brand-new file in the workspace after a confirmation prompt. */
  private async createFile(path: string, content: string): Promise<void> {
    const workspace = currentWorkspace();
    if (!workspace) {
      this.post('error', { message: 'Open a workspace folder first.' });
      return;
    }
    const allow = vscode.workspace.getConfiguration('forgecoder').get<boolean>('allowWriteFile', false);
    if (!allow) {
      const answer = await vscode.window.showWarningMessage(
        `Create new file ${path}?`,
        { modal: true },
        'Create',
        'Cancel',
      );
      if (answer !== 'Create') {
        return;
      }
    }
    const full = vscode.Uri.file(workspace + '\\' + path.replace(/\//g, '\\'));
    await vscode.workspace.fs.writeFile(full, new TextEncoder().encode(content));
    void vscode.window.showInformationMessage(`Created ${path}`);
    this.post('fileCreated', { path });
    // Open the new file in the editor.
    const doc = await vscode.workspace.openTextDocument(full);
    await vscode.window.showTextDocument(doc);
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
    <button id="viewMultiDiff">View All</button>
    <button id="applyMulti">Apply All</button>
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