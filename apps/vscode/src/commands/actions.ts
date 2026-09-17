/** Code-action implementations: explain / fix / refactor / tests / search / error. */
import * as vscode from 'vscode';
import { FilePatch, ForgeApi } from '../client/api';
import { SidebarProvider } from '../chat/SidebarProvider';
import { ActiveContext, activeFileContext, currentWorkspace, diagnosticSummary } from '../context/selection';
import { checkAnchors, normalizeLf, planInjection } from '../patch/inject';
import { showMarkdown } from '../ui/markdown';

interface InjectionOutcome {
  /** The file was written. */
  applied: boolean;
  /** The suggestion can still be reviewed in the sidebar patch workflow. */
  retryable: boolean;
}

/**
 * Inject a suggestion directly into the open file — local, confirmed, and
 * anchored: the patch must target the open file and the file must be
 * unchanged since the suggestion was generated.
 */
async function injectIntoActiveFile(
  api: ForgeApi,
  ctx: ActiveContext,
  patch: FilePatch,
  label: string,
): Promise<InjectionOutcome> {
  const editor = vscode.window.activeTextEditor;
  if (!vscode.workspace.isTrusted || !editor || !ctx.workspace || !ctx.file ||
      editor.document.uri.scheme !== 'file' || editor.document.isDirty) {
    return { applied: false, retryable: true };
  }
  const document = editor.document;
  const version = document.version;
  const original = normalizeLf(document.getText());
  const preview = await api.patchPreview(ctx.workspace, patch);
  const anchors = checkAnchors({
    patchPath: patch.path,
    activeFile: ctx.file,
    previewOriginal: preview.original,
    documentText: original,
  });
  if (!preview.ok || !anchors.ok) {
    void vscode.window.showWarningMessage(anchors.reason ?? preview.error ?? 'Preview failed.');
    return { applied: false, retryable: true };
  }
  if (!preview.changed) {
    return { applied: false, retryable: true };
  }
  showMarkdown(`ForgeCoder: ${label}`, preview.diff ?? '');
  const choice = await vscode.window.showWarningMessage(
    `Apply the proposed edit to ${ctx.file}?`, { modal: true }, 'Apply',
  );
  if (choice !== 'Apply') {
    return { applied: false, retryable: false };
  }
  if (document.isClosed || document.isDirty || document.version !== version) {
    void vscode.window.showWarningMessage('Editor changed; regenerate the edit.');
    return { applied: false, retryable: true };
  }
  const applied = await api.patchApply(ctx.workspace, patch, true, original);
  if (!applied.ok) {
    void vscode.window.showErrorMessage(applied.error ?? 'Apply failed.');
    return { applied: false, retryable: true };
  }
  void vscode.window.showInformationMessage(applied.applied ? `Updated ${ctx.file}.` : 'No changes needed.');
  return { applied: Boolean(applied.applied), retryable: false };
}

export function registerCodeActions(
  context: vscode.ExtensionContext,
  api: ForgeApi,
  sidebar: SidebarProvider,
): void {
  context.subscriptions.push(
    vscode.commands.registerCommand('forgecoder.explain', async () => {
      const ctx = activeFileContext();
      if (!ctx.code) {
        void vscode.window.showWarningMessage('Select some code first.');
        return;
      }
      const result = await api.explain(ctx.code, ctx.workspace, ctx.file);
      if (!result.ok) {
        void vscode.window.showErrorMessage(String(result.error ?? 'Explain failed'));
        return;
      }
      showMarkdown('ForgeCoder: Explain', result.explanation ?? '');
    }),

    vscode.commands.registerCommand('forgecoder.fix', async () => {
      const ctx = activeFileContext();
      if (!ctx.code) {
        void vscode.window.showWarningMessage('Select some code first.');
        return;
      }
      const errorText = diagnosticSummary();
      const result = await api.fix(ctx.code, errorText, ctx.workspace, ctx.file);
      if (!result.ok) {
        void vscode.window.showErrorMessage(String(result.error ?? 'No fix produced'));
        return;
      }
      if (result.diagnosis) {
        void vscode.window.showInformationMessage(result.diagnosis.split('\n')[0]);
      }
      const plan = planInjection({ patches: result.patches, activeFile: ctx.file, workspace: ctx.workspace });
      if (plan.kind === 'inject') {
        const injected = await injectIntoActiveFile(api, ctx, plan.patch, 'Fix');
        if (!injected.retryable) return;
      }
      if (result.patches?.length) {
        sidebar.offerPatch(ctx.workspace ?? currentWorkspace() ?? '', result.patches);
      } else {
        const fallback = plan.kind === 'offer' ? plan.reason : 'No structured patch was produced.';
        showMarkdown('ForgeCoder: Fix', result.diagnosis || result.summary || fallback);
      }
    }),

    vscode.commands.registerCommand('forgecoder.refactor', async () => {
      const ctx = activeFileContext();
      if (!ctx.code) {
        void vscode.window.showWarningMessage('Select some code first.');
        return;
      }
      const result = await api.edit(ctx.code, 'Refactor the selected code.', ctx.workspace, ctx.file);
      if (!result.ok) {
        void vscode.window.showErrorMessage(String(result.error ?? 'Refactor failed'));
        return;
      }
      if (result.patches?.length) {
        const plan = planInjection({ patches: result.patches, activeFile: ctx.file, workspace: ctx.workspace });
        if (plan.kind === 'inject') {
          const injected = await injectIntoActiveFile(api, ctx, plan.patch, 'Refactor');
          if (!injected.retryable) return;
        }
        sidebar.offerPatch(ctx.workspace ?? currentWorkspace() ?? '', result.patches);
      } else {
        showMarkdown('ForgeCoder: Refactor', result.summary || result.proposal || 'No structured patch was produced.');
      }
    }),

    vscode.commands.registerCommand('forgecoder.generateTests', async () => {
      const ctx = activeFileContext();
      if (!ctx.code) {
        void vscode.window.showWarningMessage('Select some code first.');
        return;
      }
      const result = await api.tests(ctx.code, ctx.workspace, ctx.file);
      if (!result.ok) {
        void vscode.window.showErrorMessage(String(result.error ?? 'Test generation failed'));
        return;
      }
      showMarkdown('ForgeCoder: Generated Tests', result.tests ?? '');
    }),

    vscode.commands.registerCommand('forgecoder.explainError', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        return;
      }
      const ctx = activeFileContext();
      const result = await api.fix(ctx.code, diagnosticSummary(), ctx.workspace, ctx.file);
      if (!result.ok && !result.diagnosis) {
        void vscode.window.showErrorMessage('No diagnostics found in this editor — run a build first.');
        return;
      }
      showMarkdown('ForgeCoder: Error Diagnosis', result.diagnosis ?? 'No error output detected.');
    }),

    vscode.commands.registerCommand('forgecoder.search', async () => {
      const workspace = currentWorkspace();
      if (!workspace) {
        void vscode.window.showWarningMessage('Open a workspace folder first.');
        return;
      }
      const query = await vscode.window.showInputBox({ prompt: 'Search the repository', placeHolder: 'e.g. authentication bug' });
      if (!query) {
        return;
      }
      const response = await api.search(query, workspace, 8);
      if (!response.ok) {
        void vscode.window.showErrorMessage('Search failed');
        return;
      }
      const pick = await vscode.window.showQuickPick(
        response.results.map((r) => ({
          label: `${r.path}:${r.start_line}-${r.end_line}`,
          detail: r.content.split('\n')[0]?.slice(0, 120) ?? '',
          result: r,
        })),
        { placeHolder: `Matches for "${query}"` },
      );
      if (pick) {
        const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(
          workspace + '\\' + pick.result.path.replace(/\//g, '\\'),
        ));
        const range = new vscode.Range(pick.result.start_line - 1, 0, pick.result.end_line - 1, 0);
        void vscode.window.showTextDocument(doc, { selection: range });
      }
    }),

    vscode.commands.registerCommand('forgecoder.editAndApply', async () => {
      try {
        const editor = vscode.window.activeTextEditor;
        const ctx = activeFileContext();
        if (!vscode.workspace.isTrusted || !editor || !ctx.workspace || !ctx.file ||
            editor.document.uri.scheme !== 'file' || editor.document.isDirty) {
          void vscode.window.showWarningMessage('Open a saved file in a trusted workspace first.');
          return;
        }
        const instruction = await vscode.window.showInputBox({ prompt: 'What should ForgeCoder change in this file?' });
        if (!instruction?.trim()) return;
        const result = await api.edit(ctx.code, instruction, ctx.workspace, ctx.file, ctx.selection);
        const plan = planInjection({ patches: result.patches, activeFile: ctx.file, workspace: ctx.workspace });
        if (plan.kind === 'offer') {
          void vscode.window.showWarningMessage(result.error ?? result.summary ?? plan.reason);
          return;
        }
        await injectIntoActiveFile(api, ctx, plan.patch, 'Proposed Edit');
      } catch (error) {
        void vscode.window.showErrorMessage(`Edit and Apply failed: ${String(error)}`);
      }
    }),
    vscode.commands.registerCommand('forgecoder.applyPatch', async () => {
      if (!sidebar.hasPendingPatch()) {
        void vscode.window.showWarningMessage('No pending patch — ask ForgeCoder for a fix first.');
        return;
      }
      await sidebar.applyPendingPatch();
    }),
  );
}