/** Code-action implementations: explain / fix / refactor / tests / search / error. */
import * as vscode from 'vscode';
import { ForgeApi } from '../client/api';
import { SidebarProvider } from '../chat/SidebarProvider';
import { activeFileContext, currentWorkspace, diagnosticSummary } from '../context/selection';
import { showMarkdown } from '../ui/markdown';

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
      const pending = result.patches?.length
        ? result.patches
        : undefined;
      if (pending) {
        sidebar.offerPatch(ctx.workspace ?? currentWorkspace() ?? '', pending);
      } else {
        showMarkdown('ForgeCoder: Fix', result.diagnosis ?? 'No structured patch was produced.');
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
        sidebar.offerPatch(ctx.workspace ?? currentWorkspace() ?? '', result.patches);
      } else {
        showMarkdown('ForgeCoder: Refactor', result.proposal ?? 'No structured patch was produced.');
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

    vscode.commands.registerCommand('forgecoder.applyPatch', async () => {
      if (!sidebar.hasPendingPatch()) {
        void vscode.window.showWarningMessage('No pending patch — ask ForgeCoder for a fix first.');
        return;
      }
      await sidebar.applyPendingPatch();
    }),
  );
}