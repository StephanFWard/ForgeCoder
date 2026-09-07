/** Command registration entry point. */
import * as vscode from 'vscode';
import { ForgeApi } from '../client/api';
import { SidebarProvider } from '../chat/SidebarProvider';
import { currentWorkspace } from '../context/selection';
import { registerCodeActions } from './actions';

export function registerCommands(
  context: vscode.ExtensionContext,
  api: ForgeApi,
  sidebar: SidebarProvider,
): void {
  context.subscriptions.push(
    // "Chat" focuses the activity-bar sidebar (the primary chat surface).
    vscode.commands.registerCommand('forgecoder.chat', () => {
      sidebar.reveal();
      void vscode.commands.executeCommand(SidebarProvider.viewId + '.focus');
      const workspace = currentWorkspace();
      if (workspace) {
        void api.index(workspace).catch(() => undefined);
      }
    }),

    vscode.commands.registerCommand('forgecoder.clearChat', () => {
      sidebar.reveal();
      // Route through the webview so its DOM resets too.
      sidebar.clearConversation();
    }),

    vscode.commands.registerCommand('forgecoder.reviewChanges', () => sidebar.startReview()),
    vscode.commands.registerCommand('forgecoder.commitWithForge', () => sidebar.startCommit()),
    vscode.commands.registerCommand('forgecoder.pushWithForge', () => sidebar.startPush()),
    vscode.commands.registerCommand('forgecoder.planMode', () => sidebar.enablePlanMode()),

    vscode.commands.registerCommand('forgecoder.applyPatch', async () => {
      if (!sidebar.hasPendingPatch()) {
        void vscode.window.showWarningMessage('No pending patch — ask ForgeCoder for a fix first.');
        return;
      }
      await sidebar.applyPendingPatch();
    }),

    vscode.commands.registerCommand('forgecoder.restartServer', async () => {
      const ok = await vscode.window.showInformationMessage(
        'Restarting only re-checks the Forge server (127.0.0.1:8787). Start it with: python runtime/scripts/start_forge.py',
        { modal: true },
        'Check Now',
      );
      if (ok === 'Check Now') {
        const health = await api.health().catch(() => undefined);
        void vscode.window.showInformationMessage(
          health?.status === 'ok' ? `Forge server OK (llama: ${health.llama_server.ok ? 'ready' : 'offline'})` : 'Forge server unreachable.',
        );
      }
    }),
  );

  registerCodeActions(context, api, sidebar);
}