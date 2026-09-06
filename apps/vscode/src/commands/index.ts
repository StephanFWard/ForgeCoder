/** Command registration entry point. */
import * as vscode from 'vscode';
import { ForgeApi } from '../client/api';
import { ChatPanel } from '../chat/ChatPanel';
import { currentWorkspace } from '../context/selection';
import { registerCodeActions } from './actions';

export function registerCommands(context: vscode.ExtensionContext, api: ForgeApi): void {
  let panel: ChatPanel | undefined;

  const getPanel = (): ChatPanel | undefined => panel;

  context.subscriptions.push(
    vscode.commands.registerCommand('forgecoder.chat', () => {
      panel = ChatPanel.open(context.extensionUri, api);
      const workspace = currentWorkspace();
      if (workspace) {
        void api.index(workspace).catch(() => undefined);
      }
    }),

    vscode.commands.registerCommand('forgecoder.restartServer', async () => {
      const ok = await vscode.window.showInformationMessage(
        'Restarting only re-checks the Forge server (127.0.0.1:8787). Start it with: python -m uvicorn forge_server.main:app --host 127.0.0.1 --port 8787',
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

  registerCodeActions(context, api, getPanel);
}