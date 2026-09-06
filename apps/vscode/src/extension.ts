/** ForgeCoder VS Code extension entry point. */
import * as vscode from 'vscode';
import { ForgeApi } from './client/api';
import { HttpClient } from './client/httpClient';
import { ForgeCompletionProvider } from './completion/ForgeCompletionProvider';
import { registerCommands } from './commands';

let statusBar: vscode.StatusBarItem;

export function activate(context: vscode.ExtensionContext): void {
  const serverUrl =
    vscode.workspace.getConfiguration('forgecoder').get<string>('serverUrl', 'http://127.0.0.1:8787');
  const api = new ForgeApi(new HttpClient(serverUrl));

  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.text = '$(sync~spin) ForgeCoder…';
  statusBar.tooltip = 'ForgeCoder: checking 127.0.0.1:8787';
  statusBar.show();
  context.subscriptions.push(statusBar);

  registerCommands(context, api);
  registerCompletion(context, api);
  registerWorkspaceIndexing(context, api);

  void refreshHealth(api);
  const timer = setInterval(() => void refreshHealth(api), 15_000);
  context.subscriptions.push(new vscode.Disposable(() => clearInterval(timer)));
}

export function deactivate(): void {
  // Nothing to tear down — the Forge server and llama.cpp are separate
  // processes owned by the runtime/scripts launchers.
}

function registerCompletion(context: vscode.ExtensionContext, api: ForgeApi): void {
  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider({ scheme: 'file' }, new ForgeCompletionProvider(api)),
  );
}

function registerWorkspaceIndexing(context: vscode.ExtensionContext, api: ForgeApi): void {
  const indexFolder = async (folder: vscode.WorkspaceFolder): Promise<void> => {
    const shouldIndex = vscode.workspace
      .getConfiguration('forgecoder')
      .get<boolean>('autoIndexOnOpen', true);
    if (shouldIndex) {
      await api.index(folder.uri.fsPath).catch(() => undefined);
    }
  };
  const first = vscode.workspace.workspaceFolders?.[0];
  if (first) {
    void indexFolder(first);
  }
  context.subscriptions.push(vscode.workspace.onDidChangeWorkspaceFolders((event) => {
    if (event.added?.[0]) {
      void indexFolder(event.added[0]);
    }
  }));
}

async function refreshHealth(api: ForgeApi): Promise<void> {
  const health = await api.health().catch(() => undefined);
  if (health) {
    const llama = health.llama_server.ok ? '' : ' · llama offline';
    statusBar.text = `$(flame) ForgeCoder${llama}`;
    statusBar.tooltip = `Forge API ${health.version} — llama.cpp ${health.llama_server.ok ? 'ready' : 'offline'}`;
  } else {
    statusBar.text = '$(circle-slash) ForgeCoder offline';
    statusBar.tooltip = 'Forge server not reachable at 127.0.0.1:8787 — run: python -m uvicorn forge_server.main:app --host 127.0.0.1 --port 8787';
  }
}