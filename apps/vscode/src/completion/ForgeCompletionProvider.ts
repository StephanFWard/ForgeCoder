/** Inline completion provider — fast & conservative, separate from chat. */
import * as vscode from 'vscode';
import { ForgeApi, CompletionRequest } from '../client/api';

const WINDOW_BEFORE = 40;   // preceding lines of context
const WINDOW_AFTER = 20;    // following lines of context

export class ForgeCompletionProvider implements vscode.InlineCompletionItemProvider {
  constructor(private readonly api: ForgeApi) {}

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken,
  ): Promise<vscode.InlineCompletionList | undefined> {
    const cfg = vscode.workspace.getConfiguration('forgecoder');
    if (!cfg.get<boolean>('enableAutocomplete', true)) {
      return undefined;
    }

    const line = document.lineAt(position.line).text;
    const cursorInLine = position.character;
    const beforeInLine = line.slice(0, cursorInLine);
    const afterInLine = line.slice(cursorInLine);

    // Don't suggest mid-word unless we're right at a natural boundary.
    if (cursorInLine > 0 && /[\w.]$/.test(beforeInLine) && /[\w.]/.test(afterInLine)) {
      return undefined;
    }

    const startLine = Math.max(0, position.line - WINDOW_BEFORE);
    const endLine = Math.min(document.lineCount - 1, position.line + WINDOW_AFTER);

    const prefix: string[] = [];
    for (let i = startLine; i < position.line; i += 1) {
      prefix.push(document.lineAt(i).text);
    }
    prefix.push(beforeInLine);

    const suffix: string[] = [];
    if (afterInLine) {
      suffix.push(afterInLine);
    }
    for (let i = position.line + 1; i <= endLine; i += 1) {
      suffix.push(document.lineAt(i).text);
    }

    const request: CompletionRequest = {
      language: document.languageId,
      file: document.fileName,
      prefix: prefix.join('\n'),
      suffix: suffix.join('\n'),
      max_tokens: cfg.get<number>('autocompleteMaxTokens', 64),
    };

    const controller = new AbortController();
    const onAbort = () => controller.abort();
    token.onCancellationRequested(onAbort);

    try {
      const response = await this.api.completion(request, controller.signal);
      if (token.isCancellationRequested || !response.completion) {
        return undefined;
      }
      const completion = response.completion.trimEnd();
      if (!completion) {
        return undefined;
      }
      const item = new vscode.InlineCompletionItem(completion);
      return new vscode.InlineCompletionList([item]);
    } catch {
      return undefined; // server offline / slow — stay quiet
    } finally {
      token.onCancellationRequested(onAbort);
    }
  }
}