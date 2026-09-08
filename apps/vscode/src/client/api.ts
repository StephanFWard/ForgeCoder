/** Typed wrappers around the Forge server endpoints. */
import { HttpClient, SseEvent } from './httpClient';

export interface SelectionInfo {
  start: number;
  end: number;
}

export interface ChatTurn {
  message: string;
  workspace?: string;
  file?: string;
  selection?: SelectionInfo;
  history?: Array<{ role: string; content: string }>;
}

export interface ChatContextEvent extends SseEvent {
  type: 'context';
  total_tokens: number;
  sections: string[];
}
export interface ChatDeltaEvent extends SseEvent {
  type: 'delta';
  content: string;
}
export interface ChatDoneEvent extends SseEvent {
  type: 'done';
}
export interface ChatErrorEvent extends SseEvent {
  type: 'error';
  message: string;
}

export interface CompletionRequest {
  language: string;
  file: string;
  prefix: string;
  suffix: string;
  context?: string[];
  max_tokens?: number;
}

export interface PatchOperation {
  type: 'insert' | 'replace' | 'delete';
  start_line: number;
  end_line?: number;
  content: string;
}

export interface FilePatch {
  path: string;
  operations: PatchOperation[];
}

export interface MultiPatchFile {
  path: string;
  diff?: string;
  original?: string;
  proposed?: string;
  changed?: boolean;
  created?: boolean;
  operations?: PatchOperation[];
}

export interface MultiPatchPreviewResponse {
  ok: boolean;
  error?: string;
  summary?: string;
  files?: MultiPatchFile[];
}

export interface MultiPatchApplyResponse {
  ok: boolean;
  error?: string;
  summary?: string;
  applied?: string[];
  files?: MultiPatchFile[];
}

export interface PatchPreviewResponse {
  ok: boolean;
  error?: string;
  path?: string;
  diff?: string;
  original?: string;
  proposed?: string;
  changed?: boolean;
}

export interface SearchResult {
  path: string;
  language: string;
  content: string;
  start_line: number;
  end_line: number;
}

export interface GitChangeEntry {
  path: string;
  staged: string;
  worktree: string;
  summary: string;
}

export interface GitStatusResponse {
  ok: boolean;
  error?: string;
  branch?: string | null;
  remote?: string | null;
  entries?: GitChangeEntry[];
  counts?: { total: number; staged: number; worktree: number; untracked: number };
  recent?: Array<{ hash: string; author: string; subject: string }>;
}

export interface GitChangesResponse {
  ok: boolean;
  error?: string;
  clean?: boolean;
  branch?: string | null;
  entries?: GitChangeEntry[];
  review?: string;
  diff?: string;
}

export interface PlanStep {
  title: string;
  action: 'search' | 'explain' | 'edit' | 'test' | 'commit';
  detail: string;
}

export interface PlanResponse {
  ok: boolean;
  error?: string;
  plan_id: string;
  summary: string;
  steps: PlanStep[];
  fallback?: boolean;
}

export interface ActResponse {
  ok: boolean;
  error?: string;
  step_index: number;
  action: string;
  title: string;
  output: string;
  patch?: FilePatch;
  multiPatch?: MultiPatchFile[];
  next_index?: number | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  llama_server: { ok: boolean; url: string };
  database: { ok: boolean; path: string; files: number; chunks: number; symbols: number };
}

export class ForgeApi {
  constructor(private readonly client: HttpClient) {}

  health(signal?: AbortSignal): Promise<HealthResponse> {
    return this.client.json<HealthResponse>('/health', undefined, signal);
  }

  chat(turn: ChatTurn): Promise<Response> {
    return this.client.fetch('/v1/chat', turn);
  }

  async chatStream(turn: ChatTurn, onEvent: (ev: SseEvent) => void, signal?: AbortSignal): Promise<void> {
    await this.client.stream('/v1/chat', turn, onEvent, signal);
  }

  completion(req: CompletionRequest, signal?: AbortSignal): Promise<{ completion: string; error?: string }> {
    return this.client.json('/v1/completion', req, signal);
  }

  search(query: string, workspace?: string, limit = 10): Promise<{ ok: boolean; results: SearchResult[] }> {
    return this.client.json('/v1/search', { query, workspace, limit });
  }

  index(workspace: string): Promise<{ ok: boolean; error?: string; added?: number }> {
    return this.client.json('/v1/index', { workspace });
  }

  patchPreview(workspace: string, patch: FilePatch): Promise<PatchPreviewResponse> {
    return this.client.json('/v1/patch/preview', { workspace, patch });
  }

  patchApply(workspace: string, patch: FilePatch, confirmed: boolean): Promise<{ ok: boolean; error?: string; applied?: boolean }> {
    if (!confirmed) {
      return Promise.resolve({ ok: false, error: 'Confirmation required before applying a file write.' });
    }
    return this.client.json(
      '/v1/patch/apply',
      { workspace, patch },
      undefined,
      { 'X-Forge-Confirm': 'true' },
    );
  }

  fix(code: string, errorText: string, workspace?: string, file?: string):
    Promise<{ ok: boolean; error?: string; diagnosis?: string; summary?: string; patches?: FilePatch[] }> {
    return this.client.json('/v1/fix', { code, error: errorText, workspace, file });
  }

  edit(code: string, instruction?: string, workspace?: string, file?: string):
    Promise<{ ok: boolean; error?: string; summary?: string; patches?: FilePatch[]; proposal?: string }> {
    return this.client.json('/v1/edit', { code, instruction, workspace, file });
  }

  explain(code: string, workspace?: string, file?: string): Promise<{ ok: boolean; error?: string; explanation?: string }> {
    return this.client.json('/v1/explain', { code, workspace, file });
  }

  tests(code: string, workspace?: string, file?: string): Promise<{ ok: boolean; error?: string; tests?: string }> {
    return this.client.json('/v1/tests', { code, workspace, file });
  }

  // ------------------------------------------------------------- git powers
  gitStatus(workspace: string): Promise<GitStatusResponse> {
    return this.client.json('/v1/git/status', { workspace });
  }

  gitChanges(workspace: string): Promise<GitChangesResponse> {
    return this.client.json('/v1/git/changes', { workspace }, undefined, undefined);
  }

  gitCommit(workspace: string, message: string, confirmed: boolean): Promise<{ ok: boolean; error?: string; hash?: string; subject?: string; staged?: number }> {
    if (!confirmed) {
      return Promise.resolve({ ok: false, error: 'Confirmation required before a commit.' });
    }
    return this.client.json(
      '/v1/git/commit',
      { workspace, message, add_all: true },
      undefined,
      { 'X-Forge-Confirm': 'true' },
    );
  }

  gitPush(workspace: string, confirmed: boolean): Promise<{ ok: boolean; error?: string; remote?: string; branch?: string }> {
    if (!confirmed) {
      return Promise.resolve({ ok: false, error: 'Confirmation required before a push.' });
    }
    return this.client.json('/v1/git/push', { workspace }, undefined, { 'X-Forge-Confirm': 'true' });
  }

  // ------------------------------------------------------------ multi-file patches
  multiPatchPreview(workspace: string, patches: FilePatch[], creates: Record<string, string> = {}, message = ''): Promise<MultiPatchPreviewResponse> {
    return this.client.json('/v1/patch/multi-preview', { workspace, patches, creates, message });
  }

  multiPatchApply(workspace: string, patches: FilePatch[], creates: Record<string, string> = {}, message = '', confirmed: boolean): Promise<MultiPatchApplyResponse> {
    if (!confirmed) {
      return Promise.resolve({ ok: false, error: 'Confirmation required before applying multi-file writes.' });
    }
    return this.client.json(
      '/v1/patch/multi-apply',
      { workspace, patches, creates, message },
      undefined,
      { 'X-Forge-Confirm': 'true' },
    );
  }

  // ------------------------------------------------------------ plan -> act
  plan(message: string, workspace?: string): Promise<PlanResponse> {
    return this.client.json('/v1/plan', { message, workspace }, undefined, undefined);
  }

  act(planId: string, index: number, workspace?: string, confirm = false): Promise<ActResponse> {
    return this.client.json(
      '/v1/act',
      { plan_id: planId, index, workspace },
      undefined,
      confirm ? { 'X-Forge-Confirm': 'true' } : undefined,
    );
  }

  ask(message: string, workspace?: string): Promise<{ ok: boolean; answer?: string; error?: string }> {
    return this.client.json('/v1/ask', { message, workspace }, undefined, undefined);
  }
}