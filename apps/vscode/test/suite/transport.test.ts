import * as assert from 'assert';
import { ForgeApi } from '../../src/client/api';
import { HttpClient } from '../../src/client/httpClient';

suite('Local patch transport', () => {
  test('confirmed apply forwards grant and original; unconfirmed makes no request', async () => {
    const saved = globalThis.fetch;
    let calls = 0;
    globalThis.fetch = (async (input, init) => {
      calls += 1;
      assert.strictEqual(String(input), 'http://127.0.0.1:8787/v1/patch/apply');
      assert.strictEqual(new Headers(init?.headers).get('X-Forge-Confirm'), 'true');
      assert.strictEqual(JSON.parse(String(init?.body)).expected_original, 'original\n');
      return new Response(JSON.stringify({ ok: true, applied: true }), { status: 200 });
    }) as typeof fetch;
    try {
      const api = new ForgeApi(new HttpClient('http://127.0.0.1:8787'));
      const patch = { path: 'a.py', operations: [{ type: 'replace' as const, start_line: 1, content: 'new' }] };
      assert.strictEqual((await api.patchApply('workspace', patch, false)).ok, false);
      assert.strictEqual(calls, 0);
      assert.strictEqual((await api.patchApply('workspace', patch, true, 'original\n')).applied, true);
      assert.strictEqual(calls, 1);
    } finally {
      globalThis.fetch = saved;
    }
  });
});
