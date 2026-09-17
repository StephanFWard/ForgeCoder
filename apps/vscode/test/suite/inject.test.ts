import * as assert from 'assert';
import { FilePatch } from '../../src/client/api';
import { checkAnchors, normalizeLf, planInjection } from '../../src/patch/inject';

suite('Suggestion -> local injection', () => {
  const patch = (path: string, start = 3): FilePatch => ({
    path,
    operations: [{ type: 'replace', start_line: start, end_line: start, content: 'x' }],
  });

  test('a suggestion for the open file is injected', () => {
    const plan = planInjection({
      patches: [patch('src/service.py')],
      activeFile: 'src/service.py',
      workspace: '/ws',
    });
    assert.strictEqual(plan.kind, 'inject');
    assert.strictEqual(plan.kind === 'inject' && plan.patch.path, 'src/service.py');
  });

  test('suggestions that are not for the open file are offered, never written', () => {
    const cases = [
      { patches: undefined, activeFile: 'src/service.py', workspace: '/ws' },
      { patches: [patch('src/other.py')], activeFile: 'src/service.py', workspace: '/ws' },
      { patches: [patch('src/service.py')], activeFile: undefined, workspace: '/ws' },
      { patches: [patch('src/service.py')], activeFile: 'src/service.py', workspace: undefined },
      { patches: [patch('src/service.py'), patch('src/other.py')], activeFile: 'src/service.py', workspace: '/ws' },
    ];
    for (const input of cases) {
      const plan = planInjection(input);
      assert.strictEqual(plan.kind, 'offer');
      assert.ok(plan.kind === 'offer' && plan.reason.length > 0);
    }
  });

  test('anchors require the preview to match the editor byte for byte', () => {
    const documentText = 'a\r\nb\n';
    assert.strictEqual(normalizeLf(documentText), 'a\nb\n');
    assert.strictEqual(checkAnchors({
      patchPath: 'src/service.py', activeFile: 'src/service.py',
      previewOriginal: 'a\nb\n', documentText,
    }).ok, true);
    const stale = checkAnchors({
      patchPath: 'src/service.py', activeFile: 'src/service.py',
      previewOriginal: 'a\nb\nc\n', documentText,
    });
    assert.strictEqual(stale.ok, false);
    assert.ok(stale.reason?.includes('changed'));
    assert.strictEqual(checkAnchors({
      patchPath: 'src/other.py', activeFile: 'src/service.py',
      previewOriginal: 'a\nb\n', documentText,
    }).ok, false);
    assert.strictEqual(checkAnchors({
      patchPath: 'src/service.py', activeFile: 'src/service.py',
      previewOriginal: undefined, documentText,
    }).ok, false);
  });
});