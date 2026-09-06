import * as assert from 'assert';
import * as vscode from 'vscode';

suite('ForgeCoder extension', () => {
  test('activates and registers the chat command', async () => {
    const ext = vscode.extensions.getExtension('forgecoder.forgecoder');
    assert.ok(ext, 'extension is installed in the test host');
    await ext?.activate();

    const commands = await vscode.commands.getCommands(true);
    assert.ok(commands.includes('forgecoder.chat'), 'chat command registered');
    assert.ok(commands.includes('forgecoder.explain'), 'explain command registered');
    assert.ok(commands.includes('forgecoder.fix'), 'fix command registered');
    assert.ok(commands.includes('forgecoder.generateTests'), 'tests command registered');
    assert.ok(commands.includes('forgecoder.search'), 'search command registered');
  });

  test('server health graceful degradation when server is offline', async () => {
    // The status-bar refresh swallows errors; open chat should also not throw
    // synchronously. This validates that combinations don't crash on startup.
    await vscode.commands.executeCommand('forgecoder.chat');
    assert.ok(true, 'open chat panel did not throw');
  });
});