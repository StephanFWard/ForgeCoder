/* ForgeCoder chat WebView script — renders streamed assistant messages. */
(function () {
  const messages = document.getElementById('messages');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('send');
  const clearBtn = document.getElementById('clear');
  const patchBar = document.getElementById('patchBar');
  const patchInfo = document.getElementById('patchInfo');
  const viewDiff = document.getElementById('viewDiff');
  const applyBtn = document.getElementById('apply');

  const vscode = acquireVsCodeApi();
  let currentAssistant = null;

  function addMessage(role, text) {
    const el = document.createElement('div');
    el.className = 'message ' + role;
    el.textContent = text;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
    return el;
  }

  function send() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    vscode.postMessage({ command: 'send', message: text });
  }

  sendBtn.addEventListener('click', send);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') send();
  });
  clearBtn.addEventListener('click', () => vscode.postMessage({ command: 'clear' }));
  viewDiff.addEventListener('click', () => vscode.postMessage({ command: 'viewPatches' }));
  applyBtn.addEventListener('click', () => vscode.postMessage({ command: 'applyPatch' }));

  window.addEventListener('message', (event) => {
    const msg = event.data;
    switch (msg.command) {
      case 'userMessage':
        addMessage('user', msg.content);
        break;
      case 'assistantStart': {
        currentAssistant = addMessage('assistant', '');
        const cursor = document.createElement('span');
        cursor.className = 'cursor';
        currentAssistant.appendChild(cursor);
        break;
      }
      case 'delta':
        if (!currentAssistant) return;
        currentAssistant.textContent += msg.content;
        messages.scrollTop = messages.scrollHeight;
        break;
      case 'assistantDone':
        if (currentAssistant) currentAssistant.textContent = currentAssistant.textContent.replace(/\s*$/, '');
        currentAssistant = null;
        break;
      case 'error':
        addMessage('error', msg.message);
        break;
      case 'pendingPatch':
        patchInfo.textContent = 'Patch ready for ' + msg.path;
        patchBar.classList.remove('hidden');
        break;
      case 'applied':
        patchInfo.textContent = 'Applied ' + msg.path + ' ✓';
        break;
      case 'cleared':
        messages.innerHTML = '';
        patchBar.classList.add('hidden');
        break;
    }
  });
})();