/* ForgeCoder chat WebView script — streamed messages, git reviews, Plan -> Act. */
(function () {
  const messages = document.getElementById('messages');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('send');
  const clearBtn = document.getElementById('clear');
  const reviewBtn = document.getElementById('review');
  const commitBtn = document.getElementById('commitBtn');
  const planBtn = document.getElementById('planBtn');
  const patchBar = document.getElementById('patchBar');
  const patchInfo = document.getElementById('patchInfo');
  const viewDiff = document.getElementById('viewDiff');
  const applyBtn = document.getElementById('apply');

  const vscode = acquireVsCodeApi();
  let currentAssistant = null;
  let planMode = false;
  let currentPlan = null;

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
    if (planMode) {
      vscode.postMessage({ command: 'plan', message: text });
    } else {
      vscode.postMessage({ command: 'send', message: text });
    }
  }

  function renderPlan(plan) {
    currentPlan = plan;
    const box = document.createElement('div');
    box.className = 'plan';
    const head = document.createElement('div');
    head.className = 'plan-head';
    head.textContent = 'PLAN' + (plan.fallback ? ' (fallback)' : '') + ': ' + plan.summary;
    box.appendChild(head);
    plan.steps.forEach(function (step, i) {
      const row = document.createElement('div');
      row.className = 'step';
      const label = document.createElement('span');
      label.className = 'step-label';
      label.textContent = (i + 1) + '. [' + step.action + '] ' + step.title;
      const run = document.createElement('button');
      run.className = 'step-run';
      run.textContent = 'Run';
      run.addEventListener('click', function () {
        run.disabled = true;
        run.textContent = '…';
        row.classList.add('running');
        vscode.postMessage({ command: 'act', planId: plan.planId, index: i, action: step.action });
      });
      row.appendChild(label);
      row.appendChild(run);
      box.appendChild(row);
      step._row = row;
      step._run = run;
    });
    messages.appendChild(box);
    messages.scrollTop = messages.scrollHeight;
  }

  function markStep(index, ok, output) {
    if (!currentPlan || !currentPlan.steps[index]) return;
    const step = currentPlan.steps[index];
    if (step._run) step._run.remove();
    if (step._row) {
      step._row.classList.remove('running');
      step._row.classList.add(ok ? 'ok' : 'failed');
      const out = document.createElement('div');
      out.className = 'step-output';
      out.textContent = output || (ok ? 'done' : 'failed');
      step._row.appendChild(out);
    }
    messages.scrollTop = messages.scrollHeight;
  }

  sendBtn.addEventListener('click', send);
  input.addEventListener('keydown', function (e) { if (e.key === 'Enter') send(); });
  clearBtn.addEventListener('click', function () { vscode.postMessage({ command: 'clear' }); });
  reviewBtn.addEventListener('click', function () { vscode.postMessage({ command: 'reviewChanges' }); });
  commitBtn.addEventListener('click', function () { vscode.postMessage({ command: 'commit' }); });
  viewDiff.addEventListener('click', function () { vscode.postMessage({ command: 'viewPatches' }); });
  applyBtn.addEventListener('click', function () { vscode.postMessage({ command: 'applyPatch' }); });
  planBtn.addEventListener('click', function () {
    planMode = !planMode;
    planBtn.classList.toggle('active', planMode);
    input.placeholder = planMode
      ? 'Describe a goal — ForgeCoder will plan it, then run each step…'
      : 'Ask ForgeCoder anything...';
    addMessage('system', planMode
      ? 'Plan & Act mode ON — your next message becomes a plan with runnable steps.'
      : 'Plan & Act mode off.');
  });

  window.addEventListener('message', function (event) {
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
      case 'notice':
        addMessage('system', msg.message);
        break;
      case 'planMode':
        if (!planMode) planBtn.click();
        break;
      case 'changesReview':
        addMessage('system',
          'Working changes on ' + (msg.branch || '?') + ' — ' +
          (msg.clean ? 'clean tree' : (msg.count + ' file(s) changed')));
        addMessage('assistant', msg.review || '(no review produced)');
        break;
      case 'plan':
        renderPlan(msg);
        break;
      case 'stepStart':
        if (currentPlan && currentPlan.steps[msg.index] && currentPlan.steps[msg.index]._row) {
          currentPlan.steps[msg.index]._row.classList.add('running');
        }
        break;
      case 'stepDone':
        markStep(msg.index, msg.ok, msg.output);
        if (msg.hasPatch) {
          patchInfo.textContent = 'Patch ready — review before applying.';
          patchBar.classList.remove('hidden');
        }
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
        currentPlan = null;
        break;
    }
  });
})();