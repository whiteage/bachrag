// ─── State ───────────────────────────────────────────────────────────────────
let ragReady         = false;
let isStreaming      = false;
let logVisible       = false;
let indexingProgress = { active: false, done: 0, total: 0, log: [] };
let wasIndexing      = false;
let cachedFiles      = [];  // последний известный список файлов

// ─── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadFiles();
  pollStatus();
  setInterval(pollStatus, 2000);
  autoResize(document.getElementById('queryInput'));
});

// ─── API Key ─────────────────────────────────────────────────────────────────
async function saveApiKey(val) {
  if (!val.trim()) return;
  return apiRequest('/api/config', { method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ api_key: val }) });
}

async function apiRequest(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
  return data;
}

function showError(error) {
  hideWelcome();
  appendMsg({role: 'assistant', content: '❌ ' + error.message, ts: timestamp()});
}

// ─── Drag & Drop ─────────────────────────────────────────────────────────────
const dz = document.getElementById('dropZone');
dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', ()  => dz.classList.remove('drag'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('drag');
  handleFiles(e.dataTransfer.files);
});

// ─── Upload ──────────────────────────────────────────────────────────────────
async function handleFiles(files) {
  if (!files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  try {
    const data = await apiRequest('/api/upload', { method: 'POST', body: fd });
    if (data.uploaded?.length) await loadFiles();
    await pollStatus();
  } catch(e) { showError(e); }
}

// ─── Files list ───────────────────────────────────────────────────────────────
async function loadFiles() {
  const r     = await fetch('/api/files');
  const files = await r.json();
  cachedFiles = files;
  renderFiles(files);
  updateIndexBtn(files);
}

function renderFiles(files) {
  const list  = document.getElementById('filesList');
  if (!files.length) {
    list.innerHTML = '<div class="empty-files">Нет загруженных файлов</div>';
    return;
  }
  list.innerHTML = files.map(f => {
    const ext = f.name.split('.').pop().toLowerCase();
    // Отображаемый статус для tooltip
    const statusLabel = { pending:'ожидает', modified:'изменён', indexing:'индексируется', done:'проиндексирован', error:'ошибка' }[f.status] || f.status;
    return `<div class="file-item">
      <div class="file-ext ${ext}">${ext}</div>
      <div class="file-info">
        <div class="file-name" title="${escHtml(f.name)}">${escHtml(f.name)}</div>
        <div class="file-size">${f.size}${f.indexed_at ? ' · ' + f.indexed_at : ''}</div>
      </div>
      <div class="file-dot ${f.status}" title="${statusLabel}"></div>
      <button class="file-del" data-filename="${escHtml(f.name)}" title="Удалить">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>`;
  }).join('');
  list.querySelectorAll('.file-del').forEach(button => {
    button.addEventListener('click', () => deleteFile(button.dataset.filename));
  });
}

async function deleteFile(name) {
  try {
    await apiRequest('/api/files/' + encodeURIComponent(name), { method: 'DELETE' });
    await loadFiles();
    await pollStatus();
  } catch(e) { showError(e); }
}

function updateIndexBtn(files) {
  const btn     = document.getElementById('indexBtn');
  const running = indexingProgress.active;
  const pending = (files || []).filter(f => ['pending', 'modified', 'error'].includes(f.status)).length;

  if (running) {
    btn.disabled   = true;
    btn.className  = 'index-btn running';
    btn.innerHTML  = `<div class="spinner"></div> Индексирование...`;
  } else if (pending > 0) {
    btn.disabled   = false;
    btn.className  = 'index-btn';
    btn.innerHTML  = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Индексировать (${pending})`;
  } else {
    btn.disabled   = true;
    btn.className  = 'index-btn';
    btn.innerHTML  = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Индексировать`;
  }
}

// ─── Indexing ────────────────────────────────────────────────────────────────
async function startIndexing() {
  const key = document.getElementById('apiKeyInput').value.trim();
  try {
    if (key) await saveApiKey(key);
    await apiRequest('/api/index', { method: 'POST' });
    wasIndexing = true;
    await pollStatus();
  } catch(e) { showError(e); }
}

async function pollStatus() {
  try {
    const r    = await fetch('/api/index/status');
    const data = await r.json();

    indexingProgress = data;
    ragReady         = data.rag_ready;

    // Статус-точка
    const dot = document.getElementById('statusDot');
    const txt = document.getElementById('statusText');
    if (data.active) {
      dot.className = 'status-dot loading';
      txt.textContent = data.current ? `Обработка: ${data.current}` : 'Индексирование...';
    } else if (ragReady) {
      dot.className = 'status-dot ready';
      txt.textContent = 'RAG готов';
    } else {
      dot.className = 'status-dot offline';
      txt.textContent = 'Не готов';
    }

    document.getElementById('sendBtn').disabled = !ragReady || isStreaming;

    if (data.active) {
      wasIndexing = true;
      await loadFiles();
    } else if (wasIndexing && !data.active) {
      // Только что завершилось — обновляем один раз
      wasIndexing = false;
      await loadFiles();
    } else {
      // Простой тик — используем кешированный список
      updateIndexBtn(cachedFiles);
    }

    if (data.log?.length) renderLog(data.log);
  } catch(e) { console.error('Ошибка обновления статуса:', e); }
}

function renderLog(lines) {
  const panel = document.getElementById('logPanel');
  panel.innerHTML = lines.map(l => {
    const cls = l.includes('✅') || l.includes('готов') ? 'ok'
              : l.includes('❌') || l.includes('Ошибка') ? 'err' : '';
    return `<div class="log-line ${cls}">${escHtml(l)}</div>`;
  }).join('');
  panel.scrollTop = panel.scrollHeight;
}

function toggleLog() {
  logVisible = !logVisible;
  document.getElementById('logPanel').classList.toggle('show', logVisible);
}

// ─── Chat ────────────────────────────────────────────────────────────────────
function setQuery(text) {
  const inp = document.getElementById('queryInput');
  inp.value = text;
  autoResize(inp);
  inp.focus();
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

async function sendQuery() {
  const inp = document.getElementById('queryInput');
  const q   = inp.value.trim();
  if (!q || !ragReady || isStreaming) return;

  inp.value   = '';
  autoResize(inp);
  isStreaming  = true;
  document.getElementById('sendBtn').disabled = true;
  hideWelcome();

  appendMsg({ role: 'user', content: q, ts: timestamp() });

  const msgs    = document.getElementById('messages');
  const thinkEl = document.createElement('div');
  thinkEl.className = 'msg ai';
  thinkEl.innerHTML = `<div class="avatar ai">AI</div><div><div class="bubble"><div class="thinking-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div></div>`;
  msgs.appendChild(thinkEl);
  msgs.scrollTop = msgs.scrollHeight;

  try {
    const key = document.getElementById('apiKeyInput').value.trim();
    if (key) await saveApiKey(key);
    const r    = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ query: q }),
    });
    const data = await r.json();
    thinkEl.remove();
    appendMsg(data.error ? { role: 'ai', content: '❌ ' + data.error, ts: timestamp() } : data);
  } catch(e) {
    thinkEl.remove();
    appendMsg({ role: 'ai', content: '❌ Ошибка соединения с сервером', ts: timestamp() });
  }

  isStreaming = false;
  document.getElementById('sendBtn').disabled = !ragReady;
}

function appendMsg(msg) {
  const msgs   = document.getElementById('messages');
  const isUser = msg.role === 'user';
  const div    = document.createElement('div');
  div.className = `msg ${isUser ? 'user' : 'ai'}`;

  const contentHtml = isUser ? escHtml(msg.content) : renderMarkdown(msg.content);

  let extras = '';
  if (!isUser && msg.subqueries?.length)
    extras += `<div class="subqueries-row"><div class="subq-label">Подзапросы</div><div class="subq-chips">${msg.subqueries.map(s=>`<span class="subq-chip">${escHtml(s)}</span>`).join('')}</div></div>`;
  if (!isUser && msg.sources?.length)
    extras += `<div class="sources-row">${msg.sources.map(s=>`<span class="source-chip">📄 ${escHtml(s)}</span>`).join('')}</div>`;

  div.innerHTML = `
    <div class="avatar ${isUser ? 'user' : 'ai'}">${isUser ? 'Вы' : 'AI'}</div>
    <div>
      <div class="bubble">${contentHtml}${extras}</div>
      <div class="msg-ts">${msg.ts || ''}</div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

async function clearChat() {
  await fetch('/api/chat/clear', { method: 'POST' });
  document.getElementById('messages').innerHTML = '';
  showWelcome();
}

function hideWelcome() { document.getElementById('welcome')?.remove(); }

function showWelcome() {
  const msgs = document.getElementById('messages');
  const w    = document.createElement('div');
  w.id        = 'welcome';
  w.className = 'welcome';
  w.innerHTML = `
    <div style="font-size:36px;margin-bottom:12px">🔍</div>
    <div style="font-size:16px;font-weight:500;margin-bottom:6px">Поиск по документам</div>
    <div style="font-size:12px;color:var(--txt2)">Задайте вопрос по загруженным документам</div>
    <div class="welcome-grid">
      <div class="w-card" onclick="setQuery('Краткое содержание всех документов')"><div class="w-card-icon">📋</div><div class="w-card-text">Краткое содержание</div></div>
      <div class="w-card" onclick="setQuery('Ключевые показатели в документах')"><div class="w-card-icon">📊</div><div class="w-card-text">Ключевые показатели</div></div>
    </div>`;
  msgs.appendChild(w);
}

function renderMarkdown(text) {
  if (!text) return '';
  let h = escHtml(text);
  h = h.replace(/```([\s\S]*?)```/g, (_, c) => `<pre>${c}</pre>`);
  h = h.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  h = h.replace(/^### (.+)$/gm,  '<h3>$1</h3>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.+?)\*/g,     '<em>$1</em>');
  h = h.replace(/((\|.+\|\n)+)/g, m => {
    const rows = m.trim().split('\n').filter(r => !/^\|[-| :]+\|$/.test(r));
    return '<table>' + rows.map((r, i) => {
      const cells = r.split('|').filter(c => c.trim());
      const tag   = i === 0 ? 'th' : 'td';
      return `<tr>${cells.map(c => `<${tag}>${c.trim()}</${tag}>`).join('')}</tr>`;
    }).join('') + '</table>';
  });
  h = h.replace(/^[•\-\*] (.+)$/gm, '<li>$1</li>');
  h = h.replace(/(<li>[\s\S]*?<\/li>\n?)+/g, m => `<ul>${m}</ul>`);
  h = h.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  h = h.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
  return h;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function timestamp() {
  return new Date().toLocaleTimeString('ru', { hour:'2-digit', minute:'2-digit' });
}
