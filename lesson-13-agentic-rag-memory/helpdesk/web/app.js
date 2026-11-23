// Frontend of the IT helpdesk. Plain JS, no build step. Talks to server.py through /api/*.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const PARTS = [
  ["system", "System prompt"], ["memory", "Пам'ять"], ["summary", "Переказ (compaction)"],
  ["history", "Історія"], ["question", "Питання"], ["tools", "Результати інструментів"],
];
let me = localStorage.getItem("helpdesk-user") || "emily";
let employees = [];

async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { "Content-Type": "application/json", "X-User": me, ...(opts.headers || {}) } });
  const text = await r.text();
  const data = text && r.headers.get("content-type")?.includes("json") ? JSON.parse(text) : text;
  if (!r.ok) throw new Error(data?.detail || text || r.statusText);
  return data;
}

// ------------------------------------------------------------ header
async function loadStatus() {
  const s = await api("/api/status");
  $("#mode").textContent = `${s.model} · бюджет ${s.budget ?? "—"}` + (s.key ? "" : " · немає ключа!");
}

async function loadEmployees() {
  employees = await api("/api/employees");
  $("#employee").innerHTML = employees.map((e) => `<option value="${e.id}">${esc(e.name)} · ${esc(e.department)}</option>`).join("");
  if (!employees.some((e) => e.id === me)) me = employees[0].id;   // a saved id from an older version
  $("#employee").value = me;
}

$("#employee").addEventListener("change", () => {
  me = $("#employee").value;
  localStorage.setItem("helpdesk-user", me);
  $("#messages").innerHTML = `<div class="msg sys">Ви увійшли як ${esc(employees.find((e) => e.id === me)?.name)}. Розмова цього співробітника продовжується на сервері.</div>`;
  refreshPanels();
});

$("#newSession").addEventListener("click", async () => {
  const r = await api("/api/session/new", { method: "POST" });
  addMsg("sys", r.episode ? `Сесію завершено. В епізодичну пам'ять записано: «${r.episode}»` : "Нова сесія.");
  refreshPanels();
});

// ------------------------------------------------------------ chat
function addMsg(kind, html, extra = "") {
  const div = document.createElement("div");
  div.className = `msg ${kind} ${extra}`;
  div.innerHTML = html;
  $("#messages").appendChild(div);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return div;
}

const withCites = (text) => esc(text).replace(/\[([a-z0-9]+(?:-[a-z0-9]+)+)\]/g, '<span class="cite">$1</span>');

function feedbackBar(node) {
  const bar = document.createElement("div");
  bar.className = "fb";
  bar.innerHTML = `<button title="Добре">👍</button><button title="Погано">👎</button><span class="note"></span>`;
  const [up, down] = bar.querySelectorAll("button");
  const note = bar.querySelector(".note");
  up.onclick = async () => { await api("/api/feedback", { method: "POST", body: JSON.stringify({ rating: "up" }) }); note.textContent = "дякуємо"; };
  down.onclick = () => {
    if (bar.querySelector("input")) return;
    const input = document.createElement("input");
    input.placeholder = "Що не так? Напр. «занадто довго, до трьох речень»";
    const send = document.createElement("button");
    send.textContent = "Надіслати";
    send.onclick = async () => {
      const r = await api("/api/feedback", { method: "POST", body: JSON.stringify({ rating: "down", comment: input.value }) });
      note.textContent = r.accepted ? `правило збережено: ${r.rule}` : `правило не створено: ${r.reason}`;
      input.remove(); send.remove();
      refreshPanels();
    };
    bar.insertBefore(send, note); bar.insertBefore(input, send);
    input.focus();
  };
  node.appendChild(bar);
}

$("#composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("#input").value.trim();
  if (!text) return;
  const isLog = text.split("\n").length > 10;
  addMsg("user", esc(text), isLog ? "log" : "");
  $("#input").value = "";
  $("#busy").hidden = false;
  try {
    const tr = await api("/api/chat", { method: "POST", body: JSON.stringify({ message: text }) });
    const node = addMsg("bot", withCites(tr.answer || "(агент не дав відповіді за ліміт кроків)"));
    feedbackBar(node);
    if (tr.context || tr.calls) renderContext(tr);
    refreshPanels();
  } catch (err) {
    addMsg("err", esc(err.message));
  } finally {
    $("#busy").hidden = true;
  }
});

document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => {
  $("#input").value = c.textContent;
  $("#input").focus();
}));

$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("#composer").requestSubmit();
});

$("#sampleLog").addEventListener("click", async () => {
  const log = await api("/api/sample-log");
  $("#input").value = "VPN не підключається, ось журнал клієнта:\n" + log;
});

// ------------------------------------------------------------ inspector: context of the last turn
function renderContext(tr) {
  tr = { calls: [], critic: [], tokens_in: 0, cached_tokens: 0, usd: 0, compacted: 0, ...tr };
  const p = tr.context?.parts || {};
  const total = PARTS.reduce((s, [k]) => s + (p[k] || 0), 0) || 1;
  const bar = PARTS.filter(([k]) => p[k]).map(([k, label]) =>
    `<div style="flex:${p[k]};background:var(--p-${k})" title="${label}: ${p[k]}"></div>`).join("");
  const legend = PARTS.map(([k, label]) =>
    `<span class="sw" style="background:var(--p-${k})"></span><span>${label}</span><span>${p[k] || 0}</span>`).join("");
  let html = `<h3>Склад першого запиту (оцінка токенів)</h3>
    <div class="stack" role="img" aria-label="Склад запиту">${bar}</div>
    <div class="legend">${legend}<span></span><b>Разом</b><b>${total}</b></div>
    <div class="kpis">
      <div class="kpi"><b>${tr.tokens_in}</b><span>вхідних токенів за хід (API)</span></div>
      <div class="kpi"><b>${tr.cached_tokens}</b><span>з них із кешу</span></div>
      <div class="kpi"><b>$${tr.usd.toFixed(5)}</b><span>ціна ходу</span></div>
      <div class="kpi"><b>${tr.context?.dropped ?? 0}</b><span>повідомлень пропущено</span></div>
      <div class="kpi"><b>${tr.compacted}</b><span>переказано (compaction)</span></div>
      <div class="kpi"><b>${tr.calls.length}</b><span>викликів інструментів</span></div>
    </div>`;
  if (tr.summary) html += `<h3>Переказ старішої частини розмови</h3><div class="card">${esc(tr.summary)}</div>`;
  if (tr.log) {
    html += `<h3>Журнал → субагент</h3>
      <div class="card">${tr.log.lines} рядків: ${tr.log.tokens_raw} → ${tr.log.tokens_trimmed} токенів після trim_log.<br>
      Висновок субагента: <i>${esc(tr.log.analysis)}</i></div>
      <details><summary class="muted">що побачив субагент</summary><pre class="log">${esc(tr.log.trimmed)}</pre></details>`;
  }
  html += `<h3>Кроки агента</h3>` + (tr.calls.length ? tr.calls.map((c) =>
    `<div class="call"><code>${c.step}. ${esc(c.name)}(${esc(c.args)})</code><div class="res">${esc(c.result)}</div></div>`).join("")
    : `<p class="muted">Без інструментів.</p>`);
  if (tr.critic.length) html += `<h3>Критик</h3>` + tr.critic.map((c) => `<div class="warnbox">${esc(c)}</div>`).join("");
  $("#panel-context").innerHTML = html;
}

// ------------------------------------------------------------ inspector: memory, audit, tickets, approvals
async function renderMemory() {
  const m = await api("/api/memory");
  const e = m.employee;
  const boss = employees.find((x) => x.id === e.manager)?.name || "—";
  $("#panel-memory").innerHTML = `
    <h3>Довідник HR (лише читання)</h3>
    <div class="card">${esc(e.name)} · ${esc(e.department)} · офіс ${esc(e.office)} · керівник ${esc(boss)}</div>
    <h3>Semantic: факти</h3>
    ${m.facts.length ? `<table><tr><th>ключ</th><th>значення</th><th>оновлено</th></tr>${m.facts.map((f) =>
      `<tr><td>${esc(f.key)}</td><td>${esc(f.value)}</td><td class="muted">${esc(f.updated.replace("T", " "))}</td></tr>`).join("")}</table>`
      : `<p class="muted">Порожньо.</p>`}
    <h3>Procedural: правила з відгуків</h3>
    ${m.rules.length ? `<ul>${m.rules.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : `<p class="muted">Порожньо.</p>`}
    <h3>Episodic: минулі сесії</h3>
    ${m.episodes.length ? `<ul>${m.episodes.map((x) => `<li><span class="muted">${esc(x.created.slice(0, 16).replace("T", " "))}</span> ${esc(x.summary)}</li>`).join("")}</ul>`
      : `<p class="muted">Порожньо. Епізод з'являється після «Нова сесія».</p>`}
    <h3>Short-term: поточна сесія</h3>
    <div class="card">${m.history_messages} повідомлень в історії${m.summary ? ` · є переказ: ${esc(m.summary)}` : ""}</div>
    <h3>Мої дані</h3>
    <button id="forget" class="btn danger small">Забути мене</button>
    <span class="muted"> видаляє факти, правила й епізоди; у журналі лишається лише факт видалення</span>`;
  $("#forget").onclick = async () => {
    if (!confirm("Видалити всю пам'ять про вас?")) return;
    const r = await api("/api/memory", { method: "DELETE" });
    addMsg("sys", `Пам'ять видалено: ${JSON.stringify(r.deleted)}`);
    refreshPanels();
  };
}

async function renderAudit() {
  const rows = await api("/api/audit");
  $("#panel-audit").innerHTML = `<h3>Хто і що писав у пам'ять</h3>` + (rows.length ? `<table>
    <tr><th>час</th><th>хто</th><th>дія</th><th>було → стало</th></tr>
    ${rows.map((r) => `<tr><td class="muted">${esc(r.ts.slice(5, 16).replace("T", " "))}</td><td>${esc(r.actor)}</td>
      <td>${esc(r.action)}${r.key ? ` <code>${esc(r.key)}</code>` : ""}</td>
      <td>${r.old == null && r.new == null ? '<span class="muted">приховано</span>' : `${esc(r.old ?? "—")} → ${esc(r.new ?? "—")}`}</td></tr>`).join("")}
    </table>` : `<p class="muted">Записів немає.</p>`);
}

async function renderTickets() {
  const r = await api("/api/tickets");
  $("#panel-tickets").innerHTML = `<h3>Заявки</h3>` + (r.tickets.length ? `<table><tr><th>№</th><th>що</th><th>пріоритет</th></tr>
    ${r.tickets.map((t) => `<tr><td>${t.number}</td><td>${esc(t.summary)}</td><td><span class="pill open">${t.priority}</span></td></tr>`).join("")}</table>`
    : `<p class="muted">Заявок немає.</p>`) +
    `<h3>Мої запити на доступ</h3>` + (r.requests.length ? `<table><tr><th>система</th><th>погоджує</th><th>статус</th></tr>
    ${r.requests.map((a) => `<tr><td>${esc(a.system)}</td><td>${esc(employees.find((e) => e.id === a.manager)?.name)}</td>
      <td><span class="pill ${a.status}">${a.status}</span></td></tr>`).join("")}</table>` : `<p class="muted">Запитів немає.</p>`);
}

async function renderApprovals() {
  const rows = await api("/api/approvals");
  const pending = rows.filter((a) => a.status === "pending").length;
  $("#pending").hidden = !pending;
  $("#pending").textContent = pending;
  $("#panel-approvals").innerHTML = `<h3>Запити моїх підлеглих (human-in-the-loop)</h3>` + (rows.length ? rows.map((a) => `
    <div class="call"><b>${esc(a.employee)}</b> просить доступ до <b>${esc(a.system)}</b><div class="res">${esc(a.reason)}</div>
    ${a.status === "pending" ? `<div class="fb"><button data-id="${a.id}" data-ok="1">Погодити</button><button data-id="${a.id}" data-ok="0">Відхилити</button></div>`
      : `<span class="pill ${a.status}">${a.status}</span>`}</div>`).join("")
    : `<p class="muted">Немає запитів. Перейдіть на Sarah Johnson або Michael Davis: вони керівники.</p>`);
  $("#panel-approvals").querySelectorAll("button[data-id]").forEach((b) => {
    b.onclick = async () => {
      await api(`/api/approvals/${b.dataset.id}`, { method: "POST", body: JSON.stringify({ approve: b.dataset.ok === "1" }) });
      refreshPanels();
    };
  });
}

function refreshPanels() {
  Promise.all([renderMemory(), renderAudit(), renderTickets(), renderApprovals(), loadStatus()])
    .catch((err) => addMsg("err", esc(err.message)));
}

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab, .panel").forEach((x) => x.classList.remove("active"));
  t.classList.add("active");
  $(`#panel-${t.dataset.tab}`).classList.add("active");
}));

(async () => {
  await loadEmployees();
  refreshPanels();
})();
