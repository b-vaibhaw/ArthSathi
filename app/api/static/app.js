const API = "";
const USER_ID = (() => {
  const key = "arthasathi_user_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id = "web_" + Math.random().toString(36).slice(2, 10);
    try { localStorage.setItem(key, id); } catch (e) {}
  }
  return id;
})();

const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const userIdEl = document.getElementById("user-id");
userIdEl.value = USER_ID;

function addMessage(role, text, toolCalls) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.textContent = text;
  if (toolCalls && toolCalls.length) {
    const chips = document.createElement("div");
    toolCalls.forEach(tc => {
      const chip = document.createElement("span");
      chip.className = "tool-chip";
      chip.textContent = "🔧 " + tc.tool;
      chips.appendChild(chip);
    });
    div.appendChild(chips);
  }
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

addMessage("system", "Hi! I'm ArthaSathi — ask me about debt payoff, EMI, GST/tax, SIP investing, budgeting, stock basics, or financial news.");

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  addMessage("user", text);
  const thinking = addMessage("bot", "Thinking…");

  try {
    const res = await fetch(`${API}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: USER_ID, message: text }),
    });
    const data = await res.json();
    thinking.remove();
    if (!res.ok) {
      addMessage("bot", "Error: " + (data.detail || res.statusText));
      return;
    }
    addMessage("bot", data.reply, data.tool_calls);
  } catch (err) {
    thinking.remove();
    addMessage("bot", "Network error talking to the server: " + err.message);
  }
});

// ── Profile panel ──────────────────────────────────────────

document.getElementById("toggle-profile").addEventListener("click", () => {
  document.getElementById("profile-panel").classList.toggle("hidden");
});

async function refreshProfile() {
  try {
    const res = await fetch(`${API}/api/profile/${USER_ID}`);
    const profile = await res.json();
    document.getElementById("income-input").value = profile.monthly_income || "";
    const list = document.getElementById("debt-list");
    list.innerHTML = "";
    (profile.debts || []).forEach(d => {
      const el = document.createElement("div");
      el.className = "debt-item";
      el.innerHTML = `<b>${d.name}</b> — ₹${Number(d.principal).toLocaleString()} @ ${d.annual_rate}%<br>
                      <span class="muted">min payment ₹${Number(d.min_payment).toLocaleString()}/mo · ${d.lender_type}</span>`;
      list.appendChild(el);
    });
  } catch (e) { /* profile panel is best-effort; chat still works without it */ }
}

document.getElementById("save-income").addEventListener("click", async () => {
  const val = parseFloat(document.getElementById("income-input").value || "0");
  await fetch(`${API}/api/profile/${USER_ID}/income`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ monthly_income: val }),
  });
  refreshProfile();
});

document.getElementById("add-debt").addEventListener("click", async () => {
  const payload = {
    name: document.getElementById("debt-name").value || "Debt",
    principal: parseFloat(document.getElementById("debt-principal").value || "0"),
    annual_rate: parseFloat(document.getElementById("debt-rate").value || "0"),
    min_payment: parseFloat(document.getElementById("debt-min").value || "0"),
    lender_type: document.getElementById("debt-lender").value,
  };
  await fetch(`${API}/api/profile/${USER_ID}/debt`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  refreshProfile();
});

async function loadTools() {
  try {
    const res = await fetch(`${API}/api/tools`);
    const tools = await res.json();
    const list = document.getElementById("tool-list");
    list.innerHTML = "";
    tools.forEach(t => {
      const li = document.createElement("li");
      li.innerHTML = `<b>${t.name}</b><br>${t.description}`;
      list.appendChild(li);
    });
  } catch (e) { /* ignore */ }
}

refreshProfile();
loadTools();
