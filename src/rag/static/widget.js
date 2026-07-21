/**
 * PC 零件口碑 — 浮動聊天 Widget
 *
 * 嵌入方式：
 *   <script src="http://localhost:8000/static/widget.js"></script>
 *
 * 可選設定（在 <script> 前宣告）：
 *   window.PCRAG_API_BASE = "https://your-server.com";  // 預設 http://localhost:8000
 */
(function () {
  "use strict";

  const API_BASE = (window.PCRAG_API_BASE || "http://localhost:8001").replace(/\/$/, "");
  const API_URL  = API_BASE + "/api/chat";

  const EXAMPLES = [
    "RTX5070 值得買嗎？",
    "RTX5070 跟 RX9070XT 哪個比較推薦？",
    "7800X3D 有什麼缺點？",
    "中階顯示卡推薦",
    "B650M 主機板怎麼樣？",
  ];

  // ── CSS (injected into Shadow DOM) ────────────────────────────────────────

  const CSS = `
:host {
  position: fixed;
  bottom: 24px;
  right: 24px;
  z-index: 2147483647;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC",
               Arial, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  color-scheme: light;
}

*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

/* ── Layout wrapper ───────────────────────────────────── */
.pcrag-widget {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 12px;
}

/* ── Trigger button ───────────────────────────────────── */
.pcrag-trigger {
  width: 52px;
  height: 52px;
  border-radius: 50%;
  background: #6366f1;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 20px rgba(99,102,241,.30);
  transition: transform .15s ease, box-shadow .15s ease;
  flex-shrink: 0;
}
.pcrag-trigger:hover {
  transform: scale(1.07);
  box-shadow: 0 6px 24px rgba(99,102,241,.40);
}
.pcrag-trigger svg { display: block; }

/* ── Chat panel ───────────────────────────────────────── */
.pcrag-panel {
  position: absolute;
  bottom: calc(100% + 12px);
  right: 0;
  width: 360px;
  height: 520px;
  background: #ffffff;
  border: 1px solid #e5e5ea;
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 0 12px 40px rgba(0,0,0,.12);
  /* open/close animation */
  opacity: 0;
  pointer-events: none;
  transform: scale(0.96) translateY(8px);
  transform-origin: bottom right;
  transition: opacity .18s ease, transform .18s ease;
}
.pcrag-panel.pcrag-open {
  opacity: 1;
  pointer-events: all;
  transform: scale(1) translateY(0);
}

/* ── Header ───────────────────────────────────────────── */
.pcrag-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 13px 16px;
  border-bottom: 1px solid #e5e5ea;
  background: #ffffff;
  flex-shrink: 0;
}
.pcrag-header-text {}
.pcrag-title {
  color: #1a1a1a;
  font-weight: 600;
  font-size: 13.5px;
  letter-spacing: .02em;
  display: block;
}
.pcrag-subtitle {
  color: #999999;
  font-size: 11px;
  margin-top: 1px;
  display: block;
}
.pcrag-close {
  background: none;
  border: none;
  cursor: pointer;
  color: #999999;
  padding: 6px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: color .12s, background .12s;
  flex-shrink: 0;
}
.pcrag-close:hover {
  color: #33333a;
  background: #f4f4f7;
}

/* ── Messages area ────────────────────────────────────── */
.pcrag-messages {
  flex: 1;
  overflow-y: auto;
  padding: 14px 12px;
  background: #f4f4f7;
  display: flex;
  flex-direction: column;
  gap: 10px;
  scroll-behavior: smooth;
}
.pcrag-messages::-webkit-scrollbar { width: 3px; }
.pcrag-messages::-webkit-scrollbar-track { background: transparent; }
.pcrag-messages::-webkit-scrollbar-thumb {
  background: #e5e5ea;
  border-radius: 2px;
}

/* ── Welcome / chips ──────────────────────────────────── */
.pcrag-welcome {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 4px 0;
}
.pcrag-welcome-label {
  color: #999999;
  font-size: 11.5px;
}
.pcrag-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.pcrag-chip {
  background: #ffffff;
  border: 1px solid #e5e5ea;
  border-radius: 20px;
  color: #999999;
  font-size: 12px;
  padding: 5px 12px;
  cursor: pointer;
  font-family: inherit;
  transition: border-color .12s, color .12s, background .12s;
  text-align: left;
}
.pcrag-chip:hover {
  border-color: #6366f1;
  color: #6366f1;
  background: rgba(99,102,241,.06);
}

/* ── Message bubbles ──────────────────────────────────── */
.pcrag-msg {
  max-width: 88%;
  padding: 9px 13px;
  font-size: 13.5px;
  line-height: 1.6;
  word-break: break-word;
  white-space: pre-wrap;
}
.pcrag-msg-ai {
  align-self: flex-start;
  background: #f4f4f7;
  color: #33333a;
  border: 1px solid #e5e5ea;
  border-left: 2px solid #6366f1;
  border-radius: 4px 12px 12px 4px;
}
.pcrag-msg-user {
  align-self: flex-end;
  background: #6366f1;
  color: #ffffff;
  border-radius: 12px 4px 12px 12px;
}

/* Typing indicator — three bouncing dots */
.pcrag-typing {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 2px 0;
}
.pcrag-typing span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #6366f1;
  display: block;
  animation: pcrag-bounce 1.1s ease-in-out infinite;
}
.pcrag-typing span:nth-child(2) { animation-delay: .18s; }
.pcrag-typing span:nth-child(3) { animation-delay: .36s; }
@keyframes pcrag-bounce {
  0%, 60%, 100% { transform: translateY(0);   opacity: .3; }
  30%            { transform: translateY(-5px); opacity: 1; }
}

/* ── Input row ────────────────────────────────────────── */
.pcrag-input-row {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  padding: 10px 12px;
  border-top: 1px solid #e5e5ea;
  background: #ffffff;
  flex-shrink: 0;
}
.pcrag-input {
  flex: 1;
  min-height: 38px;
  max-height: 120px;
  background: #f4f4f7;
  border: 1px solid #e5e5ea;
  border-radius: 10px;
  color: #33333a;
  font-family: inherit;
  font-size: 13px;
  padding: 9px 12px;
  resize: none;
  outline: none;
  overflow-y: auto;
  transition: border-color .12s;
  line-height: 1.5;
}
.pcrag-input::placeholder { color: #999999; }
.pcrag-input:focus         { border-color: #6366f1; }

.pcrag-send {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: #6366f1;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: transform .12s, opacity .12s;
}
.pcrag-send:hover    { transform: scale(1.07); }
.pcrag-send:disabled { opacity: .35; cursor: not-allowed; transform: none; }
`;

  // ── HTML template ──────────────────────────────────────────────────────────

  function buildHTML() {
    const chips = EXAMPLES
      .map(ex => `<button class="pcrag-chip" data-q="${escHtml(ex)}">${escHtml(ex)}</button>`)
      .join("");

    return `
<style>${CSS}</style>
<div class="pcrag-widget">

  <!-- Trigger -->
  <button class="pcrag-trigger" aria-label="開啟 PC 零件口碑查詢">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
         stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </svg>
  </button>

  <!-- Panel -->
  <div class="pcrag-panel" role="dialog" aria-label="PC 零件口碑聊天">

    <!-- Header -->
    <div class="pcrag-header">
      <div class="pcrag-header-text">
        <span class="pcrag-title">PC 零件口碑</span>
        <span class="pcrag-subtitle">PTT · 巴哈姆特 論壇評價</span>
      </div>
      <button class="pcrag-close" aria-label="關閉">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M18 6 6 18M6 6l12 12"/>
        </svg>
      </button>
    </div>

    <!-- Messages -->
    <div class="pcrag-messages">
      <div class="pcrag-welcome">
        <span class="pcrag-welcome-label">詢問任何電腦零件口碑，例如：</span>
        <div class="pcrag-chips">${chips}</div>
      </div>
    </div>

    <!-- Input -->
    <div class="pcrag-input-row">
      <textarea class="pcrag-input" placeholder="輸入型號或問題…" rows="1"
                autocomplete="off" spellcheck="false"
                aria-label="訊息輸入"></textarea>
      <button class="pcrag-send" aria-label="送出">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
             stroke="#ffffff" stroke-width="2.2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 2 11 13M22 2l-7 20-4-9-9-4 20-7z"/>
        </svg>
      </button>
    </div>

  </div>
</div>`;
  }

  // ── State ──────────────────────────────────────────────────────────────────

  // history = all completed exchanges: [{role, content}, ...]
  // The current in-flight user query is NOT in history yet when we POST.
  let history     = [];
  let isStreaming = false;

  // Shadow DOM refs
  let shadow, panel, messagesEl, inputEl, sendBtn, welcomeEl;

  // ── Init ───────────────────────────────────────────────────────────────────

  function init() {
    const host = document.createElement("div");
    document.body.appendChild(host);
    shadow = host.attachShadow({ mode: "open" });
    shadow.innerHTML = buildHTML();

    panel      = shadow.querySelector(".pcrag-panel");
    messagesEl = shadow.querySelector(".pcrag-messages");
    inputEl    = shadow.querySelector(".pcrag-input");
    sendBtn    = shadow.querySelector(".pcrag-send");
    welcomeEl  = shadow.querySelector(".pcrag-welcome");

    shadow.querySelector(".pcrag-trigger").addEventListener("click", togglePanel);
    shadow.querySelector(".pcrag-close").addEventListener("click", closePanel);

    sendBtn.addEventListener("click", submit);

    inputEl.addEventListener("keydown", function (e) {
      // e.isComposing = true 代表輸入法正在組字（注音選字中），此時 Enter 是確認選字，不是送出
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        submit();
      }
    });

    inputEl.addEventListener("input", autoResize);

    shadow.querySelectorAll(".pcrag-chip").forEach(function (chip) {
      chip.addEventListener("click", function () {
        if (isStreaming) return;
        inputEl.value = chip.dataset.q;
        autoResize();
        submit();
      });
    });
  }

  // ── Panel toggle ───────────────────────────────────────────────────────────

  function togglePanel() {
    if (panel.classList.contains("pcrag-open")) {
      closePanel();
    } else {
      panel.classList.add("pcrag-open");
      inputEl.focus();
    }
  }

  function closePanel() {
    panel.classList.remove("pcrag-open");
  }

  // ── Input resize ───────────────────────────────────────────────────────────

  function autoResize() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
  }

  // ── Submit ─────────────────────────────────────────────────────────────────

  async function submit() {
    const query = inputEl.value.trim();
    if (!query || isStreaming) return;

    // Hide welcome chips on first message
    if (welcomeEl) {
      welcomeEl.remove();
      welcomeEl = null;
    }

    // Show user bubble
    appendBubble(query, "user");

    // Snapshot history before adding current turn (server expects previous context)
    const historySnapshot = history.slice();

    // Clear input and shrink textarea back to one row
    inputEl.value = "";
    autoResize();

    // Lock UI
    isStreaming = true;
    sendBtn.disabled = true;

    // Create AI bubble with typing indicator
    const aiBubble = appendBubble("", "ai");
    aiBubble.classList.add("pcrag-msg-ai--loading");
    aiBubble.innerHTML = '<div class="pcrag-typing"><span></span><span></span><span></span></div>';

    let finalText = "";

    try {
      const resp = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, history: historySnapshot }),
      });

      if (!resp.ok) throw new Error(`伺服器錯誤 HTTP ${resp.status}`);

      const reader  = resp.body.getReader();
      const decoder = new TextDecoder();
      let   buf     = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buf += decoder.decode(value, { stream: true });

        // SSE lines end with \n; events are separated by \n\n
        const lines = buf.split("\n");
        buf = lines.pop(); // keep partial last line

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") break;

          let parsed;
          try { parsed = JSON.parse(raw); } catch { continue; }

          if (parsed.error) throw new Error(parsed.error);

          if (parsed.text !== undefined) {
            finalText = parsed.text;
            // Remove loading pulse on first real token
            aiBubble.classList.remove("pcrag-msg-ai--loading");
            aiBubble.textContent = finalText;
            scrollToBottom();
          }
        }
      }

      // Commit completed exchange to history
      history.push({ role: "user",      content: query     });
      history.push({ role: "assistant", content: finalText });

      // Keep last 12 messages (6 turns) to mirror server-side MAX_HISTORY
      if (history.length > 12) history = history.slice(-12);

    } catch (err) {
      aiBubble.classList.remove("pcrag-msg-ai--loading");
      aiBubble.textContent = "發生錯誤：" + err.message;
    } finally {
      isStreaming = false;
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  function appendBubble(text, role) {
    const el = document.createElement("div");
    el.className = "pcrag-msg " + (role === "user" ? "pcrag-msg-user" : "pcrag-msg-ai");
    el.textContent = text;
    messagesEl.appendChild(el);
    scrollToBottom();
    return el;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function escHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ── Bootstrap ──────────────────────────────────────────────────────────────

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
