<script setup>
import { ref, computed, nextTick, watch, onMounted } from "vue";

const apiBase = import.meta.env.VITE_API_BASE || "";

const modelLabel = ref("本地基座模型");

const messages = ref([
  {
    role: "assistant",
    content:
      "你好，我是本地运行的大模型。输入问题后发送，我会通过 SSE 流式回复。",
  },
]);
const input = ref("");
const sending = ref(false);
const errorText = ref("");
const listRef = ref(null);

onMounted(async () => {
  try {
    const r = await fetch(`${apiBase}/api/health`);
    if (!r.ok) return;
    const j = await r.json();
    if (j.model_name) modelLabel.value = j.model_name;
  } catch {
    /* 后端未启动时忽略 */
  }
});

const canSend = computed(
  () => input.value.trim().length > 0 && !sending.value,
);

async function scrollToBottom() {
  await nextTick();
  const el = listRef.value;
  if (el) el.scrollTop = el.scrollHeight;
}

watch(
  messages,
  () => {
    scrollToBottom();
  },
  { deep: true },
);

function parseSseLines(buffer) {
  const events = [];
  let rest = buffer;
  const parts = buffer.split("\n\n");
  rest = parts.pop() ?? "";
  for (const block of parts) {
    const line = block.split("\n").find((l) => l.startsWith("data: "));
    if (!line) continue;
    const payload = line.slice(6).trim();
    if (payload === "[DONE]") {
      events.push({ done: true });
    } else {
      try {
        events.push({ data: JSON.parse(payload) });
      } catch {
        /* ignore malformed chunk */
      }
    }
  }
  return { events, rest };
}

async function send() {
  const text = input.value.trim();
  if (!text || sending.value) return;

  errorText.value = "";
  messages.value.push({ role: "user", content: text });
  input.value = "";
  sending.value = true;

  const payloadMessages = messages.value
    .filter((m) => m.role !== "assistant" || m.content)
    .map((m) => ({ role: m.role, content: m.content }));

  const assistantIndex = messages.value.length;
  messages.value.push({ role: "assistant", content: "" });

  const url = `${apiBase}/api/chat/stream`;
  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: payloadMessages,
        max_new_tokens: 1024,
        temperature: 0.7,
        top_p: 0.9,
      }),
    });
  } catch (e) {
    errorText.value = `网络错误：${e?.message || e}`;
    messages.value.splice(assistantIndex, 1);
    sending.value = false;
    return;
  }

  if (!res.ok || !res.body) {
    const t = await res.text().catch(() => "");
    errorText.value = `请求失败 ${res.status}：${t || res.statusText}`;
    messages.value.splice(assistantIndex, 1);
    sending.value = false;
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const { events, rest } = parseSseLines(buf);
      buf = rest;
      for (const ev of events) {
        if (ev.done) continue;
        const d = ev.data;
        if (!d) continue;
        if (d.type === "error") {
          errorText.value = d.message || "生成出错";
          continue;
        }
        if (d.type === "token" && d.text) {
          messages.value[assistantIndex].content += d.text;
        }
      }
    }
  } catch (e) {
    errorText.value = `读取流失败：${e?.message || e}`;
  } finally {
    sending.value = false;
    scrollToBottom();
  }
}

function clearChat() {
  if (sending.value) return;
  messages.value = [
    {
      role: "assistant",
      content: "对话已清空，继续提问即可。",
    },
  ];
  errorText.value = "";
}
</script>

<template>
  <div class="app">
    <header class="header">
      <div class="brand">
        <span class="logo">◆</span>
        <div>
          <h1>本地 Qwen 对话</h1>
          <p class="sub">SSE 流式 · 前后端分离</p>
        </div>
      </div>
      <button type="button" class="btn ghost" :disabled="sending" @click="clearChat">
        清空对话
      </button>
    </header>

    <main class="main">
      <div ref="listRef" class="messages">
        <div
          v-for="(m, i) in messages"
          :key="i"
          class="row"
          :class="m.role"
        >
          <div class="avatar">{{ m.role === "user" ? "我" : "AI" }}</div>
          <div class="bubble-wrap">
            <div class="bubble">{{ m.content }}<span v-if="m.role === 'assistant' && sending && i === messages.length - 1" class="cursor">▍</span></div>
          </div>
        </div>
      </div>

      <p v-if="errorText" class="error">{{ errorText }}</p>

      <div class="composer">
        <textarea
          v-model="input"
          rows="3"
          class="input"
          placeholder="输入消息，Enter 发送；Shift+Enter 换行"
          :disabled="sending"
          @keydown.enter.exact.prevent="send"
        />
        <div class="actions">
          <span class="hint">当前模型：{{ modelLabel }}</span>
          <button type="button" class="btn primary" :disabled="!canSend" @click="send">
            {{ sending ? "生成中…" : "发送" }}
          </button>
        </div>
      </div>
    </main>
  </div>
</template>

<style>
:root {
  --bg: #0f1419;
  --panel: #1a2332;
  --border: #2a3544;
  --text: #e7edf5;
  --muted: #8b9cb3;
  --accent: #3b82f6;
  --user-bg: #2563eb;
  --ai-bg: #243044;
  --error: #f87171;
  font-family: "SF Pro Text", system-ui, -apple-system, sans-serif;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
}

.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  max-width: 900px;
  margin: 0 auto;
  padding: 1rem 1.25rem 1.5rem;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.logo {
  font-size: 1.5rem;
  color: var(--accent);
}

h1 {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 600;
}

.sub {
  margin: 0.15rem 0 0;
  font-size: 0.8rem;
  color: var(--muted);
}

.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  margin-top: 1rem;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 0.5rem 0 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.row {
  display: flex;
  gap: 0.65rem;
  align-items: flex-start;
}

.row.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 2rem;
  height: 2rem;
  border-radius: 0.5rem;
  background: var(--panel);
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.7rem;
  color: var(--muted);
  flex-shrink: 0;
}

.row.user .avatar {
  background: #1e3a5f;
  color: #bfdbfe;
}

.bubble-wrap {
  max-width: min(100%, 34rem);
}

.bubble {
  padding: 0.65rem 0.85rem;
  border-radius: 0.75rem;
  line-height: 1.55;
  font-size: 0.95rem;
  white-space: pre-wrap;
  word-break: break-word;
}

.row.assistant .bubble {
  background: var(--ai-bg);
  border: 1px solid var(--border);
  border-top-left-radius: 0.25rem;
}

.row.user .bubble {
  background: var(--user-bg);
  border-top-right-radius: 0.25rem;
}

.cursor {
  display: inline-block;
  animation: blink 1s step-end infinite;
  color: var(--accent);
  margin-left: 1px;
}

@keyframes blink {
  50% {
    opacity: 0;
  }
}

.error {
  color: var(--error);
  font-size: 0.85rem;
  margin: 0 0 0.5rem;
}

.composer {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  padding: 0.75rem;
}

.input {
  width: 100%;
  resize: none;
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 0.6rem 0.75rem;
  background: #121a26;
  color: var(--text);
  font: inherit;
  outline: none;
}

.input:focus {
  border-color: var(--accent);
}

.input:disabled {
  opacity: 0.6;
}

.actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 0.6rem;
  gap: 0.75rem;
}

.hint {
  font-size: 0.75rem;
  color: var(--muted);
}

.btn {
  border: none;
  border-radius: 0.5rem;
  padding: 0.45rem 1rem;
  font: inherit;
  cursor: pointer;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn.primary {
  background: var(--accent);
  color: #fff;
}

.btn.ghost {
  background: transparent;
  color: var(--muted);
  border: 1px solid var(--border);
}

.btn.ghost:hover:not(:disabled) {
  color: var(--text);
  border-color: var(--muted);
}
</style>
