<script setup>
import { ref, computed, watch, nextTick, onMounted } from "vue";

const apiBase = import.meta.env.VITE_API_BASE || "";

const deviceHint = ref("");
const engineStatus = ref("");

const greet =
  "你好。下方同一问题会同时发给左侧 4B 与右侧 8B，流式输出可对比。";

const messages4b = ref([{ role: "assistant", content: greet }]);
const messages8b = ref([{ role: "assistant", content: greet }]);

const input = ref("");
const sending4b = ref(false);
const sending8b = ref(false);
const error4b = ref("");
const error8b = ref("");

const listRef4b = ref(null);
const listRef8b = ref(null);

const sending = computed(() => sending4b.value || sending8b.value);
const canSend = computed(
  () => input.value.trim().length > 0 && !sending.value,
);

function buildPayload(list) {
  return list
    .filter((m) => m.role !== "assistant" || m.content)
    .map((m) => ({ role: m.role, content: m.content }));
}

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
        /* ignore */
      }
    }
  }
  return { events, rest };
}

function scrollPane(el) {
  if (el) el.scrollTop = el.scrollHeight;
}

watch(
  [messages4b, messages8b],
  () => {
    nextTick(() => {
      scrollPane(listRef4b.value);
      scrollPane(listRef8b.value);
    });
  },
  { deep: true },
);

onMounted(async () => {
  try {
    const r = await fetch(`${apiBase}/api/health`);
    if (!r.ok) return;
    const j = await r.json();
    const d = j.device;
    if (d && typeof d === "object") {
      if (d.cuda_device_name) {
        deviceHint.value = `${d.label_zh || d.torch_device} · ${d.cuda_device_name}`;
      } else if (d.torch_device === "mps") {
        deviceHint.value = `${d.label_zh || "Apple Silicon(MPS)"} · ${d.machine || ""}`;
      } else {
        deviceHint.value = `${d.label_zh || d.torch_device} · ${d.machine || ""}`;
      }
    }
    const eg = j.engines;
    if (eg && typeof eg === "object") {
      const a = eg["4b"]?.loaded ? "4B✓" : "4B✗";
      const b = eg["8b"]?.loaded ? "8B✓" : "8B✗";
      engineStatus.value = `服务：${a} · ${b}`;
    }
  } catch {
    /* ignore */
  }
});

/**
 * @param {string} engine
 * @param {import('vue').Ref} messagesRef
 * @param {number} assistantIndex
 * @param {import('vue').Ref<string>} errorRef
 * @param {import('vue').Ref<boolean>} sendingRef
 */
async function streamOne(engine, payloadMessages, messagesRef, assistantIndex, errorRef, sendingRef) {
  sendingRef.value = true;
  errorRef.value = "";
  const url = `${apiBase}/api/chat/stream`;
  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        engine,
        messages: payloadMessages,
        max_new_tokens: 1024,
        temperature: 0.7,
        top_p: 0.9,
      }),
    });
  } catch (e) {
    errorRef.value = `[${engine}] 网络错误：${e?.message || e}`;
    if (!messagesRef.value[assistantIndex]?.content) {
      messagesRef.value.splice(assistantIndex, 1);
    }
    sendingRef.value = false;
    return;
  }

  if (!res.ok || !res.body) {
    const t = await res.text().catch(() => "");
    errorRef.value = `[${engine}] ${res.status}：${t || res.statusText}`;
    if (!messagesRef.value[assistantIndex]?.content) {
      messagesRef.value.splice(assistantIndex, 1);
    }
    sendingRef.value = false;
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
          errorRef.value = d.message || "生成出错";
          continue;
        }
        if (d.type === "token" && d.text) {
          messagesRef.value[assistantIndex].content += d.text;
        }
      }
    }
  } catch (e) {
    errorRef.value = `[${engine}] 读取流失败：${e?.message || e}`;
  } finally {
    sendingRef.value = false;
    nextTick(() => {
      scrollPane(listRef4b.value);
      scrollPane(listRef8b.value);
    });
  }
}

async function send() {
  const text = input.value.trim();
  if (!text || sending.value) return;

  error4b.value = "";
  error8b.value = "";

  messages4b.value.push({ role: "user", content: text });
  messages8b.value.push({ role: "user", content: text });
  input.value = "";

  const payload4b = buildPayload(messages4b.value);
  const payload8b = buildPayload(messages8b.value);

  messages4b.value.push({ role: "assistant", content: "" });
  messages8b.value.push({ role: "assistant", content: "" });
  const idx4 = messages4b.value.length - 1;
  const idx8 = messages8b.value.length - 1;

  await Promise.all([
    streamOne("4b", payload4b, messages4b, idx4, error4b, sending4b),
    streamOne("8b", payload8b, messages8b, idx8, error8b, sending8b),
  ]);
}

function clearChat() {
  if (sending.value) return;
  messages4b.value = [{ role: "assistant", content: greet }];
  messages8b.value = [{ role: "assistant", content: greet }];
  error4b.value = "";
  error8b.value = "";
}
</script>

<template>
  <div class="app">
    <header class="header">
      <div class="brand">
        <span class="logo">◆</span>
        <div>
          <h1>本地 Qwen 双模型对比</h1>
          <p class="sub">一次提问 · 左 4B 右 8B · SSE 并行流式</p>
        </div>
      </div>
      <button type="button" class="btn ghost" :disabled="sending" @click="clearChat">
        清空对话
      </button>
    </header>

    <div class="dual-wrap">
      <section class="pane pane-4b">
        <div class="pane-head">
          <span class="pane-title">Qwen3-4B</span>
          <span v-if="sending4b" class="pane-status">生成中…</span>
        </div>
        <div ref="listRef4b" class="messages">
          <div
            v-for="(m, i) in messages4b"
            :key="'4b-' + i"
            class="row"
            :class="m.role"
          >
            <div class="avatar">{{ m.role === "user" ? "我" : "4B" }}</div>
            <div class="bubble-wrap">
              <div class="bubble">
                {{ m.content
                }}<span
                  v-if="m.role === 'assistant' && sending4b && i === messages4b.length - 1"
                  class="cursor"
                  >▍</span
                >
              </div>
            </div>
          </div>
        </div>
        <p v-if="error4b" class="error">{{ error4b }}</p>
      </section>

      <div class="divider" aria-hidden="true" />

      <section class="pane pane-8b">
        <div class="pane-head">
          <span class="pane-title">Qwen3-8B</span>
          <span v-if="sending8b" class="pane-status">生成中…</span>
        </div>
        <div ref="listRef8b" class="messages">
          <div
            v-for="(m, i) in messages8b"
            :key="'8b-' + i"
            class="row"
            :class="m.role"
          >
            <div class="avatar">{{ m.role === "user" ? "我" : "8B" }}</div>
            <div class="bubble-wrap">
              <div class="bubble">
                {{ m.content
                }}<span
                  v-if="m.role === 'assistant' && sending8b && i === messages8b.length - 1"
                  class="cursor"
                  >▍</span
                >
              </div>
            </div>
          </div>
        </div>
        <p v-if="error8b" class="error">{{ error8b }}</p>
      </section>
    </div>

    <div class="composer">
      <textarea
        v-model="input"
        rows="3"
        class="input"
        placeholder="输入问题，Enter 发送；Shift+Enter 换行（将同时请求 4B 与 8B）"
        :disabled="sending"
        @keydown.enter.exact.prevent="send"
      />
      <div class="actions">
        <span class="hint"
          >{{ engineStatus || "算力" }} · {{ deviceHint || "—" }}</span
        >
        <button type="button" class="btn primary" :disabled="!canSend" @click="send">
          {{ sending ? "生成中…" : "发送" }}
        </button>
      </div>
    </div>
  </div>
</template>

<style>
:root {
  --bg: #0f1419;
  --panel: #1a2332;
  --border: #2a3544;
  --text: #e7edf5;
  --muted: #8b9cb3;
  --accent-4b: #3b82f6;
  --accent-8b: #a78bfa;
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
  max-width: 1280px;
  margin: 0 auto;
  padding: 1rem 1rem 1.5rem;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding-bottom: 0.85rem;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.logo {
  font-size: 1.5rem;
  color: var(--accent-4b);
}

h1 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
}

.sub {
  margin: 0.15rem 0 0;
  font-size: 0.78rem;
  color: var(--muted);
}

.dual-wrap {
  display: flex;
  flex: 1;
  min-height: 0;
  gap: 0;
  margin-top: 0.75rem;
}

.divider {
  width: 1px;
  background: var(--border);
  flex-shrink: 0;
  margin: 0 0.35rem;
}

.pane {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  border: 1px solid var(--border);
  border-radius: 0.65rem;
  background: var(--panel);
  overflow: hidden;
}

.pane-4b {
  border-top: 3px solid var(--accent-4b);
}

.pane-8b {
  border-top: 3px solid var(--accent-8b);
}

.pane-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.45rem 0.65rem;
  border-bottom: 1px solid var(--border);
  background: #141c28;
  flex-shrink: 0;
}

.pane-title {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text);
}

.pane-4b .pane-title {
  color: var(--accent-4b);
}

.pane-8b .pane-title {
  color: var(--accent-8b);
}

.pane-status {
  font-size: 0.72rem;
  color: var(--muted);
}

.messages {
  flex: 1;
  min-height: 220px;
  max-height: min(52vh, 520px);
  overflow-y: auto;
  padding: 0.5rem 0.55rem 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.row {
  display: flex;
  gap: 0.5rem;
  align-items: flex-start;
}

.row.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 1.85rem;
  height: 1.85rem;
  border-radius: 0.45rem;
  background: var(--panel);
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.62rem;
  color: var(--muted);
  flex-shrink: 0;
}

.row.user .avatar {
  background: #1e3a5f;
  color: #bfdbfe;
}

.bubble-wrap {
  max-width: min(100%, 100%);
}

.bubble {
  padding: 0.55rem 0.7rem;
  border-radius: 0.65rem;
  line-height: 1.5;
  font-size: 0.88rem;
  white-space: pre-wrap;
  word-break: break-word;
}

.row.assistant .bubble {
  background: var(--ai-bg);
  border: 1px solid var(--border);
  border-top-left-radius: 0.2rem;
}

.row.user .bubble {
  background: var(--user-bg);
  border-top-right-radius: 0.2rem;
}

.cursor {
  display: inline-block;
  animation: blink 1s step-end infinite;
  color: var(--accent-4b);
  margin-left: 1px;
}

.pane-8b .cursor {
  color: var(--accent-8b);
}

@keyframes blink {
  50% {
    opacity: 0;
  }
}

.error {
  color: var(--error);
  font-size: 0.78rem;
  margin: 0;
  padding: 0.35rem 0.55rem 0.5rem;
  border-top: 1px solid var(--border);
}

.composer {
  flex-shrink: 0;
  margin-top: 0.75rem;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  padding: 0.65rem;
}

.input {
  width: 100%;
  resize: none;
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 0.55rem 0.65rem;
  background: #121a26;
  color: var(--text);
  font: inherit;
  outline: none;
}

.input:focus {
  border-color: var(--accent-4b);
}

.input:disabled {
  opacity: 0.6;
}

.actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 0.5rem;
  gap: 0.65rem;
}

.hint {
  font-size: 0.72rem;
  color: var(--muted);
}

.btn {
  border: none;
  border-radius: 0.5rem;
  padding: 0.42rem 0.95rem;
  font: inherit;
  cursor: pointer;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn.primary {
  background: linear-gradient(90deg, var(--accent-4b), var(--accent-8b));
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

@media (max-width: 720px) {
  .dual-wrap {
    flex-direction: column;
  }

  .divider {
    width: 100%;
    height: 1px;
    margin: 0.5rem 0;
  }

  .messages {
    max-height: 38vh;
  }
}
</style>
