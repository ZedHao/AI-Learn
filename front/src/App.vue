<script setup>
import { ref, computed, watch, nextTick, onMounted } from "vue";

const apiBase = import.meta.env.VITE_API_BASE || "";

const deviceHint = ref("");
const engineStatus = ref("");
/** @type {import('vue').Ref<{ id: string; model_name: string; loaded: boolean }[]>} */
const engines = ref([]);
const selectedEngineId = ref("4b");

const greetLeft = "你好。左侧为无 RAG：仅使用所选模型，不检索财报向量库。";
const greetRight =
  "你好。右侧为 RAG：从 doc/caibao 向量库检索后回答；下方可查看检索 request/response。";

const messagesLeft = ref([{ role: "assistant", content: greetLeft }]);
const messagesRight = ref([{ role: "assistant", content: greetRight }]);
const input = ref("");
const sendingLeft = ref(false);
const sendingRight = ref(false);
const errorLeft = ref("");
const errorRight = ref("");
const listRefLeft = ref(null);
const listRefRight = ref(null);

const ragBackend = ref("faiss");
const ragTopK = ref(5);
const ragStatusLine = ref("");
const ragMetaRight = ref(null);

const sending = computed(() => sendingLeft.value || sendingRight.value);
const engineLoaded = computed(() => {
  const e = engines.value.find((x) => x.id === selectedEngineId.value);
  return !!e?.loaded;
});
const canSend = computed(
  () => input.value.trim().length > 0 && !sending.value && engineLoaded.value,
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
  [messagesLeft, messagesRight],
  () => {
    nextTick(() => {
      scrollPane(listRefLeft.value);
      scrollPane(listRefRight.value);
    });
  },
  { deep: true },
);

async function refreshRagStatus() {
  try {
    const r = await fetch(`${apiBase}/api/rag/status`);
    if (!r.ok) return;
    const j = await r.json();
    const f = j.faiss || {};
    const es = j.elasticsearch || {};
    const fc = f.chunk_count ?? "—";
    const esOk = es.index_ready ? "ES索引就绪" : "ES未建/不可达";
    ragStatusLine.value = `RAG：FAISS 块 ${fc} · ${esOk}`;
  } catch {
    ragStatusLine.value = "";
  }
}

onMounted(async () => {
  try {
    const r = await fetch(`${apiBase}/api/health`);
    if (r.ok) {
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
      const eg = j.engines || {};
      const order =
        Array.isArray(j.engine_order) && j.engine_order.length > 0
          ? j.engine_order
          : Object.keys(eg).sort();
      engines.value = order.map((id) => ({
        id,
        model_name: eg[id]?.model_name || id,
        loaded: !!eg[id]?.loaded,
      }));
      const def = j.default_engine;
      if (def && order.includes(def)) {
        selectedEngineId.value = def;
      } else {
        const firstLoaded = order.find((id) => eg[id]?.loaded);
        selectedEngineId.value = firstLoaded || order[0] || "4b";
      }
      const parts = order.map((id) => {
        const ok = eg[id]?.loaded ? "✓" : "✗";
        return `${id}${ok}`;
      });
      engineStatus.value = parts.length ? `引擎：${parts.join(" · ")}` : "引擎：无";
    }
  } catch {
    /* ignore */
  }
  await refreshRagStatus();
});

/**
 * @param {string} engine
 * @param {import('vue').Ref} messagesRef
 * @param {number} assistantIndex
 * @param {import('vue').Ref<string>} errorRef
 * @param {import('vue').Ref<boolean>} sendingRef
 * @param {null | { enabled: boolean; backend: string; top_k: number }} ragOption
 * @param {import('vue').Ref | null} ragMetaRef 仅右侧传入，用于接收 type=rag SSE
 */
async function streamOne(
  engine,
  payloadMessages,
  messagesRef,
  assistantIndex,
  errorRef,
  sendingRef,
  ragOption,
  ragMetaRef,
) {
  sendingRef.value = true;
  errorRef.value = "";
  if (ragMetaRef) ragMetaRef.value = null;
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
        repetition_penalty: 1.15,
        no_repeat_ngram_size: 4,
        rag:
          ragOption && ragOption.enabled
            ? { enabled: true, backend: ragOption.backend, top_k: ragOption.top_k }
            : null,
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
        if (d.type === "rag" && ragMetaRef) {
          ragMetaRef.value = { request: d.request || {}, response: d.response || {} };
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
      scrollPane(listRefLeft.value);
      scrollPane(listRefRight.value);
    });
  }
}

async function send() {
  const text = input.value.trim();
  if (!text || sending.value || !engineLoaded.value) return;

  errorLeft.value = "";
  errorRight.value = "";

  messagesLeft.value.push({ role: "user", content: text });
  messagesRight.value.push({ role: "user", content: text });
  input.value = "";

  const eng = selectedEngineId.value;
  const payloadL = buildPayload(messagesLeft.value);
  const payloadR = buildPayload(messagesRight.value);

  messagesLeft.value.push({ role: "assistant", content: "" });
  messagesRight.value.push({ role: "assistant", content: "" });
  const idxL = messagesLeft.value.length - 1;
  const idxR = messagesRight.value.length - 1;

  const ragOpt = {
    enabled: true,
    backend: ragBackend.value,
    top_k: Math.min(30, Math.max(1, Number(ragTopK.value) || 5)),
  };

  await Promise.all([
    streamOne(eng, payloadL, messagesLeft, idxL, errorLeft, sendingLeft, null, null),
    streamOne(eng, payloadR, messagesRight, idxR, errorRight, sendingRight, ragOpt, ragMetaRight),
  ]);
}

function clearChat() {
  if (sending.value) return;
  messagesLeft.value = [{ role: "assistant", content: greetLeft }];
  messagesRight.value = [{ role: "assistant", content: greetRight }];
  errorLeft.value = "";
  errorRight.value = "";
  ragMetaRight.value = null;
}
</script>

<template>
  <div class="app">
    <header class="header">
      <div class="brand">
        <span class="logo">◆</span>
        <div>
          <h1>本地 Qwen · 无 RAG / RAG 对比</h1>
          <p class="sub">左栏纯模型 · 右栏财报 RAG（FAISS / ES）· 同一引擎并行 SSE</p>
        </div>
      </div>
      <button type="button" class="btn ghost" :disabled="sending" @click="clearChat">
        清空对话
      </button>
    </header>

    <div class="toolbar">
      <label class="field">
        <span class="field-label">推理引擎</span>
        <select v-model="selectedEngineId" class="select" :disabled="sending || !engines.length">
          <option v-for="e in engines" :key="e.id" :value="e.id" :disabled="!e.loaded">
            {{ e.id }} — {{ e.model_name }}{{ e.loaded ? "" : "（未加载）" }}
          </option>
        </select>
      </label>
      <label class="field">
        <span class="field-label">右侧 RAG 后端</span>
        <select v-model="ragBackend" class="select" :disabled="sending">
          <option value="faiss">FAISS</option>
          <option value="elasticsearch">Elasticsearch</option>
        </select>
      </label>
      <label class="field narrow">
        <span class="field-label">Top-K</span>
        <input v-model.number="ragTopK" type="number" min="1" max="30" class="num" :disabled="sending" />
      </label>
      <button type="button" class="btn ghost sm" :disabled="sending" @click="refreshRagStatus">
        刷新 RAG 状态
      </button>
    </div>
    <p v-if="ragStatusLine" class="rag-hint">{{ ragStatusLine }}</p>

    <div class="dual-wrap">
      <section class="pane pane-left">
        <div class="pane-head">
          <span class="pane-title">无 RAG</span>
          <span v-if="sendingLeft" class="pane-status">生成中…</span>
        </div>
        <div ref="listRefLeft" class="messages">
          <div
            v-for="(m, i) in messagesLeft"
            :key="'L-' + i"
            class="row"
            :class="m.role"
          >
            <div class="avatar">{{ m.role === "user" ? "我" : "AI" }}</div>
            <div class="bubble-wrap">
              <div class="bubble">
                {{ m.content
                }}<span
                  v-if="m.role === 'assistant' && sendingLeft && i === messagesLeft.length - 1"
                  class="cursor cursor-left"
                  >▍</span
                >
              </div>
            </div>
          </div>
        </div>
        <p v-if="errorLeft" class="error">{{ errorLeft }}</p>
      </section>

      <div class="divider" aria-hidden="true" />

      <section class="pane pane-right">
        <div class="pane-head">
          <span class="pane-title">RAG（{{ ragBackend }}）</span>
          <span v-if="sendingRight" class="pane-status">生成中…</span>
        </div>
        <div ref="listRefRight" class="messages">
          <div
            v-for="(m, i) in messagesRight"
            :key="'R-' + i"
            class="row"
            :class="m.role"
          >
            <div class="avatar">{{ m.role === "user" ? "我" : "AI" }}</div>
            <div class="bubble-wrap">
              <div class="bubble">
                {{ m.content
                }}<span
                  v-if="m.role === 'assistant' && sendingRight && i === messagesRight.length - 1"
                  class="cursor cursor-right"
                  >▍</span
                >
              </div>
            </div>
          </div>
        </div>
        <details v-if="ragMetaRight" class="rag-debug" open>
          <summary>RAG 检索参数与返回（SSE）</summary>
          <div class="rag-debug-grid">
            <div>
              <div class="rag-debug-title">request</div>
              <pre class="rag-pre">{{ JSON.stringify(ragMetaRight.request, null, 2) }}</pre>
            </div>
            <div>
              <div class="rag-debug-title">response</div>
              <pre class="rag-pre">{{ JSON.stringify(ragMetaRight.response, null, 2) }}</pre>
            </div>
          </div>
        </details>
        <p v-if="errorRight" class="error">{{ errorRight }}</p>
      </section>
    </div>

    <div class="composer">
      <textarea
        v-model="input"
        rows="3"
        class="input"
        :placeholder="
          engineLoaded
            ? '输入问题，Enter 发送；Shift+Enter 换行（左右并行，仅右侧带 RAG）'
            : '当前所选引擎未加载，请检查模型目录'
        "
        :disabled="sending"
        @keydown.enter.exact.prevent="send"
      />
      <div class="actions">
        <span class="hint">{{ engineStatus || "算力" }} · {{ deviceHint || "—" }}</span>
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

.toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: 0.65rem 1rem;
  margin-top: 0.65rem;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  min-width: 120px;
}

.field.narrow {
  min-width: 72px;
  max-width: 88px;
}

.field-label {
  font-size: 0.65rem;
  color: var(--muted);
}

.select,
.num {
  border: 1px solid var(--border);
  border-radius: 0.45rem;
  padding: 0.35rem 0.5rem;
  background: #121a26;
  color: var(--text);
  font: inherit;
  font-size: 0.82rem;
}

.rag-hint {
  margin: 0.25rem 0 0;
  font-size: 0.72rem;
  color: var(--muted);
}

.dual-wrap {
  display: flex;
  flex: 1;
  min-height: 0;
  gap: 0;
  margin-top: 0.5rem;
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

.pane-left {
  border-top: 3px solid var(--accent-4b);
}

.pane-right {
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

.pane-left .pane-title {
  color: var(--accent-4b);
}

.pane-right .pane-title {
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

.rag-debug {
  border-top: 1px solid var(--border);
  padding: 0.45rem 0.55rem 0.55rem;
  background: #121a26;
  font-size: 0.72rem;
  color: var(--muted);
}

.rag-debug summary {
  cursor: pointer;
  color: var(--text);
  font-weight: 500;
}

.rag-debug-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  margin-top: 0.45rem;
}

@media (max-width: 900px) {
  .rag-debug-grid {
    grid-template-columns: 1fr;
  }
}

.rag-debug-title {
  font-size: 0.65rem;
  color: var(--muted);
  margin-bottom: 0.25rem;
}

.rag-pre {
  margin: 0;
  padding: 0.4rem;
  border-radius: 0.35rem;
  background: #0f1419;
  border: 1px solid var(--border);
  color: #c4d0e0;
  font-size: 0.62rem;
  line-height: 1.35;
  max-height: 200px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
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
  margin-left: 1px;
}

.cursor-left {
  color: var(--accent-4b);
}

.cursor-right {
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

.btn.sm {
  padding: 0.28rem 0.55rem;
  font-size: 0.78rem;
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
