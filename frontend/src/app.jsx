import React, { useEffect, useRef, useState } from "react"
import * as api from "./api"


import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";

/* ------- Tiny typing indicator ------- */
function TypingDots() {
  return (
    <div style={{
      display: "inline-flex", gap: 6, alignItems: "center",
      padding: "10px 12px", background: "#ffffff",
      border: "1px solid #e5e7eb", borderRadius: 12,
      boxShadow: "0 1px 2px rgba(16,24,40,.05)"
    }}>
      <span style={{ width: 6, height: 6, background: "#9ca3af", borderRadius: "999px", animation: "blink 1s infinite" }} />
      <span style={{ width: 6, height: 6, background: "#9ca3af", borderRadius: "999px", animation: "blink 1s .15s infinite" }} />
      <span style={{ width: 6, height: 6, background: "#9ca3af", borderRadius: "999px", animation: "blink 1s .3s infinite" }} />
      <style>{`
        @keyframes blink {
          0% { opacity: .3; transform: translateY(0); }
          50% { opacity: 1; transform: translateY(-2px); }
          100% { opacity: .3; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

/* ------- Layout ------- */
function Sidebar({ tab, setTab }) {
  const items = [
    { key: "chat",     label: "Chat",          icon: "💬" },
    { key: "upload",   label: "Upload",        icon: "📤" },
    { key: "docs",     label: "Documents",     icon: "📚" },
    { key: "db",       label: "DB Info",       icon: "🗄️" },
    { key: "rerank",   label: "Rerank Debug",  icon: "🔎" },
    { key: "memory",   label: "Memory",        icon: "🧠" },
    { key: "settings", label: "Settings",      icon: "⚙️" },
  ];
  return (
    <aside className="sidebar">
      <div className="brand">Agentic RAG</div>
      <div className="muted" style={{ fontSize: 12 }}>Dashboard</div>
      <nav className="nav" style={{ display: "grid", gap: 6, marginTop: 8 }}>
        {items.map(it => (
          <a key={it.key} onClick={() => setTab(it.key)}
             className={tab === it.key ? "active" : ""} style={{ cursor: "pointer" }}>
            <span>{it.icon}</span><span>{it.label}</span>
          </a>
        ))}
      </nav>
      <div style={{ marginTop: "auto" }} className="muted">
        <div className="card" style={{ fontSize: 12 }}>
          <div style={{ fontWeight: 600 }}>Tips</div>
          Upload PDFs/DOCX, then ask:
          <ul>
            <li>“Summarize the docs I uploaded”</li>
            <li>“Show last 5 memories (SQL)”</li>
          </ul>
        </div>
      </div>
    </aside>
  );
}

function Topbar({ right = null }) {
  return (
    <div className="topbar">
      <div className="muted">Build your perfect Agentic RAG</div>
      <div>{right}</div>
    </div>
  );
}

/* ------- Copy Button ------- */
function CopyButton({ getText, label = "Copy" }) {
  const [ok, setOk] = useState(false);
  return (
    <button
      type="button"
      className="copybtn"
      onClick={async () => {
        const t = typeof getText === "function" ? getText() : String(getText || "");
        await navigator.clipboard.writeText(t);
        setOk(true);
        setTimeout(() => setOk(false), 1000);
      }}
      title="Copy to clipboard"
    >
      {ok ? "Copied" : label} <span className="copyicon">⧉</span>
    </button>
  );
}

/* ------- Chat (WhatsApp style + typing) ------- */
function ChatPage() {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState([]); // [{role:'user'|'assistant'|'typing', ...}]
  const endRef = useRef(null);

  async function onAsk(e) {
    e.preventDefault();
    const text = q.trim();
    if (!text || busy) return;

    // 1) Push user message immediately (right bubble)
    setLog(l => [...l, { role: "user", text }]);
    setQ("");
    setBusy(true);

    // 2) Show typing indicator from assistant (left bubble)
    const typingId = Math.random().toString(36).slice(2);
    setLog(l => [...l, { role: "typing", id: typingId }]);

    try {
      // 3) Call backend with payload expected by /ask
      const res = await api.ask({ query: text, session_id: "default" });

      // 4) Replace typing bubble with the real response
      setLog(l => {
        const next = l.filter(m => m.id !== typingId);
        next.push({
          role: "assistant",
          answer: res?.answer || "(no response)",
          citations: res?.citations || [],
          route: res?.route || "RAG"
        });
        return next;
      });
    } catch (err) {
      setLog(l => {
        const next = l.filter(m => m.id !== typingId);
        next.push({ role: "assistant", answer: `**Error:** ${String(err)}` });
        return next;
      });
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [log]);

  return (
    <div className="content">
      <Topbar right={<span className="badge">Chat</span>} />

      <div className="chat-wrap chat-col">
        {log.map((m, i) => (
          <div key={i} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {m.role === "user" ? (
              <>
                <div className="msg user">{m.text}</div>
                <div className="meta right">
                  You · <CopyButton getText={m.text} label="Copy" />
                </div>
              </>
            ) : m.role === "typing" ? (
              <div className="msg ai">
                <TypingDots />
              </div>
            ) : (
              <>
                <div className="msg ai">
                  {m.route && <div className="route" style={{ marginBottom: 6 }}>Route: {m.route}</div>}
                  <ReactMarkdown
                    children={m.answer || "No response received"}
                    components={{
                      code({ inline, className, children, ...props }) {
                        const raw = String(children);
                        const match = /language-(\w+)/.exec(className || "");
                        if (!inline && match) {
                          const lang = match[1];
                          return (
                            <div>
                              <div className="codebar">
                                <div>{lang.toUpperCase()} snippet</div>
                                <CopyButton getText={raw} />
                              </div>
                              <div className="codewrap">
                                <SyntaxHighlighter style={oneLight} language={lang} PreTag="div" {...props}>
                                  {raw.replace(/\n$/, "")}
                                </SyntaxHighlighter>
                              </div>
                            </div>
                          );
                        }
                        // inline code
                        return (
                          <code
                            style={{
                              background: "#e2e8f0",
                              padding: "2px 6px",
                              borderRadius: "6px",
                              fontSize: "0.92em",
                              color: "#111827",
                            }}
                            {...props}
                          >
                            {children}
                          </code>
                        );
                      },
                    }}
                  />
                  {m.citations?.length ? (
                    <details style={{ marginTop: 8 }}>
                      <summary style={{ cursor: "pointer", color: "#64748b" }}>Sources</summary>
                      <ul style={{ margin: 0, paddingLeft: 16 }}>
                        {m.citations.map((c, idx) => (
                          <li key={idx} style={{ color: "#64748b" }}>
                            <code>{c.doc_id}</code>
                            {c.meta?.source ? ` — ${c.meta.source}` : ""}
                            {typeof c.rrf === "number" ? ` — RRF: ${c.rrf.toFixed(4)}` : ""}
                          </li>
                        ))}
                      </ul>
                    </details>
                  ) : null}
                </div>
                <div className="meta left">Assistant</div>
              </>
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form onSubmit={onAsk} className="inputbar">
        <input
          className="input"
          value={q}
          onChange={e => setQ(e.target.value)}
          placeholder="Send a message… (supports Markdown and ```python``` code)"
        />
        <button className="btn primary" disabled={busy}>
          {busy ? "Thinking…" : "Send"}
        </button>
      </form>
    </div>
  );
}

/* ------- Upload ------- */
function UploadPage() {
  const ref = useRef(null);
  const [last, setLast] = useState(null);
  async function onUpload() {
    const files = ref.current?.files;
    if (!files || !files.length) return alert("Choose files first.");
    try {
      const res = await api.uploadFiles(files);
      setLast(res);
      alert(`Uploaded: ${res.uploaded}, Ingested: ${res.ingested}`);
      ref.current.value = "";
    } catch (e) {
      alert(e.message);
    }
  }
  return (
    <div className="content">
      <Topbar right={<span className="badge">Upload</span>} />
      <div className="panel" style={{ margin: 16 }}>
        <h3 style={{ marginTop: 0 }}>Upload & Ingest</h3>
        <p className="muted">TXT, MD, CSV, LOG, PDF, DOCX</p>
        <input ref={ref} multiple type="file" accept=".txt,.md,.csv,.log,.pdf,.docx" />
        <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
          <button className="btn primary" onClick={onUpload}>Upload & Ingest</button>
          <button className="btn" onClick={async () => {
            const text = prompt("Quick ingest text:");
            if (text) { await api.ingest(text); alert("Ingested!"); }
          }}>Quick Ingest</button>
        </div>
        <pre className="muted" style={{ whiteSpace: "pre-wrap", marginTop: 16 }}>
          {last ? JSON.stringify(last, null, 2) : "No uploads yet."}
        </pre>
      </div>
    </div>
  );
}

/* ------- Documents ------- */
function DocsPage() {
  const [page, setPage] = useState(1), [size, setSize] = useState(10);
  const [data, setData] = useState({ total: 0, items: [] });
  useEffect(() => { (async () => setData(await api.getDocuments(page, size)))(); }, [page, size]);
  const totalPages = Math.max(1, Math.ceil(data.total / size));
  return (
    <div className="content">
      <Topbar right={<span className="badge">Documents</span>} />
      <div className="panel" style={{ margin: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3 style={{ margin: 0 }}>Documents</h3>
          <div className="muted">Total: {data.total}</div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", margin: "10px 0" }}>
          <span className="muted">Page</span>
          <button className="btn" onClick={() => setPage(p => Math.max(1, p - 1))}>Prev</button>
          <span className="badge">{page}/{totalPages}</span>
          <button className="btn" onClick={() => setPage(p => Math.min(totalPages, p + 1))}>Next</button>
          <span className="muted" style={{ marginLeft: 10 }}>Page size</span>
          <select value={size} onChange={e => { setPage(1); setSize(parseInt(e.target.value)); }}>
            {[5, 10, 20, 50].map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: "0 8px" }}>
          <thead><tr><th style={{ width: 220 }}>Doc ID</th><th>Source</th><th>Snippet</th></tr></thead>
          <tbody>
            {data.items.map(it => (
              <tr key={it.id}>
                <td><code>{it.id}</code></td>
                <td>{it.source || <span className="muted">unknown</span>}</td>
                <td className="muted">{it.snippet}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ------- DB Info ------- */
function DbInfoPage() {
  const [info, setInfo] = useState(null);
  useEffect(() => { (async () => setInfo(await getDbInfo()))(); }, []);
  return (
    <div className="content">
      <Topbar right={<span className="badge">DB Info</span>} />
      <div style={{ display: "grid", gap: 16, gridTemplateColumns: "1fr 1fr", margin: 16 }}>
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Vector DB</h3>
          {info ? (<div className="card">
            <div><b>Name:</b> {info.vector_db.name}</div>
            <div><b>Collection:</b> {info.vector_db.collection}</div>
            <div><b>Path:</b> <code>{info.vector_db.path}</code></div>
            <div><b>Docs:</b> {info.vector_db.doc_count ?? "?"}</div>
          </div>) : <div className="muted">Loading…</div>}
        </div>
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Sparse DB</h3>
          {info ? (<div className="card">
            <div><b>Name:</b> {info.sparse_db.name}</div>
            <div><b>Tables:</b> {info.sparse_db.tables.join(", ")}</div>
            <div><b>Path:</b> <code>{info.sparse_db.path}</code></div>
            <div><b>Docs:</b> {info.sparse_db.doc_count ?? "?"}</div>
          </div>) : <div className="muted">Loading…</div>}
        </div>
      </div>
    </div>
  );
}

/* ------- Rerank Debug ------- */
function RerankPage() {
  const [q, setQ] = useState(""), [data, setData] = useState(null);
  return (
    <div className="content">
      <Topbar right={<span className="badge">Rerank Debug</span>} />
      <div className="panel" style={{ margin: 16 }}>
        <form onSubmit={(e) => { e.preventDefault(); (async () => setData(await api.rerankDebug(q)))(); }} style={{ display: "flex", gap: 10 }}>
          <input value={q} onChange={e => setQ(e.target.value)} placeholder="Type a query to debug ranking…" style={{ flex: 1 }} />
          <button className="btn primary">Run</button>
        </form>
      </div>
      {data && (
        <div style={{ display: "grid", gap: 16, gridTemplateColumns: "1fr 1fr", margin: 16 }}>
          <div className="panel"><h3 style={{ marginTop: 0 }}>Dense</h3><ul className="muted">{data.dense.map((d, i) => <li key={i}><code>{d.id}</code> — {d.src || "?"}<br />{d.text}</li>)}</ul></div>
          <div className="panel"><h3 style={{ marginTop: 0 }}>Fused (RRF)</h3><ul className="muted">{data.fused.map((d, i) => <li key={i}><code>{d.id}</code> — {d.src || "?"} — RRF {d.rrf?.toFixed(4)}<br />{d.text}</li>)}</ul></div>
        </div>
      )}
    </div>
  );
}

/* ------- Memory ------- */
function MemoryPage() {
  const [session, setSession] = useState("demo"), [rows, setRows] = useState([]);
  async function load() { const res = await api.getMemory(session); setRows(res.rows || []); }
  useEffect(() => { load(); }, []);
  return (
    <div className="content">
      <Topbar right={<span className="badge">Memory</span>} />
      <div className="panel" style={{ margin: 16, display: "flex", gap: 10 }}>
        <input value={session} onChange={e => setSession(e.target.value)} placeholder="session id" style={{ flex: 1 }} />
        <button className="btn" onClick={load}>Load</button>
      </div>
      <div className="panel" style={{ margin: "0 16px 16px" }}>
        <table style={{ width: "100%" }}>
          <thead><tr><th style={{ width: 180 }}>Time</th><th>User</th><th>Assistant (first 400 chars)</th></tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}><td className="muted">{r[0]}</td><td>{r[1]}</td><td className="muted">{r[2]}</td></tr>
            ))}
          </tbody>
        </table>
        {!rows.length && <div className="muted">No rows</div>}
      </div>
    </div>
  );
}

/* ------- Settings ------- */
function SettingsPage() {
  const [models, setModels] = useState(null);
  useEffect(() => { (async () => setModels(await api.getModels()))(); }, []);
  return (
    <div className="content">
      <Topbar right={
        <div>
          <button className="btn" onClick={async () => {
            if (!confirm("This will clear vector store + sqlite docs + memories")) return;
            const res = await api.clearAll(); alert(res.cleared ? "Cleared" : "Failed");
          }}>Clear All</button>
          <span style={{ marginLeft: 10 }} className="badge">Settings</span>
        </div>
      } />
      <div style={{ display: "grid", gap: 16, gridTemplateColumns: "1fr 1fr", margin: 16 }}>
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Active Models</h3>
          {models ? (
            <div className="card">
              {Object.entries(models.models || {}).map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid var(--border)" }}>
                  <div className="muted">{k.toUpperCase()}</div><div>{v}</div>
                </div>
              ))}
            </div>
          ) : <div className="muted">Loading…</div>}
          <div className="muted" style={{ marginTop: 10 }}>Change via backend env: <code>LLM_RAG</code>, <code>LLM_CODE</code>, <code>LLM_SQL</code>.</div>
        </div>
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>About</h3>
          <div className="muted">Hybrid search (dense TF-IDF + BM25) with RRF. Uploads feed the store.</div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("chat");
  return (
    <div className="app">
      <Sidebar tab={tab} setTab={setTab} />
      {tab === "chat" && <ChatPage />}
      {tab === "upload" && <UploadPage />}
      {tab === "docs" && <DocsPage />}
      {tab === "db" && <DbInfoPage />}
      {tab === "rerank" && <RerankPage />}
      {tab === "memory" && <MemoryPage />}
      {tab === "settings" && <SettingsPage />}
    </div>
  );
}
