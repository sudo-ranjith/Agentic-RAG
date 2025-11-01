// frontend/src/pages/Chat.jsx
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import TypingDots from "../components/TypingDots.jsx";
import { ask } from "../api";

// Simple bubble wrapper
function Bubble({ side = "left", children }) {
  const isRight = side === "right";
  return (
    <div style={{
      display: "flex",
      justifyContent: isRight ? "flex-end" : "flex-start",
      padding: "8px 12px"
    }}>
      <div style={{
        maxWidth: 820,
        width: "fit-content",
        background: isRight ? "#2563eb" : "#ffffff",
        color: isRight ? "#ffffff" : "#111827",
        border: "1px solid #e5e7eb",
        borderRadius: isRight ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
        padding: "10px 12px",
        boxShadow: "0 1px 2px rgba(16,24,40,.05)",
        whiteSpace: "pre-wrap",
        overflowWrap: "anywhere"
      }}>
        {children}
      </div>
    </div>
  );
}

// Copy button shown on code blocks
function CopyBtn({ text }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(text || "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1000);
    } catch {}
  };
  return (
    <button
      onClick={onCopy}
      title="Copy"
      style={{
        position: "absolute", top: 8, right: 8,
        border: "1px solid #e5e7eb", background: "#fff",
        borderRadius: 8, padding: "4px 8px", fontSize: 12, cursor: "pointer"
      }}
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

// Markdown with code highlighting + copy
function Markdown({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ inline, className, children, ...props }) {
          const lang = /language-(\w+)/.exec(className || "")?.[1] || "";
          const codeText = String(children || "");
          if (inline) {
            return <code style={{
              background: "#f3f4f6",
              border: "1px solid #e5e7eb",
              borderRadius: 6,
              padding: "2px 6px"
            }} {...props}>{children}</code>;
          }
          return (
            <div style={{ position: "relative" }}>
              <CopyBtn text={codeText} />
              <SyntaxHighlighter style={oneLight} language={lang} PreTag="div" {...props}>
                {codeText}
              </SyntaxHighlighter>
            </div>
          );
        }
      }}
    >
      {content || ""}
    </ReactMarkdown>
  );
}

export default function Chat() {
  const [messages, setMessages] = useState([]); // {role:'user'|'assistant'|'typing', content:string}
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  const scrollToBottom = () => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  };

  useEffect(() => { scrollToBottom(); }, [messages]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;

    // push user message immediately (right bubble)
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);

    // show typing indicator from assistant (left)
    const typingId = Math.random().toString(36).slice(2);
    setMessages((prev) => [...prev, { role: "typing", id: typingId }]);

    try {
      // backend expects {query, session_id}
      const res = await ask({ query: text, session_id: "default" });
      const answer = res?.answer || "(no response)";
      // replace the typing bubble with actual assistant text
      setMessages((prev) => {
        const next = prev.filter((m) => m.id !== typingId); // remove typing
        next.push({ role: "assistant", content: answer, meta: { route: res?.route, citations: res?.citations || [] } });
        return next;
      });
    } catch (e) {
      setMessages((prev) => {
        const next = prev.filter((m) => m.id !== typingId);
        next.push({ role: "assistant", content: `**Error:** ${String(e)}` });
        return next;
      });
    } finally {
      setLoading(false);
    }
  };

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div style={{ height: "100%", display: "grid", gridTemplateRows: "1fr auto" }}>
      {/* Chat history */}
      <div style={{ overflowY: "auto", background: "#f8fafc", padding: "8px 0" }}>
        {messages.map((m, idx) => {
          if (m.role === "user") {
            return <Bubble key={idx} side="right"><Markdown content={m.content} /></Bubble>;
          }
          if (m.role === "typing") {
            return (
              <div key={idx} style={{ display: "flex", justifyContent: "flex-start", padding: "8px 12px" }}>
                <TypingDots />
              </div>
            );
          }
          // assistant
          return (
            <Bubble key={idx} side="left">
              <div style={{ display: "grid", gap: 8 }}>
                {m.meta?.route && (
                  <div style={{
                    fontSize: 12, color: "#6b7280", display: "inline-flex",
                    gap: 8, alignItems: "center"
                  }}>
                    <span style={{ padding: "2px 8px", border: "1px solid #e5e7eb", borderRadius: 999 }}>
                      Route: {m.meta.route}
                    </span>
                  </div>
                )}
                <Markdown content={m.content} />
                {!!(m.meta?.citations?.length) && (
                  <details>
                    <summary style={{ cursor: "pointer", color: "#374151" }}>Sources</summary>
                    <ul style={{ margin: "6px 0 0 16px", color: "#6b7280", fontSize: 12 }}>
                      {m.meta.citations.map((c, i) => (
                        <li key={i}>
                          <code>{c.doc_id}</code>
                          {c.meta?.source ? ` — ${c.meta.source}` : ""}
                          {typeof c.rrf === "number" ? ` — RRF: ${c.rrf.toFixed(4)}` : ""}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            </Bubble>
          );
        })}
        <div ref={scrollRef} />
      </div>

      {/* Composer */}
      <div style={{
        display: "flex", gap: 10, padding: 12, borderTop: "1px solid #e5e7eb", background: "#fff"
      }}>
        <textarea
          placeholder="Send a message… (supports Markdown and ```python``` code)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKey}
          style={{
            flex: 1,
            resize: "none",
            height: 48,
            borderRadius: 12,
            border: "1px solid #e5e7eb",
            padding: "10px 12px",
            outline: "none"
          }}
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          style={{
            padding: "0 18px",
            borderRadius: 10,
            border: "1px solid #e5e7eb",
            background: loading ? "#dbeafe" : "#111827",
            color: loading ? "#111827" : "#fff",
            fontWeight: 700,
            cursor: loading ? "not-allowed" : "pointer"
          }}
        >
          {loading ? "Thinking…" : "Send"}
        </button>
      </div>
    </div>
  );
}
