// frontend/src/pages/Dashboard.jsx
import { useEffect, useMemo, useState } from "react";
import { getMetricsSummary, getMetricsByModel, getMetricsRecent, getDbInfo } from "../api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid,
} from "recharts";

const Card = ({ title, value, subtitle }) => (
  <div style={{
    background: "#fff",
    border: "1px solid #e5e7eb",
    borderRadius: 12,
    padding: 16,
    display: "flex",
    flexDirection: "column",
    gap: 6,
    boxShadow: "0 1px 2px rgba(16,24,40,0.05)"
  }}>
    <div style={{ fontSize: 12, color: "#6b7280", fontWeight: 600 }}>{title}</div>
    <div style={{ fontSize: 28, fontWeight: 700, color: "#111827" }}>{value}</div>
    {subtitle ? <div style={{ fontSize: 12, color: "#6b7280" }}>{subtitle}</div> : null}
  </div>
);

const Section = ({ title, children, right }) => (
  <div style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12, padding: 16 }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
      <div style={{ fontWeight: 700, color: "#111827" }}>{title}</div>
      {right}
    </div>
    {children}
  </div>
);

function useTimeRange() {
  const [range, setRange] = useState("7d"); // today | 7d | 30d | all
  const compute = () => {
    const now = new Date();
    if (range === "all") return { since: null, until: null };
    if (range === "today") {
      const start = new Date(now); start.setHours(0, 0, 0, 0);
      return { since: start.toISOString(), until: now.toISOString() };
    }
    const days = range === "30d" ? 30 : 7;
    const start = new Date(now.getTime() - days * 24 * 3600 * 1000);
    return { since: start.toISOString(), until: now.toISOString() };
  };
  return { range, setRange, ...compute() };
}

export default function Dashboard() {
  const { range, setRange, since, until } = useTimeRange();
  const [summary, setSummary] = useState({ input_tokens: 0, output_tokens: 0, guard_tokens_saved: 0, guard_topics_count: 0 });
  const [byModel, setByModel] = useState([]);
  const [recent, setRecent] = useState([]);
  const [dbinfo, setDbinfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancel = false;
    (async () => {
      setLoading(false);
      try {
        const [s, m, r, d] = await Promise.all([
          getMetricsSummary(since, until),
          getMetricsByModel(since, until),
          getMetricsRecent(10),
          getDbInfo(),
        ]);
        if (!cancel) {
          setSummary(s || {});
          setByModel((m && m.rows) || []);
          setRecent((r && r.rows) || []);
          setDbinfo(d || null);
        }
      } catch (e) {
        console.error(e);
      } finally {
        !cancel && setLoading(false);
      }
    })();
    return () => { cancel = true; };
  }, [since, until]);

  const chartData = useMemo(() => {
    // Normalize roles to pretty labels
    const pretty = {
      embed: "Embedding",
      dense: "Dense",
      sparse: "Sparse",
      rag: "RAG",
      code: "Code",
      sql: "SQL",
    };
    return byModel.map((r) => ({
      role: pretty[r.model_role] || r.model_role,
      Input: r.prompt_tokens || 0,
      Output: r.completion_tokens || 0,
      Total: r.total_tokens || 0,
      Calls: r.calls || 0,
    }));
  }, [byModel]);

  return (
    <div style={{ padding: 18, display: "grid", gridTemplateColumns: "1fr", gap: 16 }}>
      {/* Filters */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: 24, fontWeight: 800, color: "#111827" }}>Agentic RAG — Dashboard</div>
          <div style={{ color: "#6b7280" }}>Overview of tokens, guardrails, and activity</div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {["today", "7d", "30d", "all"].map((opt) => (
            <button
              key={opt}
              onClick={() => setRange(opt)}
              style={{
                padding: "8px 12px",
                borderRadius: 8,
                border: "1px solid #e5e7eb",
                background: range === opt ? "#111827" : "#fff",
                color: range === opt ? "#fff" : "#111827",
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              {opt.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* KPI cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <Card title="Input Tokens" value={summary.input_tokens?.toLocaleString() ?? 0} />
        <Card title="Output Tokens" value={summary.output_tokens?.toLocaleString() ?? 0} />
        <Card title="Tokens Prevented (Guard)" value={summary.guard_tokens_saved?.toLocaleString() ?? 0} />
        <Card title="Guard Topics Found" value={summary.guard_topics_count ?? 0} />
      </div>

      {/* Tokens by model/layer */}
      <Section
        title="Tokens by Model / Layer"
        right={<div style={{ fontSize: 12, color: "#6b7280" }}>{chartData.reduce((s, r) => s + (r.Total || 0), 0).toLocaleString()} total</div>}
      >
        <div style={{ height: 280 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="role" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="Input" stackId="a" />
              <Bar dataKey="Output" stackId="a" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      {/* DB info */}
      <Section title="Knowledge Stores">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div style={{ display: "grid", gap: 8 }}>
            <div style={{ fontWeight: 700 }}>Vector DB</div>
            <div style={{ fontSize: 14, color: "#374151" }}>
              <div><b>Name:</b> {dbinfo?.vector_db?.name || "—"}</div>
              <div><b>Collection:</b> {dbinfo?.vector_db?.collection || "—"}</div>
              <div><b>Path:</b> {dbinfo?.vector_db?.path || "—"}</div>
              <div><b>Docs:</b> {dbinfo?.vector_db?.doc_count ?? "—"}</div>
            </div>
          </div>
          <div style={{ display: "grid", gap: 8 }}>
            <div style={{ fontWeight: 700 }}>Sparse DB</div>
            <div style={{ fontSize: 14, color: "#374151" }}>
              <div><b>Name:</b> {dbinfo?.sparse_db?.name || "—"}</div>
              <div><b>Path:</b> {dbinfo?.sparse_db?.path || "—"}</div>
              <div><b>Tables:</b> {(dbinfo?.sparse_db?.tables || []).join(", ") || "—"}</div>
              <div><b>Docs:</b> {dbinfo?.sparse_db?.doc_count ?? "—"}</div>
            </div>
          </div>
        </div>
      </Section>

      {/* Recent activity */}
      <Section title="Recent Activity">
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ textAlign: "left", color: "#6b7280", fontSize: 12 }}>
                <th style={{ padding: 8, borderBottom: "1px solid #e5e7eb" }}>Time</th>
                <th style={{ padding: 8, borderBottom: "1px solid #e5e7eb" }}>Session</th>
                <th style={{ padding: 8, borderBottom: "1px solid #e5e7eb" }}>Route</th>
                <th style={{ padding: 8, borderBottom: "1px solid #e5e7eb" }}>Role</th>
                <th style={{ padding: 8, borderBottom: "1px solid #e5e7eb" }}>Model</th>
                <th style={{ padding: 8, borderBottom: "1px solid #e5e7eb" }}>In</th>
                <th style={{ padding: 8, borderBottom: "1px solid #e5e7eb" }}>Out</th>
                <th style={{ padding: 8, borderBottom: "1px solid #e5e7eb" }}>Total</th>
                <th style={{ padding: 8, borderBottom: "1px solid #e5e7eb" }}>Latency</th>
              </tr>
            </thead>
            <tbody>
              {recent.map((r, i) => (
                <tr key={i} style={{ fontSize: 13 }}>
                  <td style={{ padding: 8, borderBottom: "1px solid #f3f4f6", whiteSpace: "nowrap" }}>{r.ts?.replace("T", " ") || "—"}</td>
                  <td style={{ padding: 8, borderBottom: "1px solid #f3f4f6" }}>{r.session_id || "—"}</td>
                  <td style={{ padding: 8, borderBottom: "1px solid #f3f4f6" }}>{r.route || "—"}</td>
                  <td style={{ padding: 8, borderBottom: "1px solid #f3f4f6" }}>{r.model_role || "—"}</td>
                  <td style={{ padding: 8, borderBottom: "1px solid #f3f4f6" }}>{r.model_name || "—"}</td>
                  <td style={{ padding: 8, borderBottom: "1px solid #f3f4f6" }}>{r.prompt_tokens ?? 0}</td>
                  <td style={{ padding: 8, borderBottom: "1px solid #f3f4f6" }}>{r.completion_tokens ?? 0}</td>
                  <td style={{ padding: 8, borderBottom: "1px solid #f3f4f6" }}>{r.total_tokens ?? 0}</td>
                  <td style={{ padding: 8, borderBottom: "1px solid #f3f4f6" }}>{r.latency_ms ? `${r.latency_ms} ms` : "—"}</td>
                </tr>
              ))}
              {!recent.length && (
                <tr><td colSpan="9" style={{ padding: 16, color: "#6b7280" }}>No activity yet.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Section>

      {loading && <div style={{ textAlign: "center", color: "#6b7280" }}>Loading…</div>}
    </div>
  );
}
