import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Reuse the same DB file your app already uses (rag_memory.db by default)
SQLITE_PATH = os.getenv("SQLITE_PATH", "./rag_memory.db")


class Telemetry:
    """
    Lightweight telemetry layer for Agentic RAG.

    Tables:
      - transactions: one row per model invocation (RAG/CODE/SQL)
        id TEXT PK
        ts DATETIME DEFAULT CURRENT_TIMESTAMP
        session_id TEXT
        route TEXT                 -- RAG / CODE / SQL
        model_role TEXT            -- rag / code / sql / embed / dense / sparse (if you log those)
        model_name TEXT            -- e.g., mistral:7b-instruct
        prompt_tokens INTEGER
        completion_tokens INTEGER
        total_tokens INTEGER
        guard_action TEXT          -- allow / block / redact (optional)
        guard_topics TEXT          -- JSON array of topics
        guard_tokens_saved INTEGER -- tokens 'prevented' by guardrails
        embed_tokens INTEGER       -- estimated tokens for embeddings (optional)
        dense_tokens INTEGER       -- same as embed_tokens or extra
        sparse_terms INTEGER       -- number of unique sparse terms (BM25)
        latency_ms INTEGER
        query TEXT                 -- original user query (optional)
        citations TEXT             -- JSON array of citations (optional)
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or SQLITE_PATH
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._ensure()

    def _ensure(self) -> None:
        cur = self.conn.cursor()
        # transactions table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
              id TEXT PRIMARY KEY,
              ts DATETIME DEFAULT CURRENT_TIMESTAMP,
              session_id TEXT,
              route TEXT,
              model_role TEXT,
              model_name TEXT,
              prompt_tokens INTEGER,
              completion_tokens INTEGER,
              total_tokens INTEGER,
              guard_action TEXT,
              guard_topics TEXT,
              guard_tokens_saved INTEGER,
              embed_tokens INTEGER,
              dense_tokens INTEGER,
              sparse_terms INTEGER,
              latency_ms INTEGER,
              query TEXT,
              citations TEXT
            )
            """
        )
        # small index for time range scans
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_ts ON transactions(ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_role ON transactions(model_role)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_route ON transactions(route)")
        self.conn.commit()

    # ---------- writers ----------

    def record(
        self,
        *,
        id: str,
        session_id: str,
        route: str,
        model_role: str,
        model_name: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
        guard_action: Optional[str] = None,
        guard_topics: Optional[List[str]] = None,
        guard_tokens_saved: int = 0,
        embed_tokens: Optional[int] = None,
        dense_tokens: Optional[int] = None,
        sparse_terms: Optional[int] = None,
        latency_ms: Optional[int] = None,
        query: Optional[str] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        ts: Optional[str] = None,
    ) -> None:
        """
        Insert one row. id should be a UUID you generate per invocation.
        """
        total = total_tokens if total_tokens is not None else (prompt_tokens or 0) + (completion_tokens or 0)
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO transactions (
              id, ts, session_id, route, model_role, model_name,
              prompt_tokens, completion_tokens, total_tokens,
              guard_action, guard_topics, guard_tokens_saved,
              embed_tokens, dense_tokens, sparse_terms,
              latency_ms, query, citations
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                id,
                ts or datetime.utcnow().isoformat(timespec="seconds"),
                session_id,
                route,
                model_role,
                model_name,
                int(prompt_tokens or 0),
                int(completion_tokens or 0),
                int(total or 0),
                guard_action,
                json.dumps(guard_topics or []),
                int(guard_tokens_saved or 0),
                None if embed_tokens is None else int(embed_tokens),
                None if dense_tokens is None else int(dense_tokens),
                None if sparse_terms is None else int(sparse_terms),
                None if latency_ms is None else int(latency_ms),
                query,
                json.dumps(citations or []),
            ),
        )
        self.conn.commit()

    # ---------- readers for dashboard ----------

    def summary(
        self,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns totals needed for KPI cards:
          - input_tokens (sum of prompt_tokens)
          - output_tokens (sum of completion_tokens)
          - guard_tokens_saved
          - guard_topics_count (distinct topics count)
        """
        cur = self.conn.cursor()

        where, params = self._time_where(since, until)

        # input / output tokens
        cur.execute(
            f"""SELECT
                  COALESCE(SUM(prompt_tokens),0),
                  COALESCE(SUM(completion_tokens),0),
                  COALESCE(SUM(guard_tokens_saved),0)
                FROM transactions {where}""",
            params,
        )
        inp, outp, saved = cur.fetchone()

        # topics count
        cur.execute(
            f"""SELECT guard_topics FROM transactions {where} AND guard_topics IS NOT NULL""",
            params,
        )
        topic_set = set()
        for (raw,) in cur.fetchall():
            try:
                for t in json.loads(raw or "[]"):
                    topic_set.add(t)
            except Exception:
                continue

        return {
            "input_tokens": int(inp or 0),
            "output_tokens": int(outp or 0),
            "guard_tokens_saved": int(saved or 0),
            "guard_topics_count": len(topic_set),
        }

    def tokens_by_model_role(
        self, *, since: Optional[str] = None, until: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns aggregated tokens by model_role (embedding/dense/sparse/rag/code/sql)
        """
        cur = self.conn.cursor()
        where, params = self._time_where(since, until)

        cur.execute(
            f"""
            SELECT model_role,
                   COALESCE(SUM(prompt_tokens),0) as prompt_sum,
                   COALESCE(SUM(completion_tokens),0) as completion_sum,
                   COALESCE(SUM(total_tokens),0) as total_sum,
                   COUNT(*) as calls
            FROM transactions
            {where}
            GROUP BY model_role
            ORDER BY total_sum DESC
            """,
            params,
        )
        rows = cur.fetchall()
        return [
            {
                "model_role": r[0],
                "prompt_tokens": int(r[1]),
                "completion_tokens": int(r[2]),
                "total_tokens": int(r[3]),
                "calls": int(r[4]),
            }
            for r in rows
        ]

    def recent(self, *, limit: int = 10) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT ts, session_id, route, model_role, model_name,
                   prompt_tokens, completion_tokens, total_tokens, latency_ms
            FROM transactions
            ORDER BY ts DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
        cols = ["ts","session_id","route","model_role","model_name","prompt_tokens","completion_tokens","total_tokens","latency_ms"]
        return [dict(zip(cols, r)) for r in rows]

    def _time_where(self, since: Optional[str], until: Optional[str]) -> Tuple[str, Tuple[Any, ...]]:
        """
        Build WHERE clause for time range:
          - since/until are ISO strings; if None, defaults to last 30 days.
        """
        params: List[Any] = []
        if not since and not until:
            since = (datetime.utcnow() - timedelta(days=30)).isoformat(timespec="seconds")

        clauses = []
        if since:
            clauses.append("ts >= ?")
            params.append(since)
        if until:
            clauses.append("ts <= ?")
            params.append(until)

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return where, tuple(params)
