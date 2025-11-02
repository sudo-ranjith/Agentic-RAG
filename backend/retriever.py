import os
import re
import sqlite3
import uuid
import math
import time
from typing import Any, Dict, List, Optional, Tuple

# Optional Telemetry import (safe if missing)
try:
    from telemetry import Telemetry
except Exception:
    Telemetry = None  # type: ignore


SQLITE_PATH = os.getenv("SQLITE_PATH", "./rag_memory.db")

# ----------------- Utilities -----------------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _tokenize(text: str) -> List[str]:
    # simple word tokenizer; lowercased, drop empty
    return [t for t in re.split(r"[^\w]+", (text or "").lower()) if t]

def _estimate_tokens(text: str) -> int:
    # lightweight token estimate ~= 1.3 * words
    return int(len(_tokenize(text)) * 1.3)


# ----------------- Retriever -----------------
class Retriever:
    """
    Lightweight hybrid retriever:
      - SQLite `docs` table for persistence: (id TEXT PK, text TEXT, source TEXT)
      - In-memory indices for BM25 + TF-IDF
      - Dense = cosine(tf-idf), Sparse = BM25
      - RRF fusion
    """

    def __init__(self, sqlite_path: Optional[str] = None):
        self.sqlite_path = sqlite_path or SQLITE_PATH
        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self._ensure_tables()
        self.telemetry = Telemetry() if Telemetry else None

        # in-memory indices
        self._docs: Dict[str, Dict[str, Any]] = {}       # id -> {"text":..., "meta": {...}, "tokens":[...]}
        self._df: Dict[str, int] = {}                    # term -> doc frequency
        self._tf: Dict[str, Dict[str, int]] = {}         # doc_id -> {term: freq}
        self._len: Dict[str, int] = {}                   # doc_id -> token count
        self._N: int = 0                                 # number of docs
        self._avgdl: float = 0.0                         # avg doc length
        self._idf_cache: Dict[str, float] = {}           # cache idf
        self._tfidf_norms: Dict[str, float] = {}         # cosine norms

        self._load_index_from_db()

    # ---------- schema ----------
    def _ensure_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS docs(
              id TEXT PRIMARY KEY,
              text TEXT,
              source TEXT
            )
        """)
        # Memories + transactions may already exist (created elsewhere)
        self.conn.commit()

    def list_tables(self) -> List[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [r[0] for r in cur.fetchall()]

    # ---------- persistence ----------
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Insert docs into SQLite and update in-memory indices.
        """
        metadatas = metadatas or [{} for _ in texts]
        cur = self.conn.cursor()
        added = 0
        for text, meta in zip(texts, metadatas):
            text = (text or "").strip()
            if not text:
                continue
            doc_id = str(uuid.uuid4())
            src = (meta or {}).get("source")
            cur.execute("INSERT OR REPLACE INTO docs(id,text,source) VALUES(?,?,?)", (doc_id, text, src))
            added += 1

            # update in-memory
            toks = _tokenize(text)
            self._docs[doc_id] = {"text": text, "meta": {"source": src}, "tokens": toks}
            # tf
            tf_map: Dict[str, int] = {}
            for t in toks:
                tf_map[t] = tf_map.get(t, 0) + 1
            self._tf[doc_id] = tf_map
            self._len[doc_id] = len(toks)
            # df
            for t in set(toks):
                self._df[t] = self._df.get(t, 0) + 1

        self.conn.commit()

        # refresh counts
        self._N = len(self._docs)
        self._avgdl = (sum(self._len.values()) / self._N) if self._N else 0.0
        # invalidate caches
        self._idf_cache.clear()
        self._tfidf_norms.clear()
        # rebuild tf-idf norms
        for doc_id in self._docs.keys():
            self._tfidf_norms[doc_id] = self._tfidf_doc_norm(doc_id)

        return added

    def _load_index_from_db(self) -> None:
        """
        Load all docs into memory (OK for a few thousand docs).
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id, text, source FROM docs")
        rows = cur.fetchall()
        for doc_id, text, source in rows:
            text = text or ""
            toks = _tokenize(text)
            self._docs[doc_id] = {"text": text, "meta": {"source": source}, "tokens": toks}
            # tf
            tf_map: Dict[str, int] = {}
            for t in toks:
                tf_map[t] = tf_map.get(t, 0) + 1
            self._tf[doc_id] = tf_map
            self._len[doc_id] = len(toks)
            for t in set(toks):
                self._df[t] = self._df.get(t, 0) + 1

        self._N = len(self._docs)
        self._avgdl = (sum(self._len.values()) / self._N) if self._N else 0.0
        # precompute tf-idf norms
        for doc_id in self._docs.keys():
            self._tfidf_norms[doc_id] = self._tfidf_doc_norm(doc_id)

    # ---------- counts / info ----------
    def count_dense(self) -> int:
        # Here dense == total docs we index (we use TF-IDF as a lightweight dense stand-in)
        return self._N

    def sparse_count(self) -> int:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM docs")
            return int(cur.fetchone()[0])
        except Exception:
            return 0

    def get_collection_info(self) -> Dict[str, Any]:
        return {
            "collection": "docs",
            "path": os.path.abspath(self.sqlite_path),
            "doc_count": self._N
        }

    # ---------- listing ----------
    def list_docs(self, offset: int = 0, limit: int = 20) -> Tuple[List[Dict[str, Any]], int]:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM docs")
        total = int(cur.fetchone()[0])
        cur.execute("SELECT id, source, substr(text,1,800) FROM docs ORDER BY rowid DESC LIMIT ? OFFSET ?", (limit, offset))
        items = []
        for r in cur.fetchall():
            items.append({"id": r[0], "meta": {"source": r[1]}, "text": r[2]})
        return items, total

    # ---------- TF-IDF (dense proxy) ----------
    def _idf(self, term: str) -> float:
        if term in self._idf_cache:
            return self._idf_cache[term]
        df = self._df.get(term, 0)
        # classic idf with +1 smoothing
        val = math.log((self._N + 1) / (df + 1)) + 1.0 if self._N > 0 else 0.0
        self._idf_cache[term] = val
        return val

    def _tfidf_doc_norm(self, doc_id: str) -> float:
        vec = self._tf.get(doc_id, {})
        s = 0.0
        for t, tf in vec.items():
            w = (1 + math.log(tf)) * self._idf(t) if tf > 0 else 0.0
            s += w * w
        return math.sqrt(s) if s > 0 else 1.0

    def _tfidf_query(self, query: str) -> Dict[str, float]:
        qtf: Dict[str, int] = {}
        for t in _tokenize(query):
            qtf[t] = qtf.get(t, 0) + 1
        wq: Dict[str, float] = {}
        for t, tf in qtf.items():
            wq[t] = (1 + math.log(tf)) * self._idf(t) if tf > 0 else 0.0
        return wq

    def search_dense(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """
        Cosine similarity over TF-IDF (dense proxy). Returns sorted docs with "_score".
        """
        if not self._N:
            return []
        wq = self._tfidf_query(query)
        norm_q = math.sqrt(sum(v * v for v in wq.values())) or 1.0
        scores: List[Tuple[str, float]] = []
        for doc_id, tf_map in self._tf.items():
            num = 0.0
            for t, tf in tf_map.items():
                if t in wq:
                    num += ((1 + math.log(tf)) * self._idf(t)) * wq[t]
            den = (self._tfidf_norms.get(doc_id) or 1.0) * norm_q
            sim = num / den if den > 0 else 0.0
            scores.append((doc_id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        out = []
        for doc_id, sc in scores[:k]:
            d = self._docs[doc_id]
            out.append({"id": doc_id, "text": d["text"], "meta": d["meta"], "_score": sc})
        return out

    # ---------- BM25 (sparse) ----------
    def _bm25_score(self, query_tokens: List[str], doc_id: str, k1: float = 1.5, b: float = 0.75) -> float:
        score = 0.0
        dl = self._len.get(doc_id, 0) or 1
        avdl = self._avgdl or 1.0
        tf_map = self._tf.get(doc_id, {})
        for t in query_tokens:
            tf = tf_map.get(t, 0)
            if tf == 0:
                continue
            idf = self._idf(t)
            denom = tf + k1 * (1 - b + b * (dl / avdl))
            score += idf * (tf * (k1 + 1)) / (denom if denom > 0 else 1.0)
        return score

    def search_bm25(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        if not self._N:
            return []
        q_tokens = _tokenize(query)
        scores: List[Tuple[str, float]] = []
        for doc_id in self._docs.keys():
            sc = self._bm25_score(q_tokens, doc_id)
            if sc > 0:
                scores.append((doc_id, sc))
        scores.sort(key=lambda x: x[1], reverse=True)
        out = []
        for doc_id, sc in scores[:k]:
            d = self._docs[doc_id]
            out.append({"id": doc_id, "text": d["text"], "meta": d["meta"], "_score": sc})
        return out

    # ---------- RRF ----------
    def _rrf(self, dense: List[Dict[str, Any]], sparse: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion: score = Σ 1 / (k + rank)
        """
        pos: Dict[str, float] = {}
        # dense ranks
        for i, d in enumerate(dense):
            pos[d["id"]] = pos.get(d["id"], 0.0) + 1.0 / (k + (i + 1))
        # sparse ranks
        for i, d in enumerate(sparse):
            pos[d["id"]] = pos.get(d["id"], 0.0) + 1.0 / (k + (i + 1))

        # build full doc rows with fused score
        out = []
        for doc_id, rrf in pos.items():
            d = self._docs.get(doc_id)
            if not d:
                continue
            out.append({"id": doc_id, "text": d["text"], "meta": d["meta"], "_rrf": rrf})
        out.sort(key=lambda x: x["_rrf"], reverse=True)
        return out

    def fuse(self, dense: List[Dict[str, Any]], sparse: List[Dict[str, Any]], *, k_rrf: int = 60, top_k: int = 6) -> List[Dict[str, Any]]:
        """Public helper that performs reciprocal rank fusion and truncates the result.

        This is primarily used by the agentic orchestrator so it can expose
        intermediate retrieval timings (dense, sparse, fusion) separately while
        still relying on the same fusion logic that powers :meth:`hybrid_search`.
        """

        fused = self._rrf(dense, sparse, k=k_rrf)
        return fused[:top_k]

    # ---------- Hybrid ----------
    def hybrid_search(
        self,
        query: str,
        *,
        k_dense: int = 6,
        k_sparse: int = 6,
        k_rrf: int = 60,
        top_k: int = 6,
        session_id: Optional[str] = None,  # optional; not used by your current main.py, safe default
    ) -> List[Dict[str, Any]]:
        """
        1) optional 'embed' accounting for the query
        2) dense (tf-idf) search
        3) sparse (bm25) search
        4) RRF fuse and return top_k
        """
        t0 = _now_ms()
        # ---- telemetry: embed (query side) ----
        if self.telemetry:
            try:
                self.telemetry.record(
                    id=str(uuid.uuid4()),
                    session_id=session_id or "default",
                    route="RAG",
                    model_role="embed",
                    model_name="tfidf-lite",
                    prompt_tokens=_estimate_tokens(query),
                    completion_tokens=0,
                    latency_ms=0,
                    query=query
                )
            except Exception:
                pass

        # ---- dense ----
        dense = self.search_dense(query, k=k_dense)
        if self.telemetry:
            try:
                self.telemetry.record(
                    id=str(uuid.uuid4()),
                    session_id=session_id or "default",
                    route="RAG",
                    model_role="dense",
                    model_name="tfidf-lite",
                    prompt_tokens=_estimate_tokens(query),
                    completion_tokens=0,
                    latency_ms=max(1, _now_ms() - t0),
                    query=query
                )
            except Exception:
                pass
        t1 = _now_ms()

        # ---- sparse ----
        sparse = self.search_bm25(query, k=k_sparse)
        if self.telemetry:
            try:
                self.telemetry.record(
                    id=str(uuid.uuid4()),
                    session_id=session_id or "default",
                    route="RAG",
                    model_role="sparse",
                    model_name="bm25-lite",
                    prompt_tokens=_estimate_tokens(query),
                    completion_tokens=0,
                    latency_ms=max(1, _now_ms() - t1),
                    query=query
                )
            except Exception:
                pass

        fused = self._rrf(dense, sparse, k=k_rrf)
        return fused[:top_k]
