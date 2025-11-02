import os, sqlite3, json, re, psutil, time, sys, uuid, logging
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from ollama_client import generate, generate_with_stats
from telemetry import Telemetry  # <-- new import for transaction logging

# ================= Logging =================
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

file_handler = logging.FileHandler(
    f'{log_dir}/agents_{datetime.now().strftime("%Y%m%d")}.log',
    encoding="utf-8"  # ✅ ensure UTF-8 in file
)
stream_handler = logging.StreamHandler()  # console (keep ASCII-safe content)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

# ================= Env Config =================
load_dotenv()
SQLITE_PATH = os.getenv("SQLITE_PATH", "./rag_memory.db")

MODEL = {
    "rag":      os.getenv("LLM_RAG",      "mistral:7b-instruct"),
    "code":     os.getenv("LLM_CODE",     "Qwen2.5-Coder:latest"),
    "sql":      os.getenv("LLM_SQL",      "sqlcoder:latest"),
    "fallback": os.getenv("LLM_FALLBACK", "llama3:latest"),
}

process = psutil.Process()
logger.info("=== System Information ===")
logger.info(f"Python Version: {sys.version}")
logger.info(f"CPU Count: {psutil.cpu_count()}")
logger.info(f"Available Memory: {psutil.virtual_memory().available / (1024*1024*1024):.2f} GB")
logger.info(f"PID: {process.pid}")
logger.info(f"Models: {MODEL}")
logger.info("========================")

telemetry = Telemetry()  # global telemetry DB handler

# ================= Helper Decorator =================
def log_execution_time(func):
    def wrapper(*args, **kwargs):
        req_id = str(uuid.uuid4())[:8]
        start = time.time()
        start_mem = process.memory_info().rss / (1024 * 1024)
        logger.info(f"[{req_id}] Start {func.__name__}")
        try:
            result = func(*args, **kwargs)
            dur = time.time() - start
            mem = process.memory_info().rss / (1024 * 1024) - start_mem
            logger.info(f"[{req_id}] Done {func.__name__} in {dur:.2f}s  - {mem:.2f}MB")
            return result
        except Exception as e:
            logger.error(f"[{req_id}] Fail {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper


def _ensure_fenced_code(text: str, default_lang: str = "python") -> str:
    if "```" in (text or ""):
        return text
    clean = (text or "").strip() or "# (no content returned)"
    return f"```{default_lang}\n{clean}\n```"


# ================= Memory =================
class Memory:
    def __init__(self, path: str = SQLITE_PATH):
        logger.info(f"Memory DB path: {path}")
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._ensure()

    def _ensure(self):
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS memories(
            session_id TEXT, user TEXT, assistant TEXT, citations TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        self.conn.commit()

    @log_execution_time
    def save(self, session_id: str, user: str, assistant: str, citations: List[Dict[str, Any]]):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO memories(session_id,user,assistant,citations) VALUES(?,?,?,?)",
                    (session_id, user, assistant, json.dumps(citations)))
        self.conn.commit()

    @log_execution_time
    def fetch(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return the most recent conversation turns for the given session."""

        cur = self.conn.cursor()
        cur.execute(
            "SELECT user, assistant, citations, ts FROM memories WHERE session_id=? ORDER BY ts DESC LIMIT ?",
            (session_id, int(limit)),
        )
        rows = cur.fetchall()
        history: List[Dict[str, Any]] = []
        for user_text, assistant_text, raw_citations, ts in reversed(rows):
            try:
                citations = json.loads(raw_citations or "[]")
            except Exception:
                citations = []
            history.append(
                {
                    "user": user_text,
                    "assistant": assistant_text,
                    "citations": citations,
                    "ts": ts,
                }
            )
        return history


# ================= Routing =================
class Route(Enum):
    RAG = "RAG"
    CODE = "CODE"
    SQL  = "SQL"

class Policy:
    @staticmethod
    @log_execution_time
    def decide(query: str) -> Route:
        q = (query or "").lower()
        if any(k in q for k in ["select ", "sql ", "schema", "table", "join ", "group by", "where "]):
            return Route.SQL
        if any(k in q for k in ["code", "function", "class", "bug", "refactor", "python", "typescript", " js ", "javascript"]):
            return Route.CODE
        return Route.RAG


# ================= Answer Functions =================
@log_execution_time
def answer_rag(query: str, context: str, session_id: str | None = None) -> str:
    logger.info(f"Processing RAG query: {query} (session={session_id})")
    logger.debug(f"Context length: {len(context)} characters")
    try:
        prompt = f"""You are a helpful assistant. Use CONTEXT to answer QUESTION concisely.
If the answer is not in the context, You can feel free to use the available models, but just mention  that you asked message is not in the provided context or injested content, it is generated by the model, you can mention the model name too.

CONTEXT:
{context}

QUESTION: {query}

Answer in bullet points. Add short citations like [doc-id] when you directly use a snippet.
"""
        response = generate("llama3", prompt)
        logger.info("RAG response generated successfully")
        logger.debug(f"Response: {response[:100]}...")
        return response
    except Exception as e:
        logger.error(f"RAG answer generation failed: {str(e)}", exc_info=True)
        raise

@log_execution_time
def answer_code(query: str, context: str, session_id: str | None = None) -> str:
    logger.info(f"Processing CODE query: {query} (session={session_id})")
    logger.debug(f"Context length: {len(context)} characters")
    try:
        prompt = f"""You are a senior engineer. Based on REQUEST and any useful CONTEXT,
produce a brief plan and working code. If assumptions are needed, list them.

CONTEXT:
{context}

REQUEST:
{query}

Return:
- Plan (3-5 bullets)
- Code block(s)
- Notes on how to run
"""
        response = generate("llama3", prompt)
        logger.info("Code response generated successfully")
        logger.debug(f"Response: {response[:100]}...")
        return response
    except Exception as e:
        logger.error(f"Code answer generation failed: {str(e)}", exc_info=True)
        raise

@log_execution_time
def answer_sql(query: str, session_id: str | None = None) -> str:
    logger.info(f"Processing SQL query: {query} (session={session_id})")
    try:
        schema_hint = """
-- Tables available:
-- memories(session_id TEXT, user TEXT, assistant TEXT, citations TEXT, ts DATETIME)
-- docs(id TEXT PRIMARY KEY, text TEXT, source TEXT)
"""
        llm = generate("sqlcoder", f"""You are SQLCoder. Produce a single SQLite query in a fenced ```sql``` block
that satisfies the USER REQUEST. Use only existing columns. Then add a 1-2 line explanation.

DB SCHEMA:
{schema_hint}

USER REQUEST:
{query}
""")
        logger.info("SQL query generated by LLM")
        logger.debug(f"LLM response: {llm[:100]}...")

        sql = _extract_sql(llm)
        if not sql:
            logger.warning("No SQL query found in LLM response")
            return llm

        try:
            conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            payload = {"columns": cols, "rows": rows}
            logger.info(f"SQL execution successful, returned {len(rows)} rows")
            return llm + "\n\n-- Execution Result --\n" + json.dumps(payload, indent=2, default=str)
        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}", exc_info=True)
            return llm + f"\n\n[Execution Error] {e}"
    except Exception as e:
        logger.error(f"SQL answer generation failed: {str(e)}", exc_info=True)
        raise

def _extract_sql(text: str) -> Optional[str]:
    m = re.search(r"sql\s*([\s\S]*?)", text or "", re.IGNORECASE)
    return m.group(1).strip().rstrip(";") if m else None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from raw LLM output."""

    if not text:
        return None

    candidates = []
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates.extend(fenced)
    candidates.append(text)

    for raw in candidates:
        raw = (raw or "").strip()
        if not raw:
            continue
        try:
            return json.loads(raw)
        except Exception:
            continue
    return None


class StageTracker:
    def __init__(self) -> None:
        self._stages: List[Dict[str, Any]] = []

    def start(self, name: str, description: str) -> Dict[str, Any]:
        entry = {
            "name": name,
            "description": description,
            "status": "running",
            "started_at": datetime.utcnow().isoformat(timespec="seconds"),
            "detail": {},
            "_t0": time.time(),
        }
        self._stages.append(entry)
        logger.info(f"[Stage] {name} — started: {description}")
        return entry

    def end(self, entry: Dict[str, Any], *, status: str = "completed", detail: Optional[Dict[str, Any]] = None) -> None:
        entry["status"] = status
        entry["ended_at"] = datetime.utcnow().isoformat(timespec="seconds")
        entry["duration_ms"] = max(1, int((time.time() - entry.get("_t0", time.time())) * 1000))
        if detail:
            entry.setdefault("detail", {}).update(detail)
        logger.info(
            "[Stage] %s — %s in %sms detail=%s",
            entry.get("name"),
            status,
            entry.get("duration_ms"),
            entry.get("detail"),
        )

    def fail(self, entry: Dict[str, Any], error: Exception) -> None:
        self.end(entry, status="failed", detail={"error": str(error)})

    def skip(self, name: str, description: str, reason: str) -> None:
        entry = self.start(name, description)
        self.end(entry, status="skipped", detail={"reason": reason})

    def export(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for stage in self._stages:
            clean = {k: v for k, v in stage.items() if k != "_t0"}
            out.append(clean)
        return out


class AgentPlanner:
    """Lightweight planner that lets the model outline steps and pick a route."""

    @staticmethod
    def plan(query: str, history_text: str, fallback_route: Route) -> Dict[str, Any]:
        prompt = f"""
You are the orchestration module for an agentic retrieval-augmented assistant.
Review the short conversation history and the new user request. Decide which
capability to use (RAG, CODE, SQL) and outline 2-4 high level steps the agent
should follow. Respond with JSON matching this schema:

{{
  "route": "RAG|CODE|SQL",
  "plan": ["step 1", "step 2", ...],
  "should_retrieve": true/false,
  "notes": "optional clarifying note"
}}

Conversation history (latest last):
{history_text or '(no prior turns)'}

User request: {query}

Return ONLY the JSON object with double quotes.
"""

        text, stats = generate_with_stats(
            MODEL.get("fallback", "llama3:latest"),
            prompt,
            max_tokens=512,
            temperature=0.2,
        )
        logger.info("Planner raw output: %s", text)

        data = _extract_json_object(text) or {}
        route_value = str(data.get("route") or "").strip().upper()
        if route_value not in Route._value2member_map_:
            route_value = fallback_route.value
        data["route"] = route_value

        plan_steps = [str(step).strip() for step in data.get("plan", []) if str(step).strip()]
        data["plan"] = plan_steps
        data.setdefault("should_retrieve", True)
        data.setdefault("notes", "")
        data["_stats"] = stats
        return data


class AgenticRAG:
    def __init__(self, retriever: "Retriever", memory: Memory, telemetry: Telemetry, *, memory_turns: int = 6):
        self.retriever = retriever
        self.memory = memory
        self.telemetry = telemetry
        self.memory_turns = memory_turns

    def _history_to_text(self, history: List[Dict[str, Any]]) -> str:
        parts = []
        for turn in history:
            user = (turn.get("user") or "").strip()
            assistant = (turn.get("assistant") or "").strip()
            if user:
                parts.append(f"User: {user}")
            if assistant:
                parts.append(f"Assistant: {assistant}")
        return "\n".join(parts[-12:])

    def _context_from_docs(self, docs: List[Dict[str, Any]]) -> str:
        blocks = []
        for idx, doc in enumerate(docs, start=1):
            snippet = (doc.get("text") or "").strip()
            blocks.append(f"[doc-{idx}] {snippet}")
        return "\n\n".join(blocks)

    def _record_stats(
        self,
        stats: Dict[str, Any],
        *,
        session_id: str,
        route: str,
        model_role: str,
        query: str,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not stats or not self.telemetry:
            return
        try:
            self.telemetry.record(
                id=str(uuid.uuid4()),
                session_id=session_id,
                route=route,
                model_role=model_role,
                model_name=stats.get("model", "unknown"),
                prompt_tokens=int(stats.get("prompt_tokens") or 0),
                completion_tokens=int(stats.get("completion_tokens") or 0),
                total_tokens=int(stats.get("total_tokens") or 0),
                latency_ms=int(stats.get("latency_ms") or 0),
                query=query,
                citations=citations,
            )
        except Exception as exc:
            logger.warning("Telemetry record failed: %s", exc)

    def run(self, query: str, *, session_id: str = "default") -> Dict[str, Any]:
        tracker = StageTracker()
        history: List[Dict[str, Any]] = []
        plan_steps: List[str] = []
        planner_notes = ""
        citations: List[Dict[str, Any]] = []
        context_docs: List[Dict[str, Any]] = []

        logger.info("Agentic run start — session=%s query=%s", session_id, query)

        # ---- memory recall ----
        memory_stage = tracker.start("Memory Recall", "Load recent conversation turns")
        try:
            history = self.memory.fetch(session_id, limit=self.memory_turns)
            tracker.end(memory_stage, detail={"turns": len(history)})
        except Exception as exc:
            tracker.fail(memory_stage, exc)
            history = []

        history_text = self._history_to_text(history)
        policy_route = Policy.decide(query)

        # ---- planner ----
        plan_stage = tracker.start("Planner", "Select best tool and outline approach")
        try:
            plan_payload = AgentPlanner.plan(query, history_text, policy_route)
            planner_notes = plan_payload.get("notes") or ""
            plan_steps = plan_payload.get("plan", [])
            route_value = plan_payload.get("route", policy_route.value)
            should_retrieve = bool(plan_payload.get("should_retrieve", True))
            planner_stats = plan_payload.pop("_stats", {})
            self._record_stats(
                planner_stats,
                session_id=session_id,
                route="AGENT",
                model_role="planner",
                query=query,
            )
            if route_value in Route._value2member_map_:
                route = Route(route_value)
            else:
                route = policy_route
            tracker.end(plan_stage, detail={
                "route": route.value,
                "steps": len(plan_steps),
                "should_retrieve": should_retrieve,
            })
        except Exception as exc:
            tracker.fail(plan_stage, exc)
            route = policy_route
            should_retrieve = route != Route.SQL

        # ---- retrieval ----
        dense_results: List[Dict[str, Any]] = []
        sparse_results: List[Dict[str, Any]] = []

        if should_retrieve:
            dense_stage = tracker.start("Dense Retrieval", "TF-IDF similarity search")
            try:
                dense_results = self.retriever.search_dense(query, k=8)
                tracker.end(
                    dense_stage,
                    detail={
                        "candidates": len(dense_results),
                        "top_ids": [d["id"] for d in dense_results[:3]],
                    },
                )
            except Exception as exc:
                tracker.fail(dense_stage, exc)
                raise

            sparse_stage = tracker.start("Sparse Retrieval", "BM25 lexical search")
            try:
                sparse_results = self.retriever.search_bm25(query, k=8)
                tracker.end(
                    sparse_stage,
                    detail={
                        "candidates": len(sparse_results),
                        "top_ids": [d["id"] for d in sparse_results[:3]],
                    },
                )
            except Exception as exc:
                tracker.fail(sparse_stage, exc)
                raise

            fusion_stage = tracker.start("Fusion (RRF)", "Fuse dense and sparse rankings")
            try:
                context_docs = self.retriever.fuse(dense_results, sparse_results, k_rrf=60, top_k=6)
                citations = [
                    {
                        "doc_id": doc["id"],
                        "meta": doc.get("meta", {}),
                        "rrf": doc.get("_rrf"),
                    }
                    for doc in context_docs
                ]
                tracker.end(
                    fusion_stage,
                    detail={
                        "fused": len(context_docs),
                        "top_ids": [d["id"] for d in context_docs[:3]],
                    },
                )
            except Exception as exc:
                tracker.fail(fusion_stage, exc)
                raise
        else:
            tracker.skip("Dense Retrieval", "TF-IDF similarity search", "Planner disabled retrieval")
            tracker.skip("Sparse Retrieval", "BM25 lexical search", "Planner disabled retrieval")
            tracker.skip("Fusion (RRF)", "Fuse dense and sparse rankings", "Planner disabled retrieval")
            route = route if 'route' in locals() else policy_route

        # ---- final skill execution ----
        answer = ""
        if route == Route.SQL:
            answer = self._handle_sql(query, session_id=session_id, tracker=tracker)
            citations = []
        elif route == Route.CODE:
            answer = self._handle_code(
                query,
                context_docs,
                history_text,
                plan_steps,
                session_id=session_id,
                tracker=tracker,
                citations=citations,
            )
        else:
            answer = self._handle_rag(
                query,
                context_docs,
                history_text,
                plan_steps,
                session_id=session_id,
                tracker=tracker,
                citations=citations,
            )

        # ---- memory write ----
        memory_write = tracker.start("Memory Write", "Persist turn to long-term store")
        try:
            self.memory.save(session_id, query, answer, citations)
            tracker.end(memory_write, detail={"stored": True})
        except Exception as exc:
            tracker.fail(memory_write, exc)

        logger.info("Agentic run completed — route=%s", route.value)

        context_preview = [
            {
                "id": doc.get("id"),
                "meta": doc.get("meta", {}),
                "snippet": (doc.get("text") or "")[:400],
                "rrf": doc.get("_rrf"),
            }
            for doc in context_docs
        ]

        return {
            "answer": answer,
            "citations": citations,
            "route": route.value,
            "plan": plan_steps,
            "trace": tracker.export(),
            "memory": history,
            "context": context_preview,
            "planner_notes": planner_notes,
        }

    # ---- helpers for each skill ----

    def _handle_rag(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        history_text: str,
        plan_steps: List[str],
        *,
        session_id: str,
        tracker: StageTracker,
        citations: List[Dict[str, Any]],
    ) -> str:
        context_text = self._context_from_docs(context_docs)
        plan_text = "\n".join(f"- {step}" for step in plan_steps) or "- Retrieve relevant passages\n- Compose grounded answer"

        prompt = f"""
You are the answering agent in a retrieval-augmented system. Follow the plan and
ground the answer strictly in the provided CONTEXT when possible. If the
information is not available, say so explicitly and avoid fabrication.

PLAN:
{plan_text}

CONVERSATION MEMORY:
{history_text or '(no prior turns)'}

RETRIEVED CONTEXT:
{context_text or '(no documents retrieved)'}

USER QUESTION:
{query}

Respond with bullet points and cite sources like [doc-1].
"""

        stage = tracker.start("LLM Generation", "Compose grounded answer")
        try:
            text, stats = generate_with_stats(MODEL.get("rag", "llama3"), prompt, max_tokens=800, temperature=0.1)
            self._record_stats(
                stats,
                session_id=session_id,
                route=Route.RAG.value,
                model_role="rag",
                query=query,
                citations=citations,
            )
            tracker.end(
                stage,
                detail={
                    "model": MODEL.get("rag", "llama3"),
                    "prompt_tokens": stats.get("prompt_tokens"),
                    "completion_tokens": stats.get("completion_tokens"),
                },
            )
            return text
        except Exception as exc:
            tracker.fail(stage, exc)
            raise

    def _handle_code(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        history_text: str,
        plan_steps: List[str],
        *,
        session_id: str,
        tracker: StageTracker,
        citations: List[Dict[str, Any]],
    ) -> str:
        context_text = self._context_from_docs(context_docs)
        plan_text = "\n".join(f"- {step}" for step in plan_steps) or "- Understand the requested change\n- Provide annotated code"

        prompt = f"""
You are a senior software engineer. Follow the provided PLAN and use CONTEXT and
conversation MEMORY when relevant. Produce:
- A brief reasoning section
- Annotated code snippets in fences
- Testing or run instructions

PLAN:
{plan_text}

CONVERSATION MEMORY:
{history_text or '(no prior turns)'}

CONTEXT:
{context_text or '(no documents retrieved)'}

REQUEST:
{query}
"""

        stage = tracker.start("LLM Generation", "Author code response")
        try:
            text, stats = generate_with_stats(MODEL.get("code", "llama3"), prompt, max_tokens=900, temperature=0.2)
            self._record_stats(
                stats,
                session_id=session_id,
                route=Route.CODE.value,
                model_role="code",
                query=query,
                citations=citations,
            )
            tracker.end(
                stage,
                detail={
                    "model": MODEL.get("code", "llama3"),
                    "prompt_tokens": stats.get("prompt_tokens"),
                    "completion_tokens": stats.get("completion_tokens"),
                },
            )
            return text
        except Exception as exc:
            tracker.fail(stage, exc)
            raise

    def _handle_sql(
        self,
        query: str,
        *,
        session_id: str,
        tracker: StageTracker,
    ) -> str:
        schema_hint = """
-- Tables available:
-- memories(session_id TEXT, user TEXT, assistant TEXT, citations TEXT, ts DATETIME)
-- docs(id TEXT PRIMARY KEY, text TEXT, source TEXT)
"""

        stage = tracker.start("SQL Generation", "Draft SQL query from instructions")
        try:
            llm_text, stats = generate_with_stats(
                MODEL.get("sql", "sqlcoder"),
                f"""You are SQLCoder. Produce a single SQLite query in a fenced ```sql``` block that satisfies the user request.
Use only the tables described below. After the fenced block, add a one sentence explanation.

DB SCHEMA:
{schema_hint}

USER REQUEST:
{query}
""",
                max_tokens=512,
                temperature=0.0,
            )
            self._record_stats(
                stats,
                session_id=session_id,
                route=Route.SQL.value,
                model_role="sql",
                query=query,
            )
            tracker.end(
                stage,
                detail={
                    "model": MODEL.get("sql", "sqlcoder"),
                    "prompt_tokens": stats.get("prompt_tokens"),
                    "completion_tokens": stats.get("completion_tokens"),
                },
            )
        except Exception as exc:
            tracker.fail(stage, exc)
            raise

        sql = _extract_sql(llm_text)
        exec_stage = tracker.start("SQL Execution", "Run generated SQL against knowledge store")
        if not sql:
            tracker.end(exec_stage, status="skipped", detail={"reason": "No SQL fenced block detected"})
            return llm_text

        try:
            conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            conn.close()
            payload = {"columns": cols, "rows": rows}
            tracker.end(exec_stage, detail={"rows": len(rows)})
            return llm_text + "\n\n-- Execution Result --\n" + json.dumps(payload, indent=2, default=str)
        except Exception as exc:
            tracker.fail(exec_stage, exc)
            return llm_text + f"\n\n[Execution Error] {exc}"
