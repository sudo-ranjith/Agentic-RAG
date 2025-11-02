import os, io, logging, time
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, Body, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from retriever import Retriever
from agents import Memory, AgenticRAG
from telemetry import Telemetry  # ✅ NEW import
from pypdf import PdfReader
from docx import Document
import sqlite3
import json

# ---------- Logging ----------
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

file_handler = logging.FileHandler(
    f'{log_dir}/api_{datetime.now().strftime("%Y%m%d")}.log',
    encoding="utf-8"  # ✅ file logs in UTF-8
)
stream_handler = logging.StreamHandler()  # console

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)


# ---------- Config ----------
load_dotenv()
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]

app = FastAPI(title="Agentic RAG (Lite) — Hybrid + RRF + Telemetry")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = Retriever()
memory = Memory()
telemetry = Telemetry()  # ✅ instantiate telemetry
agent = AgenticRAG(retriever, memory, telemetry)

logger.info("Initialized Retriever, Memory, and Telemetry services")
logger.info("=== API Server Starting ===")
logger.info(f"CORS Origins: {CORS_ORIGINS}")

# ---------- Schemas ----------
class AskRequest(BaseModel):
    query: str
    session_id: str = "default"

class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    route: str
    plan: List[str] = Field(default_factory=list)
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    context: List[Dict[str, Any]] = Field(default_factory=list)
    planner_notes: str | None = None

# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Ingest ----------
@app.post("/ingest")
def ingest(text: str = Body(..., embed=True)):
    logger.info("Processing text ingestion request")
    try:
        n = retriever.add_texts([text], metadatas=[{"source": "api"}])
        return {"added": n}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise

# ---------- Upload ----------
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    logger.info(f"Processing upload request for {len(files)} files")
    texts, metas = [], []
    for f in files:
        name = f.filename or "file"
        start_time = time.time()
        try:
            content = await f.read()
            ext = (name.split(".")[-1] or "").lower()
            if ext in ("txt", "md", "csv", "log"):
                text = content.decode("utf-8", errors="ignore")
            elif ext == "pdf":
                reader = PdfReader(io.BytesIO(content))
                text = "\n".join([p.extract_text() or "" for p in reader.pages])
            elif ext == "docx":
                doc = Document(io.BytesIO(content))
                text = "\n".join([p.text for p in doc.paragraphs])
            else:
                metas.append({"source": name, "skipped": True})
                continue
            texts.append(text); metas.append({"source": name})
            logger.info(f"Processed {name} in {time.time()-start_time:.2f}s")
        except Exception as e:
            metas.append({"source": name, "error": str(e)})
            logger.error(f"Failed to process {name}: {e}", exc_info=True)

    valid = [(t, m) for t, m in zip(texts, metas) if t and not m.get("skipped")]
    n = retriever.add_texts([v[0] for v in valid], metadatas=[v[1] for v in valid]) if valid else 0
    return {"uploaded": len(files), "ingested": n, "metas": metas}

# ---------- Ask ----------
@app.post("/ask", response_model=AskResponse)
def ask(body: Dict[str, Any] = Body(...)):
    """
    Accepts flexible payloads:
      - {"query": "text", "session_id": "default"}
      - {"query": {"query":"text","session_id":"default"}}
      - form-like where 'query' may be a JSON string
    Normalizes to (query_text, session_id) and validates.
    """
    req_id = f"req_{int(time.time())}"
    logger.info(f"[{req_id}] Received /ask payload: keys={list(body.keys())}")

    # Normalize payload
    q_val = body.get("query")
    session = body.get("session_id") or "default"

    # If query missing entirely, try if body itself is a string (raw)
    if q_val is None and isinstance(body, str):
        q_val = body

    # If query looks like a JSON string, try parsing
    if isinstance(q_val, str):
        q_text = q_val.strip()
        if q_text.startswith("{") and q_text.endswith("}"):
            try:
                parsed = json.loads(q_text)
                # If nested object: {"query": "...", "session_id": "..."} 
                if isinstance(parsed, dict):
                    if "query" in parsed:
                        q_text = parsed.get("query")
                        session = parsed.get("session_id", session)
            except Exception:
                # keep q_text as-is (plain string)
                pass
    elif isinstance(q_val, dict):
        # nested object case
        q_text = q_val.get("query") or q_val.get("text") or None
        session = q_val.get("session_id", session)
    else:
        q_text = None

    # Final validation
    if not isinstance(q_text, str) or not q_text.strip():
        logger.warning(f"[{req_id}] Invalid query payload: {q_val!r}")
        raise HTTPException(status_code=422, detail=[{
            "type": "string_type",
            "loc": ["body", "query"],
            "msg": "Input should be a valid string",
            "input": q_val
        }])

    q_text = q_text.strip()
    logger.info(f"[{req_id}] Normalized query='{q_text[:80]}' session_id={session}")

    try:
        start_time = time.time()
        result = agent.run(q_text, session_id=session)
        logger.info(
            f"[{req_id}] Done in {time.time()-start_time:.2f}s route={result.get('route')} plan_steps={len(result.get('plan', []))}"
        )
        return AskResponse(**result)
    except Exception as e:
        logger.error(f"[{req_id}] Ask failed: {e}", exc_info=True)
        raise

# ---------- DB Info ----------
@app.get("/dbinfo")
def dbinfo():
    logger.info("DB info requested")
    try:
        dense_count = retriever.count_dense() if hasattr(retriever, "count_dense") else None
    except Exception:
        dense_count = None

    try:
        conn = sqlite3.connect(os.getenv("SQLITE_PATH", "./rag_memory.db"))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM docs")
        sparse_count = cur.fetchone()[0]
        conn.close()
    except Exception:
        sparse_count = None

    return {
        "vector_db": {
            "name": "Chroma",
            "collection": getattr(retriever, "collection_name", "rag_docs"),
            "path": os.getenv("CHROMA_DIR", ".chroma"),
            "doc_count": dense_count,
        },
        "sparse_db": {
            "name": "SQLite",
            "path": os.getenv("SQLITE_PATH", "./rag_memory.db"),
            "tables": ["docs", "memories", "transactions"],
            "doc_count": sparse_count,
        },
    }

# ---------- Documents ----------
@app.get("/documents")
def docs(offset: int = 0, limit: int = 20):
    logger.info(f"Docs list offset={offset} limit={limit}")
    try:
        conn = sqlite3.connect(os.getenv("SQLITE_PATH", "./rag_memory.db"))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM docs")
        total = cur.fetchone()[0]
        cur.execute("SELECT id, source, substr(text,1,500) FROM docs ORDER BY rowid DESC LIMIT ? OFFSET ?", (limit, offset))
        rows = [{"id": r[0], "source": r[1], "snippet": r[2]} for r in cur.fetchall()]
        conn.close()
        return {"offset": offset, "limit": limit, "total": total, "items": rows}
    except Exception as e:
        logger.error(f"Docs fetch failed: {e}", exc_info=True)
        return {"offset": offset, "limit": limit, "total": 0, "items": []}

# ---------- Metrics Dashboard ----------
@app.get("/metrics/summary")
def metrics_summary(
    since: str = Query(None, description="ISO datetime (UTC). Default last 30 days."),
    until: str = Query(None, description="ISO datetime (UTC)."),
):
    try:
        return telemetry.summary(since=since, until=until)
    except Exception as e:
        logger.error(f"Metrics summary failed: {e}", exc_info=True)
        return {"input_tokens": 0, "output_tokens": 0, "guard_tokens_saved": 0, "guard_topics_count": 0}

@app.get("/metrics/by-model")
def metrics_by_model(
    since: str = Query(None),
    until: str = Query(None),
):
    try:
        rows = telemetry.tokens_by_model_role(since=since, until=until)
        return {"rows": rows}
    except Exception as e:
        logger.error(f"Metrics by model failed: {e}", exc_info=True)
        return {"rows": []}

@app.get("/metrics/recent")
def metrics_recent(limit: int = 10):
    try:
        return {"rows": telemetry.recent(limit=limit)}
    except Exception as e:
        logger.error(f"Metrics recent failed: {e}", exc_info=True)
        return {"rows": []}
