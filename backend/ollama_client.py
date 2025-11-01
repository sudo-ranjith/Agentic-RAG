import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

import requests
from dotenv import load_dotenv

# ---------------- Logging ----------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{log_dir}/ollama_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------- Config ----------------
load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# Module-level storage of the last stats (for backward-compatible generate())
_LAST_STATS: Dict = {}


def _normalize_stats(model: str, started_at: float, payload: dict, meta: Optional[dict]) -> Dict:
    """
    Build a uniform stats dict from non-stream or stream metadata.
    Ollama fields of interest:
      - prompt_eval_count
      - eval_count
      - total_duration (nanoseconds)
      - load_duration (nanoseconds)
      - prompt_eval_duration (nanoseconds)
      - eval_duration (nanoseconds)
    """
    now = time.time()
    latency_ms = int((now - started_at) * 1000)

    # Prefer meta (stream final object) if present; otherwise read from payload response.
    prompt_count = 0
    completion_count = 0
    total_duration_ns = None
    load_duration_ns = None
    prompt_dur_ns = None
    eval_dur_ns = None

    src = meta or payload or {}

    # Ollama non-stream response: fields are at the top level
    prompt_count = int(src.get("prompt_eval_count") or 0)
    completion_count = int(src.get("eval_count") or 0)

    total_duration_ns = src.get("total_duration")
    load_duration_ns = src.get("load_duration")
    prompt_dur_ns = src.get("prompt_eval_duration")
    eval_dur_ns = src.get("eval_duration")

    out = {
        "model": model,
        "prompt_tokens": prompt_count,
        "completion_tokens": completion_count,
        "total_tokens": prompt_count + completion_count,
        "latency_ms": latency_ms,
        "raw": {
            "total_duration_ns": total_duration_ns,
            "load_duration_ns": load_duration_ns,
            "prompt_eval_duration_ns": prompt_dur_ns,
            "eval_duration_ns": eval_dur_ns,
        },
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
    }
    return out


def generate_with_stats(
    model: str,
    prompt: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    stream: bool = False,
    options: Optional[dict] = None,
    system: Optional[str] = None,
) -> Tuple[str, Dict]:
    """
    Call Ollama /api/generate.

    Returns:
        (text, stats_dict)

    stats_dict fields:
        - model
        - prompt_tokens
        - completion_tokens
        - total_tokens
        - latency_ms
        - raw: { total_duration_ns, load_duration_ns, prompt_eval_duration_ns, eval_duration_ns }
        - ts
    """
    url = f"{OLLAMA_HOST}/api/generate"

    opts = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }
    if options:
        # allow caller to override anything (e.g., top_p, repeat_penalty)
        opts.update(options)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": bool(stream),
        "options": opts,
    }
    if system:
        payload["system"] = system

    headers = {"Content-Type": "application/json"}

    started = time.time()

    try:
        if not stream:
            r = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            text = (data.get("response") or "").strip()
            stats = _normalize_stats(model, started, data, None)
            logger.info(
                "Ollama generate (model=%s, stream=%s): %d chars, %s ms, in=%d out=%d",
                model, stream, len(text), stats["latency_ms"],
                stats["prompt_tokens"], stats["completion_tokens"]
            )
            return text, stats

        # ---- streaming ----
        buff = []
        last_meta = None
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=DEFAULT_TIMEOUT) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                # Append text if present
                piece = obj.get("response") or ""
                if piece:
                    buff.append(piece)
                # Capture final metadata when done
                if obj.get("done"):
                    last_meta = obj
                    break

        text = "".join(buff).strip()
        stats = _normalize_stats(model, started, {}, last_meta)
        logger.info(
            "Ollama stream (model=%s): %d chars, %s ms, in=%d out=%d",
            model, len(text), stats["latency_ms"],
            stats["prompt_tokens"], stats["completion_tokens"]
        )
        return text, stats

    except requests.exceptions.RequestException as e:
        logger.error("Ollama request failed (model=%s): %s", model, e, exc_info=True)
        # Return empty text + zeroed stats
        empty_stats = {
            "model": model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": int((time.time() - started) * 1000),
            "raw": {},
            "ts": datetime.utcnow().isoformat(timespec="seconds"),
            "error": str(e),
        }
        return "", empty_stats


def generate(
    model: str,
    prompt: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    stream: bool = False,
    options: Optional[dict] = None,
    system: Optional[str] = None,
) -> str:
    """
    Backward-compatible helper used by your code today.
    Returns only the text but also stores stats in a module-global
    so you can call get_last_stats() after this to log telemetry.
    """
    text, stats = generate_with_stats(
        model,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        options=options,
        system=system,
    )
    global _LAST_STATS
    _LAST_STATS = stats
    return text


def get_last_stats() -> Dict:
    """
    Returns the stats dict for the last generate() call in this process.
    If none, returns {}.
    """
    return _LAST_STATS.copy() if _LAST_STATS else {}
