"""Complete, append-only logging of everything an experiment computes.

A :class:`RunRecorder` writes one directory per run, so that every result
(for every approach, on every split) can be recomputed from the logs alone,
without re-running experiments.  Records are appended as JSON lines and
flushed immediately, so an interrupted run loses nothing already computed.

Streams (one ``.jsonl`` file each)
----------------------------------
``samples``
    Full content of every sample used, per role (search / test): sample id,
    source split, raw lm-eval document, target and every prompt field.
``sections``
    The uncompressed instruction / shots / input data / output cue of every
    sample, as split by :class:`~prompt_piper.transformations.sections.PromptSplitter`.
``texts``
    Content-addressed store of every prompt text: ``{"sha", "text"}``.
    Written once per unique text; other streams reference texts by ``sha``.
``evaluations``
    One record per *actual* target-LLM evaluation (cache misses only):
    sample id, split, prompt sha, token counts, raw output, filtered answer,
    metrics, score, batch timing, and the context that triggered it.
``candidates``
    One record per scoring of a candidate strategy (every NSGA-II individual,
    every Random Search / MAB draw, every baseline, every test re-evaluation),
    with the per-sample breakdown (sample id, evaluation id, tokens, score,
    cache hit) and aggregate objectives.
``states``
    Algorithm state snapshots: full NSGA-II population per generation,
    MAB Beta parameters per round, etc.
``llm_calls``
    Every auxiliary LLM call (semantic operators, fitting): prompt, output,
    timing, cache hit.
``phases``
    Start / end / elapsed of every phase (fitting, search, baselines, test).
``meta.json``
    Configuration, sample ids of every split, disjointness check, software
    versions, commit hash, hardware; updated at the end of the run.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

STREAMS = ("texts", "samples", "sections", "evaluations", "candidates", "states", "llm_calls", "phases")


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds")


def _default(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def text_sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


class RunRecorder:
    """Append-only JSONL recorder for one experiment run.

    Parameters
    ----------
    run_dir:
        Directory that receives the streams.  Created if missing.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._files = {
            name: open(self.run_dir / f"{name}.jsonl", "a", encoding="utf-8")
            for name in STREAMS
        }
        self._seen_texts: set[str] = set()
        self._counters: dict[str, int] = defaultdict(int)
        self._context: dict[str, Any] = {}
        # (method, split, bits) -> id of the candidate record holding the
        # per-sample values, so state snapshots can reference it explicitly.
        self._candidate_ids: dict[tuple, int] = {}
        self._meta: dict[str, Any] = {"created": _now(), "environment": _environment()}
        self.write_meta()

    # ------------------------------------------------------------------
    # Context (who is evaluating what)
    # ------------------------------------------------------------------

    @property
    def context(self) -> dict[str, Any]:
        return dict(self._context)

    @contextlib.contextmanager
    def scope(self, **kwargs) -> Iterator[None]:
        """Temporarily extend the context attached to every record."""
        previous = self._context
        self._context = {**previous, **kwargs}
        try:
            yield
        finally:
            self._context = previous

    @contextlib.contextmanager
    def phase(self, name: str, **kwargs) -> Iterator[None]:
        """Time a phase and attach ``phase=name`` to records emitted inside it."""
        start = time.perf_counter()
        wall_start = time.time()
        started = _now()
        self.write("phases", {"event": "start", "phase": name, "time": started, **kwargs})
        with self.scope(phase=name, **kwargs):
            try:
                yield
            finally:
                self.write("phases", {
                    "event": "end", "phase": name, "start_time": started, "time": _now(),
                    # Monotonic time excludes system suspension; wall-clock time
                    # includes it, so a difference reveals a suspended machine.
                    "elapsed_seconds": time.perf_counter() - start,
                    "wall_elapsed_seconds": time.time() - wall_start, **kwargs,
                })

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def next_id(self, stream: str) -> int:
        with self._lock:
            self._counters[stream] += 1
            return self._counters[stream]

    def write(self, stream: str, record: dict) -> None:
        line = json.dumps(record, default=_default, ensure_ascii=False)
        with self._lock:
            f = self._files[stream]
            f.write(line + "\n")
            f.flush()

    def register_candidate(self, method, split, bits, candidate_id: int) -> None:
        self._candidate_ids[(method, split, tuple(int(b) for b in bits))] = candidate_id

    def candidate_id(self, method, split, bits) -> int | None:
        """Id of the latest candidate record for these bits (per-sample values)."""
        return self._candidate_ids.get((method, split, tuple(int(b) for b in bits)))

    def store_text(self, text: str) -> str:
        """Store *text* once and return its sha."""
        sha = text_sha(text)
        with self._lock:
            new = sha not in self._seen_texts
            self._seen_texts.add(sha)
        if new:
            self.write("texts", {"sha": sha, "text": text})
        return sha

    def update_meta(self, **kwargs) -> None:
        self._meta.update(kwargs)
        self.write_meta()

    def write_meta(self) -> None:
        tmp = self.run_dir / "meta.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2, default=_default, ensure_ascii=False)
        tmp.replace(self.run_dir / "meta.json")

    def close(self) -> None:
        self.update_meta(closed=_now())
        with self._lock:
            for f in self._files.values():
                f.close()


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "argv": sys.argv,
    }
    try:
        root = Path(__file__).resolve().parents[3]
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        env["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip())
    except Exception:
        pass
    versions = {}
    for pkg in ("numpy", "pymoo", "lm_eval", "torch", "transformers", "llmlingua", "bitsandbytes"):
        try:
            from importlib.metadata import version
            versions[pkg] = version(pkg.replace("_", "-"))
        except Exception:
            pass
    env["packages"] = versions
    try:
        import torch
        if torch.cuda.is_available():
            env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return env


# ---------------------------------------------------------------------------
# Crash recovery: replay evaluations already paid for in interrupted runs
# ---------------------------------------------------------------------------

#: Fields of an evaluation record that describe *this* run's logging rather
#: than the model's answer; everything else is the evaluator's detail.
_EVAL_META_KEYS = frozenset({
    "id", "split", "sample_index", "sample_id", "target", "original_sha", "prompt_sha",
    "original_tokens", "input_tokens", "batch_id", "batch_size", "batch_elapsed_seconds",
    "output_tokens", "replayed_from", "phase", "method",
})


class ReplayStore:
    """Evaluations from interrupted runs of the same experiment, for reuse.

    Keyed by ``(method, split, sample_id, prompt_sha)``: a restarted run that
    needs exactly this evaluation (same method, same sample, same prompt text)
    reuses the stored answer instead of calling the model again.  Because the
    restart repeats the search with the same seeds and gets the same answers,
    it retraces the interrupted run exactly up to the crash point.

    Only logs whose ``replay_fingerprint`` (model, generation settings, task,
    splits) equals the current one are used.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple, tuple[dict, str]] = {}
        self.sources: list[dict] = []
        self.used = 0

    def __len__(self) -> int:
        return len(self._entries)

    @classmethod
    def from_logs(cls, log_dirs, fingerprint: str) -> "ReplayStore":
        store = cls()
        for d in sorted(Path(p) for p in log_dirs):
            meta_path = d / "meta.json"
            info = {"log": str(d), "used": False, "records": 0, "reason": None}
            store.sources.append(info)
            if not meta_path.exists():
                info["reason"] = "no meta.json"
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("replay_fingerprint") != fingerprint:
                info["reason"] = "different model / generation settings / task (fingerprint mismatch)"
                continue
            for line in open(d / "evaluations.jsonl", encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:  # last line cut by the crash
                    continue
                key = (e.get("method"), e.get("split"), e.get("sample_id"), e.get("prompt_sha"))
                store._entries.setdefault(key, (e, str(d)))
                info["records"] += 1
            info["used"] = True
        return store

    def lookup(self, method, split, sample_id, prompt: str):
        """Return ``(detail, replayed_from)`` or ``None``."""
        hit = self._entries.get((method, split, sample_id, text_sha(prompt)))
        if hit is None:
            return None
        e, source = hit
        self.used += 1
        detail = {k: v for k, v in e.items() if k not in _EVAL_META_KEYS}
        timing = {"batch_size": e.get("batch_size"),
                  "batch_elapsed_seconds": e.get("batch_elapsed_seconds")}
        origin = e.get("replayed_from") or {"log": source, "id": e.get("id")}
        return detail, timing, origin
