"""Load experiment results and logs, and compute every reported quantity.

Everything here works from what a run saved (result JSON + ``{run}_log/``),
without re-running experiments.  Design choices that affect reported numbers:

* **Compromise point** — the solution closest (Euclidean) to the ideal point
  (performance 1, normalised tokens 0).  It is selected on the **search** split
  by default (``selection="search"``); ``"test"`` reproduces the selection used
  in the submitted paper, which looked at test values.
* **Every multi-solution method is a point**: NSGA-II, Random Search and MAB
  are each represented by their own compromise solution, selected with the
  same rule (requires the per-solution ``solutions`` field of new results).
* **Hypervolume** — exact 2-D hypervolume (pymoo) of (compression, performance)
  with reference point (0, 0).
* **No silent fallbacks** — a missing test value is an error, never replaced by
  the search value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SEARCH_METHODS = ("nsga2", "random_search", "multi_armed_bandit")
POINT_BASELINES = (
    "no_compression", "all_transformations", "summarization",
    "truncated_prompt", "random_deletion", "llmlingua2",
)
METHOD_LABELS = {
    "nsga2": "NSGA-II",
    "no_compression": "No compression",
    "all_transformations": "All transforms",
    "random_search": "Random search",
    "multi_armed_bandit": "MAB",
    "summarization": "Summarization",
    "truncated_prompt": "Truncated",
    "random_deletion": "Rand. deletion",
    "llmlingua2": "LLMLingua-2",
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """One (model, task) run: result JSON plus its log directory."""

    path: Path
    data: dict
    model: str
    _streams: dict = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return self.data["run_name"]

    @property
    def algorithm(self) -> str:
        return self.data["config"]["run"].get("algorithm", "nsga2")

    @cached_property
    def log_dir(self) -> Path | None:
        d = self.data.get("log_dir")
        if d and Path(d).exists():
            return Path(d)
        local = self.path.parent / f"{self.name}_log"
        return local if local.exists() else None

    def stream(self, name: str) -> list[dict]:
        """Records of a log stream (``candidates``, ``evaluations``, …)."""
        if name not in self._streams:
            if self.log_dir is None:
                raise FileNotFoundError(f"{self.name}: no log directory")
            self._streams[name] = _read_jsonl(self.log_dir / f"{name}.jsonl")
        return self._streams[name]

    @cached_property
    def meta(self) -> dict:
        if self.log_dir is None:
            return {}
        return json.loads((self.log_dir / "meta.json").read_text(encoding="utf-8"))

    def baseline(self, name: str) -> dict | None:
        return next((b for b in self.data["baselines"] if b["name"] == name), None)

    def ref_tokens(self, split: str) -> float:
        """Uncompressed prompt length on *split* (normalises token counts)."""
        nc = self.baseline("no_compression")
        if nc is None:
            raise ValueError(f"{self.name}: no_compression baseline missing")
        return float(_value(nc, split, "tokens"))

    def solutions(self, method: str) -> list[dict]:
        """All solutions of *method* with search and test objectives."""
        if method == self.algorithm:
            return self.data["pareto_solutions"]
        b = self.baseline(method)
        if b is None:
            return []
        if b.get("solutions"):
            return b["solutions"]
        return [b]  # legacy result: single (possibly averaged) point


def _value(entry: dict, split: str, what: str) -> float:
    key = f"{split}_avg_{'tokens' if what == 'tokens' else 'performance'}"
    v = entry.get(key)
    if v is None:
        raise ValueError(f"missing {key} in {entry.get('name', 'solution')}")
    return float(v)


def load_runs(results_dir: str | Path, models: Iterable[str] | None = None) -> list[Run]:
    """Load every run result under ``results_dir/<model>/<run>.json``."""
    results_dir = Path(results_dir)
    runs = []
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if models is not None and model_dir.name not in models:
            continue
        for path in sorted(model_dir.glob("*.json")):
            if path.name.startswith("transfer_"):
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if "pareto_solutions" in data:
                runs.append(Run(path=path, data=data, model=model_dir.name))
    return runs


def load_transfers(results_dir: str | Path) -> pd.DataFrame:
    """One row per transferred solution: model, source, target, test objectives."""
    rows = []
    for path in sorted(Path(results_dir).glob("*/transfer_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        for i, s in enumerate(d["solutions"]):
            rows.append({
                "model": path.parent.name, "source": d["source_run"], "target": d["target_run"],
                "solution": i, "test_tokens": s["test_avg_tokens"],
                "test_performance": s["test_avg_performance"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Selection and quality indicators
# ---------------------------------------------------------------------------

def compromise_index(perfs: Iterable[float], tokens: Iterable[float], ref_tokens: float) -> int:
    """Index of the point closest to (performance 1, normalised tokens 0)."""
    perfs, tokens = np.asarray(list(perfs), float), np.asarray(list(tokens), float)
    return int(np.argmin(np.hypot(1.0 - perfs, tokens / max(ref_tokens, 1e-12))))


def select_compromise(run: Run, method: str, selection: str = "search") -> tuple[int, dict]:
    """Return ``(index, solution)`` of *method*'s compromise, chosen on *selection*."""
    sols = run.solutions(method)
    if not sols:
        raise ValueError(f"{run.name}: no solutions for {method}")
    i = compromise_index(
        [_value(s, selection, "perf") for s in sols],
        [_value(s, selection, "tokens") for s in sols],
        run.ref_tokens(selection),
    )
    return i, sols[i]


def hypervolume(points: Iterable[tuple[float, float]]) -> float:
    """Exact 2-D hypervolume of (compression, performance) points, reference (0, 0)."""
    from pymoo.indicators.hv import HV

    pts = np.clip(np.asarray(list(points), float).reshape(-1, 2), 0.0, 1.0)
    if len(pts) == 0:
        return 0.0
    return float(HV(ref_point=np.zeros(2))(-pts))


def front_hypervolume(run: Run, method: str, split: str = "test") -> float:
    ref = run.ref_tokens(split)
    return hypervolume(
        (1 - _value(s, split, "tokens") / ref, _value(s, split, "perf"))
        for s in run.solutions(method)
    )


def method_points(runs: Iterable[Run], selection: str = "search") -> pd.DataFrame:
    """Test-set point of every method in every run (one row per run × method).

    Multi-solution methods contribute their compromise solution selected on
    *selection*; single-point baselines contribute their only point.  The
    ``legacy_front_mean`` column flags old results whose RS/MAB value is the
    mean of their front (per-solution values were not saved).
    """
    rows = []
    for run in runs:
        ref = run.ref_tokens("test")
        methods = [run.algorithm] + [b["name"] for b in run.data["baselines"]]
        for method in methods:
            sols = run.solutions(method)
            if not sols:
                continue
            multi = method in SEARCH_METHODS
            idx, sol = select_compromise(run, method, selection) if multi else (0, sols[0])
            legacy = multi and method != run.algorithm and not run.baseline(method).get("solutions")
            tok, perf = _value(sol, "test", "tokens"), _value(sol, "test", "perf")
            rows.append({
                "model": run.model, "task": run.name, "method": method,
                "label": METHOD_LABELS.get(method, method), "solution_index": idx,
                "test_tokens": tok, "test_performance": perf,
                "compression": 1 - tok / ref, "utility": perf + (1 - tok / ref),
                "search_tokens": sol.get("search_avg_tokens"),
                "search_performance": sol.get("search_avg_performance"),
                "legacy_front_mean": legacy,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-sample data, convergence and cost (require logs)
# ---------------------------------------------------------------------------

def _test_label(run: Run, method: str, index: int) -> str | None:
    if method == run.algorithm:
        return f"pareto:{index}"
    if method in SEARCH_METHODS:
        return f"{method}:{index}"
    return None


def provider_tokens(run: Run, method: str, index: int = 0, split: str = "test") -> float | None:
    """Mean provider-reported prompt tokens of one solution (``None`` if not reported).

    Exact billed input tokens for API / llama.cpp models, logged for free with
    every model call; includes the provider's fixed request overhead (chat
    wrapping, system prompt).  Use it to report token counts in the provider's
    own units next to the tokenizer-based objective.
    """
    d = per_sample(run, method, index, split)
    if "provider_prompt_tokens" not in d or d.provider_prompt_tokens.isna().any():
        return None
    return float(d.provider_prompt_tokens.mean())


def per_sample(run: Run, method: str, index: int = 0, split: str = "test") -> pd.DataFrame:
    """Per-sample scores and token counts of one solution, from ``candidates.jsonl``."""
    label = _test_label(run, method, index) if split == "test" else None
    bits = None
    if split == "search" and method in SEARCH_METHODS:
        bits = [int(b) for b in run.solutions(method)[index]["bits"]]
    matches = [
        c for c in run.stream("candidates")
        if c["method"] == method and c["split"] == split
        and (label is None or c.get("label") == label)
        and (bits is None or [int(b) for b in (c.get("bits") or [])] == bits)
    ]
    if not matches or (split == "test" and len(matches) != 1):
        raise ValueError(f"{run.name}/{method}/{label or index}: {len(matches)} {split} candidates")
    rows = [
        {"sample_id": ps["sample_id"], "score": ps["score"], "tokens": ps["tokens"],
         "provider_prompt_tokens": ps.get("provider_prompt_tokens"),
         "cached": ps["cached"], "eval_id": ps["eval_id"]}
        for ps in matches[-1]["per_sample"]
    ]
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def standalone_cost(run: Run) -> pd.DataFrame:
    """Model calls each search method would need on its own (search split).

    Methods share one evaluation cache within a run, so a method that runs
    after another may find evaluations already done; wall-clock times and
    logged calls then understate its cost.  This counts, per method, the
    distinct (sample, prompt) pairs it evaluated and the distinct candidate
    prompt sets, independent of who evaluated them first.
    """
    per: dict = {}
    for c in run.stream("candidates"):
        if c["split"] != "search":
            continue
        m = per.setdefault(c["method"], {"candidates": 0, "pairs": set(), "prompt_sets": set()})
        m["candidates"] += 1
        shas = tuple(ps["prompt_sha"] for ps in c["per_sample"])
        m["prompt_sets"].add(shas)
        m["pairs"].update((ps["sample_id"], ps["prompt_sha"]) for ps in c["per_sample"])
    new_calls: dict = {}
    for e in run.stream("evaluations"):
        if e["split"] == "search":
            new_calls[e.get("method")] = new_calls.get(e.get("method"), 0) + 1
    return pd.DataFrame([
        {"model": run.model, "task": run.name, "method": k, "candidates": v["candidates"],
         "distinct_prompt_sets": len(v["prompt_sets"]), "standalone_llm_calls": len(v["pairs"]),
         "llm_calls_made_in_run": new_calls.get(k, 0)}
        for k, v in per.items()
    ])


def convergence(run: Run) -> pd.DataFrame:
    """Per-step front statistics from the logged intermediate fronts.

    Columns: method, step, n_visited, hypervolume, best_performance,
    min_norm_tokens, front_size (search-split objectives, tokens normalised by
    the uncompressed search prompt length).
    """
    ref = run.ref_tokens("search")
    rows = []
    for s in run.stream("states"):
        if s.get("kind") != "front":
            continue
        front = s["pareto_front"]
        perf = [p["avg_performance"] for p in front]
        tok = [p["avg_tokens"] / ref for p in front]
        rows.append({
            "method": s["algorithm"], "step": s["step"], "n_visited": s["n_visited"],
            "hypervolume": hypervolume(zip(1 - np.asarray(tok), perf)),
            "best_performance": max(perf), "min_norm_tokens": min(tok),
            "front_size": len(front),
        })
    return pd.DataFrame(rows)


def mab_trajectory(run: Run, split: str = "search") -> pd.DataFrame:
    """Per-round, per-arm evolution of the Multi-Armed Bandit beliefs.

    One row per (round, bit) with the sampled theta, whether the bit was
    active, the Beta parameters after the update, their posterior mean and
    the round's reward components (performance, compression, reward).
    """
    rows = []
    for s in run.stream("states"):
        if s.get("kind") != "mab_round" or s.get("split") != split:
            continue
        for bit, (theta, active, a, b) in enumerate(zip(s["theta"], s["bits"], s["alpha_after"], s["beta_after"])):
            rows.append({
                "model": run.model, "task": run.name, "round": s["round"], "bit": bit,
                "theta": theta, "active": bool(active), "alpha": a, "beta": b,
                "posterior_mean": a / (a + b),
                "avg_performance": s.get("avg_performance", s["reward"]),
                "compression": s.get("compression"), "reward": s["reward"],
                "reward_weight": s.get("reward_weight", 1.0),
            })
    return pd.DataFrame(rows)


def cost(run: Run) -> dict:
    """Search/test cost from the logs: calls, cache hits, tokens and time per phase."""
    evals = run.stream("evaluations")
    cands = run.stream("candidates")
    out: dict = {"model": run.model, "task": run.name}
    for split in ("search", "test"):
        e = [x for x in evals if x["split"] == split]
        out[f"{split}_llm_calls"] = len(e)
        out[f"{split}_input_tokens"] = int(sum(x["input_tokens"] for x in e))
        out[f"{split}_output_tokens"] = int(sum(x.get("output_tokens", 0) for x in e))
        # Provider-reported (billed) usage, when the backend returns it.
        usages = [x.get("provider_usage") for x in e if x.get("provider_usage")]
        out[f"{split}_provider_usage_records"] = len(usages)
        out[f"{split}_provider_prompt_tokens"] = int(sum(u.get("prompt_tokens", 0) for u in usages))
        out[f"{split}_provider_completion_tokens"] = int(sum(u.get("completion_tokens", 0) for u in usages))
        out[f"{split}_provider_reasoning_tokens"] = int(sum(
            (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in usages
        ))
        out[f"{split}_cache_hits"] = int(sum(
            ps["cached"] for c in cands if c["split"] == split for ps in c["per_sample"]
        ))
    by_method: dict = {}
    for e in evals:
        if e.get("split") == "search":
            by_method[e.get("method")] = by_method.get(e.get("method"), 0) + 1
    out["search_llm_calls_by_method"] = by_method
    for p in run.stream("phases"):
        if p["event"] == "end":
            key = f"seconds_{p['phase']}_{p.get('method', '')}".rstrip("_")
            out[key] = out.get(key, 0.0) + p["elapsed_seconds"]
    calls = run.stream("llm_calls")
    out["aux_llm_calls"] = sum(c.get("source", "llm") == "llm" for c in calls)
    aux_usage = [c.get("provider_usage") for c in calls if c.get("provider_usage")]
    out["aux_provider_prompt_tokens"] = int(sum(u.get("prompt_tokens", 0) for u in aux_usage))
    out["aux_provider_completion_tokens"] = int(sum(u.get("completion_tokens", 0) for u in aux_usage))
    out["aux_llm_cache_hits"] = sum(c.get("source") in ("disk", "memory") for c in calls)
    return out


def operator_frequency(runs: Iterable[Run], effective_only: bool = True) -> pd.DataFrame:
    """Share of Pareto solutions using each (section, operator), per run.

    With ``effective_only`` (default) only operators whose removal changes at
    least one search prompt are counted; neutral bits (set but without effect)
    are ignored.  Results without that information raise an error rather than
    silently counting neutral bits.
    """
    rows = []
    for run in runs:
        sols = run.data["pareto_solutions"]
        counts: dict = {}
        for s in sols:
            names = s["transformation_names"]
            if effective_only:
                names = s.get("effective_transformation_names")
                if names is None:
                    raise ValueError(f"{run.name}: no effective operators saved; use effective_only=False")
            for name in names:
                counts[name] = counts.get(name, 0) + 1
        for name, n in counts.items():
            section, _, op = name.partition(":")
            rows.append({
                "model": run.model, "task": run.name,
                "section": {"I": "instruction", "S": "shots", "D": "data"}.get(section, section),
                "operator": op, "frequency": n / len(sols),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def rank_biserial(x, y) -> float:
    """Matched-pairs rank-biserial correlation (positive: x > y)."""
    from scipy.stats import rankdata

    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def compare_methods(points: pd.DataFrame, metric: str, reference: str = "nsga2",
                    higher_is_better: bool = True) -> tuple[dict, pd.DataFrame]:
    """Friedman test over (model, task) cells + Holm-corrected Wilcoxon post-hoc.

    Returns ``(omnibus, posthoc)``.  The comparison against No compression is
    two-sided; all others are one-sided in favour of *reference*.
    """
    from scipy.stats import friedmanchisquare, wilcoxon

    wide = points.pivot_table(index=["model", "task"], columns="method", values=metric)
    missing = wide.columns[wide.isna().any()].tolist()
    if missing:
        raise ValueError(f"{metric}: methods missing in some (model, task) cells: {missing}")
    stat, p = friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    n, k = wide.shape
    omnibus = {"metric": metric, "n_cells": n, "n_methods": k, "chi2": stat, "p": p,
               "kendall_w": stat / (n * (k - 1))}
    rows, pvals = [], []
    x = wide[reference].to_numpy()
    for method in wide.columns:
        if method == reference:
            continue
        y = wide[method].to_numpy()
        if method == "no_compression":
            alt = "two-sided"
        else:
            alt = "greater" if higher_is_better else "less"
        pv = 1.0 if np.allclose(x, y) else wilcoxon(x, y, alternative=alt).pvalue
        pvals.append(pv)
        rows.append({
            "metric": metric, "baseline": method, "alternative": alt,
            "reference_median": float(np.median(x)), "baseline_median": float(np.median(y)),
            "wins": int((x > y + 1e-12).sum()), "losses": int((x < y - 1e-12).sum()),
            "rank_biserial": rank_biserial(x, y) * (1 if higher_is_better else -1),
            "p_raw": pv,
        })
    posthoc = pd.DataFrame(rows)
    if len(posthoc):
        posthoc["p_holm"] = holm(pvals)
    return omnibus, posthoc
