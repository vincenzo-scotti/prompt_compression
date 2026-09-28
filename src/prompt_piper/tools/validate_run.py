"""Check that a run's logs are complete: no missing data points.

Usage::

    python -m prompt_piper.tools.validate_run results/icse_26/<model>/<run>_log [...]

Exits with status 1 if any check fails.  The same function is called by the
compressor at the end of every run and its report is stored in the result
JSON (``validation``) and in the log's ``meta.json``.

Checks
------
* search and test sample ids are recorded, and disjoint when drawn from the
  same split;
* every phase that started also ended;
* NSGA-II: one candidate record per evaluated individual (matching pymoo's
  evaluation counter), and one population snapshot per generation;
* Random Search / MAB: exactly ``n_evaluations`` candidate records;
* one intermediate Pareto front per generation (NSGA-II) or per evaluation
  (Random Search, MAB);
* every expected test re-evaluation (each Pareto solution, each solution of
  each baseline) is present;
* every sample's full content, sections, the seeds and bit layouts are
  recorded, and every candidate's operator names round-trip from its bits;
* every population member, front point and MAB round links to a candidate
  record whose per-sample values reproduce its averages;
* every candidate covers every sample of its split, with a score, token count
  and a resolvable evaluation record, and every referenced text is stored.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_run(log_dir: str | Path) -> dict:
    """Validate the logs in *log_dir*; return ``{"ok", "errors", "counts"}``."""
    log_dir = Path(log_dir)
    errors: list[str] = []
    meta_path = log_dir / "meta.json"
    if not meta_path.exists():
        return {"ok": False, "errors": [f"missing {meta_path}"], "counts": {}}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = meta.get("expected", {})

    candidates = _read(log_dir / "candidates.jsonl")
    evaluations = _read(log_dir / "evaluations.jsonl")
    states = _read(log_dir / "states.jsonl")
    phases = _read(log_dir / "phases.jsonl")
    shas = {t["sha"] for t in _read(log_dir / "texts.jsonl")}

    # --- splits ---------------------------------------------------------
    split_ids = meta.get("sample_ids", {})
    for split in ("search", "test"):
        if split not in split_ids:
            errors.append(f"sample ids of split {split!r} not recorded")
    if meta.get("same_source_split") and {"search", "test"} <= split_ids.keys():
        overlap = set(split_ids["search"]) & set(split_ids["test"])
        if overlap:
            errors.append(f"search and test samples overlap: {sorted(overlap)}")

    # --- samples, sections, seeds, layouts --------------------------------
    samples = _read(log_dir / "samples.jsonl")
    sections = _read(log_dir / "sections.jsonl")
    for role in ("search", "test"):
        if role not in split_ids:
            continue
        logged = sorted(r["sample_id"] for r in samples if r.get("role") == role)
        if logged != sorted(split_ids[role]) and (role == "test" or any(
            r.get("role") == "search" for r in samples
        )):
            errors.append(f"samples.jsonl: {role} samples {len(logged)} != {len(split_ids[role])} ids")
        missing_doc = [r["sample_id"] for r in samples if r.get("role") == role and "doc" not in r
                       and meta.get("dry_run") is False]
        if missing_doc:
            errors.append(f"samples.jsonl: {role} samples without raw document: {missing_doc[:5]}")
    sectioned = {(r["split"], r["sample_id"]) for r in sections}
    for c in candidates:
        for ps in c.get("per_sample", []):
            if (c.get("split"), ps.get("sample_id")) not in sectioned:
                errors.append(f"sections.jsonl: no sections for {c.get('split')} sample {ps.get('sample_id')}")
                break
    if not meta.get("seeds"):
        errors.append("meta.json: seeds not recorded")
    layouts = meta.get("bit_layouts", {})
    if not layouts:
        errors.append("meta.json: bit layouts not recorded")
    else:
        from prompt_piper.optimization.bits import bits_to_names

        by_len = {len(v): v for v in layouts.values()}
        for c in candidates:
            bits = c.get("bits")
            if bits is None:
                continue
            layout = by_len.get(len(bits))
            if layout is None:
                errors.append(f"candidate {c['id']}: {len(bits)} bits match no recorded layout")
            elif bits_to_names(layout, bits) != c.get("transformation_names"):
                errors.append(f"candidate {c['id']}: operator names do not match bits under the layout")

    # --- phases ---------------------------------------------------------
    open_phases = Counter()
    for p in phases:
        key = (p.get("phase"), p.get("method"))
        open_phases[key] += 1 if p["event"] == "start" else -1
    for key, n in open_phases.items():
        if n != 0:
            errors.append(f"phase {key} started but did not end ({n} open)")

    # --- search candidates ------------------------------------------------
    by_method = Counter(
        c.get("method") for c in candidates if c.get("phase") == "search"
        or c.get("phase") == "baseline_search"
    )
    for method, n in expected.get("search_candidates", {}).items():
        if by_method.get(method, 0) != n:
            errors.append(
                f"{method}: {by_method.get(method, 0)} search candidates logged, expected {n}"
            )
    pops = [s for s in states if s.get("kind") == "population"]
    if expected.get("n_gen") is not None and expected.get("algorithm") in ("nsga2", "nsga3", "moead"):
        if len(pops) != expected["n_gen"]:
            errors.append(f"{len(pops)} population snapshots, expected {expected['n_gen']}")
        budget = expected.get("pop_size", 0) * expected["n_gen"]
        logged_alg = by_method.get(expected["algorithm"], 0)
        if expected.get("pop_size") and logged_alg != budget:
            errors.append(
                f"{expected['algorithm']}: {logged_alg} candidates evaluated, budget is "
                f"pop_size × n_gen = {budget}"
            )
        if pops:
            n_evals = pops[-1].get("n_evals")
            logged = by_method.get(expected["algorithm"], 0)
            if n_evals is not None and n_evals >= 0 and logged != n_evals:
                errors.append(f"{logged} search candidates logged but pymoo evaluated {n_evals}")
    fronts = Counter(s.get("algorithm") for s in states if s.get("kind") == "front")
    for method, n in expected.get("search_candidates", {}).items():
        if fronts.get(method, 0) != n:
            errors.append(f"{method}: {fronts.get(method, 0)} intermediate fronts logged, expected {n}")
    if expected.get("n_gen") is not None and expected.get("algorithm") in ("nsga2", "nsga3", "moead"):
        if fronts.get(expected["algorithm"], 0) != expected["n_gen"]:
            errors.append(
                f"{fronts.get(expected['algorithm'], 0)} {expected['algorithm']} fronts logged, "
                f"expected {expected['n_gen']}"
            )
    mab_rounds = sum(1 for s in states if s.get("kind") == "mab_round")
    n_mab = expected.get("search_candidates", {}).get("multi_armed_bandit")
    if n_mab is not None and mab_rounds != n_mab:
        errors.append(f"{mab_rounds} MAB rounds logged, expected {n_mab}")

    # --- test candidates --------------------------------------------------
    test_labels = defaultdict(set)
    for c in candidates:
        if c.get("split") == "test":
            test_labels[c.get("method")].add(c.get("label"))
    for method, labels in expected.get("test_labels", {}).items():
        missing = set(labels) - test_labels.get(method, set())
        if missing:
            errors.append(f"{method}: missing test evaluations for {sorted(missing)}")

    for method in expected.get("baselines", []):
        for phase in ("baseline_search", "baseline_test"):
            if not any(c.get("method") == method and c.get("phase") == phase for c in candidates):
                errors.append(f"{method}: no {phase} candidates logged")

    # --- per-sample completeness -----------------------------------------
    eval_ids = {e["id"] for e in evaluations}
    for e in evaluations:
        for key in ("prompt_sha", "original_sha"):
            if e.get(key) not in shas:
                errors.append(f"evaluation {e['id']}: text {key} not stored")
        if not isinstance(e.get("score"), (int, float)) or math.isnan(e["score"]):
            errors.append(f"evaluation {e['id']}: invalid score {e.get('score')!r}")
    for c in candidates:
        n_expected = len(split_ids.get(c.get("split"), [])) or c.get("n_samples")
        per = c.get("per_sample", [])
        if len(per) != n_expected:
            errors.append(
                f"candidate {c['id']} ({c.get('method')}, {c.get('split')}): "
                f"{len(per)} samples, expected {n_expected}"
            )
        for ps in per:
            if ps.get("eval_id") not in eval_ids:
                errors.append(f"candidate {c['id']}: unresolved evaluation {ps.get('eval_id')}")
                break
            if ps.get("prompt_sha") not in shas or ps.get("tokens") is None:
                errors.append(f"candidate {c['id']}: missing text or token count")
                break
            if not isinstance(ps.get("score"), (int, float)) or math.isnan(ps["score"]):
                errors.append(f"candidate {c['id']}: invalid score")
                break
        if split_ids.get(c.get("split")) and per:
            ids = sorted(ps["sample_id"] for ps in per)
            if ids != sorted(split_ids[c["split"]]):
                errors.append(f"candidate {c['id']}: sample ids differ from the {c['split']} split")

    # --- state snapshots: every average must be backed by per-sample values ---
    by_id = {c["id"]: c for c in candidates}
    for st in states:
        entries = []
        if st.get("kind") == "population":
            entries = st.get("population", []) + st.get("offspring", [])
        elif st.get("kind") == "front":
            entries = st.get("pareto_front", [])
        elif st.get("kind") == "mab_round":
            entries = [{"candidate_id": st.get("candidate_id"), "bits": st.get("bits"),
                        "avg_performance": st.get("avg_performance", st.get("reward")),
                        "avg_tokens": st.get("avg_tokens")}]
        for e in entries:
            c = by_id.get(e.get("candidate_id"))
            if c is None:
                errors.append(f"state {st.get('kind')} step {st.get('step', st.get('generation', st.get('round')))}: "
                              f"no candidate record for bits (per-sample values missing)")
                break
            per = c["per_sample"]
            mean_tok = sum(ps["tokens"] for ps in per) / len(per)
            mean_perf = sum(ps["score"] for ps in per) / len(per)
            if [int(b) for b in c["bits"]] != [int(b) for b in e["bits"]] or \
                    abs(mean_tok - e["avg_tokens"]) > 1e-9 or abs(mean_perf - e["avg_performance"]) > 1e-9:
                errors.append(f"state {st.get('kind')}: averages do not match candidate {c['id']}")
                break

    counts = {
        "candidates": len(candidates),
        "evaluations": len(evaluations),
        "texts": len(shas),
        "states": len(states),
        "search_candidates_by_method": dict(by_method),
        "test_candidates_by_method": {m: len(v) for m, v in test_labels.items()},
    }
    return {"ok": not errors, "errors": errors[:200], "n_errors": len(errors), "counts": counts}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    status = 0
    for d in argv:
        report = validate_run(d)
        print(f"{d}: {'OK' if report['ok'] else 'FAILED'}  {json.dumps(report['counts'])}")
        for e in report["errors"]:
            print(f"  - {e}")
        status |= not report["ok"]
    return status


if __name__ == "__main__":
    raise SystemExit(main())
