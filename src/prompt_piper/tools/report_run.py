"""Human-readable sanity report of one or more run logs.

Usage::

    python -m prompt_piper.tools.report_run results/smoke/qwen3.5-9b/*_log [--show-prompt]

For each run log it prints what is needed to check that a run behaved as
intended before trusting (or scaling up) it:

* validation status, seeds, sample ids, model endpoint;
* the static prompt structure (instruction / shots / cue / tail sizes) and,
  with ``--show-prompt``, one full uncompressed prompt and the model's answer;
* per method: candidates, model calls, cache hits, mean score;
* answer quality signals: errors, ``[invalid]`` extractions, empty outputs;
* thinking: calls that returned reasoning text or reasoning tokens (must be 0
  where thinking is disabled);
* token accounting: tokenizer count vs provider-reported prompt tokens;
* timing: seconds per model call, time per phase, auxiliary LLM calls.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def report(log_dir: Path, show_prompt: bool = False) -> None:
    from prompt_piper.tools.validate_run import validate_run

    meta = json.loads((log_dir / "meta.json").read_text(encoding="utf-8"))
    evals = _read(log_dir / "evaluations.jsonl")
    cands = _read(log_dir / "candidates.jsonl")
    sections = _read(log_dir / "sections.jsonl")
    calls = _read(log_dir / "llm_calls.jsonl")
    phases = _read(log_dir / "phases.jsonl")
    texts = {t["sha"]: t["text"] for t in _read(log_dir / "texts.jsonl")}
    v = validate_run(log_dir)

    print("=" * 100)
    print(f"{log_dir}  —  validation: {'OK' if v['ok'] else 'FAILED'}")
    for e in v["errors"][:10]:
        print(f"   ! {e}")
    print(f"endpoint: {meta.get('model_endpoint')}   seeds: {meta.get('seeds')}")
    ids = meta.get("sample_ids", {})
    print(f"samples: search {len(ids.get('search', []))}  test {len(ids.get('test', []))}  "
          f"disjoint: {meta.get('disjoint')}  static sections identical S/T: "
          f"{meta.get('static_sections_identical_across_splits')}")

    if sections:
        s0 = sections[0]
        print(f"structure: instruction {len(s0.get('instruction', ''))} chars | shots "
              f"{len(s0.get('shots', ''))} chars | data {len(s0.get('input_data', ''))} chars | "
              f"cue {s0.get('output_cue')!r} | sep_before_cue {s0.get('sep_before_cue')!r} | "
              f"tail {s0.get('tail')!r}")

    # --- per method ----------------------------------------------------------
    print(f"\n{'method':22s} {'split':6s} {'cands':>6s} {'calls':>6s} {'hits':>6s} {'mean score':>10s}")
    by = defaultdict(lambda: {"c": 0, "hits": 0, "scores": []})
    for c in cands:
        k = (c["method"], c["split"])
        by[k]["c"] += 1
        by[k]["hits"] += c["n_cache_hits"]
        by[k]["scores"].append(c["avg_performance"])
    ncalls = Counter((e.get("method"), e["split"]) for e in evals)
    for (m, sp), d in sorted(by.items()):
        print(f"{m:22s} {sp:6s} {d['c']:6d} {ncalls[(m, sp)]:6d} {d['hits']:6d} {_mean(d['scores']):10.3f}")

    # --- answers -------------------------------------------------------------
    errors = Counter(e.get("error") for e in evals if e.get("error"))
    invalid = sum(1 for e in evals if any(f == "[invalid]" for f in (e.get("filtered") or [])))
    empty = sum(1 for e in evals if "output" in e and not str(e.get("output", "")).strip())
    replayed = sum(1 for e in evals if "replayed_from" in e)
    print(f"\nmodel calls: {len(evals)} (of which {replayed} reused from interrupted runs) | "
          f"errors: {dict(errors) or 0} | '[invalid]' extractions: {invalid} | empty outputs: {empty}")

    # --- thinking ------------------------------------------------------------
    reasoning_text = sum(1 for e in evals if e.get("provider_reasoning"))
    reasoning_toks = [((e.get("provider_usage") or {}).get("completion_tokens_details") or {})
                      .get("reasoning_tokens", 0) for e in evals]
    print(f"thinking: calls with reasoning text {reasoning_text} | calls with reasoning tokens "
          f"{sum(1 for r in reasoning_toks if r)} (total {sum(reasoning_toks)})")

    # --- tokens --------------------------------------------------------------
    pairs = [(e["input_tokens"], (e.get("provider_usage") or {}).get("prompt_tokens"))
             for e in evals if (e.get("provider_usage") or {}).get("prompt_tokens") is not None]
    if pairs:
        diffs = [p - t for t, p in pairs]
        print(f"tokens: tokenizer vs provider prompt tokens over {len(pairs)} calls — provider − "
              f"tokenizer: mean {_mean(diffs):.1f}, min {min(diffs)}, max {max(diffs)} "
              f"(constant ≈ chat-template overhead)")
    out_toks = [e.get("output_tokens") for e in evals if e.get("output_tokens") is not None]
    print(f"output tokens per call: mean {_mean(out_toks):.0f}, max {max(out_toks) if out_toks else '-'}")

    # --- timing --------------------------------------------------------------
    per_call = {}
    for e in evals:
        per_call.setdefault(e.get("batch_id"), e["batch_elapsed_seconds"] / max(e["batch_size"], 1))
    print(f"seconds per model call (batch time / batch size): mean {_mean(per_call.values()):.2f}")
    tot = defaultdict(float)
    for p in phases:
        if p["event"] == "end":
            tot[f"{p['phase']}/{p.get('method', '')}"] += p["elapsed_seconds"]
    print("time per phase: " + ", ".join(f"{k} {v:.0f}s" for k, v in sorted(tot.items())))
    src = Counter(c.get("source") for c in calls)
    print(f"auxiliary LLM calls (fitting / semantic operators): {dict(src) or 0}")

    if show_prompt:
        e = next((e for e in evals if e.get("method") == "no_compression" and e["split"] == "test"), None)
        if e:
            print("\n--- uncompressed test prompt (sample "
                  f"{e['sample_id']}, score {e['score']}) ---\n{texts.get(e['prompt_sha'], '')}")
            print(f"--- model output ---\n{e.get('output')}")
            print(f"--- extracted: {e.get('filtered')}  target: {str(e.get('target'))[:200]!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_dirs", nargs="+")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args(argv)
    for d in args.log_dirs:
        report(Path(d), show_prompt=args.show_prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
