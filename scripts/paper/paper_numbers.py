"""All numbers quoted in the paper text, from the CSVs written by make_figures.py and
from the result files.  Output: $PP_PAPER_DIR/numbers/paper_numbers.txt"""
from paths import ROOT, RESULTS, PAPER, FIG, TAB, NUM  # noqa: E402,F401  (chdir to repo root)
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from prompt_piper import analysis as A
F = FIG
out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))
runs = A.load_runs(RESULTS)
pts = pd.read_csv(F / "rq2_compromise_points.csv")
OPT = ["nsga2", "random_search", "multi_armed_bandit"]

P("== Uncompressed prompt (test) per task: mean tokens, mean performance, per model")
for t, g in pts[pts.method == "no_compression"].groupby("task"):
    P(t, "tokens %.0f perf %.3f" % (g.test_tokens.mean(), g.test_performance.mean()),
      {m: (round(r.test_tokens), round(r.test_performance, 2)) for m, r in zip(g.model, g.itertuples())})

P("\n== Best compromise per task and method (means over models): perf, tokens, compression")
for t, g in pts[pts.method.isin(OPT + ["no_compression", "llmlingua2"])].groupby("task"):
    P(t, {m: (round(x.test_performance.mean(), 3), round(x.test_tokens.mean()), round(x.compression.mean(), 3))
          for m, x in g.groupby("method")})

P("\n== Overall means (16 cells)")
s = pts.groupby("method")[["test_performance", "test_tokens", "compression"]].mean()
P(s.round(3).to_string())

P("\n== Largest test compression on each technique's test front with performance >= uncompressed (no-loss), per task range over models")
rows = []
for r in runs:
    nc = r.baseline("no_compression"); ref = nc["test_avg_tokens"]
    for m in OPT:
        ok = [1 - s["test_avg_tokens"] / ref for s in r.solutions(m) if s["test_avg_performance"] >= nc["test_avg_performance"]]
        rows.append({"task": r.name, "model": r.model, "method": m, "noloss": max(ok) if ok else np.nan,
                     "max_comp": max(1 - s["test_avg_tokens"] / ref for s in r.solutions(m))})
nl = pd.DataFrame(rows)
for t, g in nl.groupby("task"):
    best = g.groupby("model").noloss.max()
    P(t, "no-loss compression (best of 3 techniques) per model:", best.round(3).to_dict(),
      "| max compression on fronts: %.3f-%.3f" % (g.max_comp.min(), g.max_comp.max()))
P("cells with a no-loss strategy (any technique):", int(nl.groupby(["task", "model"]).noloss.max().notna().sum()), "of 16")
P("front max compression overall range: %.3f-%.3f" % (nl.max_comp.min(), nl.max_comp.max()))

P("\n== Hypervolume"); P(pd.read_csv(F / "rq2_hypervolume.csv").groupby("method")[["hv_search", "hv_test", "front_size"]].agg(["mean", "median"]).round(3).to_string())
P("\n== Friedman"); P(pd.read_csv(F / "rq2_friedman.csv").round(5).to_string())
post = pd.read_csv(F / "rq2_posthoc.csv")
P("\n== Post-hoc (Holm)"); P(post[["metric", "family", "a", "b", "alternative", "median_a", "median_b", "wins_a", "wins_b", "rank_biserial", "p_holm"]].round(4).to_string())
P("\n== LLMLingua-2 matched"); P(pd.read_csv(F / "rq2_llmlingua_matched_summary.csv").round(4).to_string())
ll = pd.read_csv(F / "rq2_llmlingua_matched.csv")
P("LLMLingua-2 test compression range %.3f-%.3f, mean %.3f; perf mean %.3f; MBPP perf %s" % (
    ll.llmlingua2_compression.min(), ll.llmlingua2_compression.max(), ll.llmlingua2_compression.mean(),
    ll.llmlingua2_performance.mean(), ll[ll.task == "mbpp_instruct_nsga2"].llmlingua2_performance.round(2).tolist()))
P("\n== Cost summary"); P(pd.read_csv(F / "rq3_cost_summary.csv").round(2).to_string())
cost = pd.read_csv(F / "rq3_cost.csv")
P("search hours per optimizer, range over cells:", cost[cost.method.isin(OPT)].groupby("method").search_hours.agg(["min", "max", "mean"]).round(2).to_dict())
P("single-point baselines total minutes (search+test) max:", round(((cost[~cost.method.isin(OPT)].search_hours + cost[~cost.method.isin(OPT)].test_hours) * 60).max(), 1))
P("\n== Convergence 95% HV (strategies), per method median/min/max:")
c95 = pd.read_csv(F / "rq3_convergence_95.csv"); P(c95.groupby("method").strategies_to_95pct_hv.agg(["median", "min", "max"]).to_string())
P("\n== Operators"); g = pd.read_csv(F / "rq4_operator_usage.csv")
P("neutral share of set bits: %.3f" % (1 - g.effective_count.sum() / g.set_count.sum()))
P(pd.read_csv(F / "rq4_operator_effective_share.csv", index_col=0).round(2).to_string())
P("\n== Transfer"); tr = pd.read_csv(F / "rq4_transfer.csv")
P(tr.groupby(["direction", "kind"])[["performance", "tokens"]].mean().round(3).to_string())
P(tr.groupby(["model", "direction", "kind"])[["performance", "tokens"]].mean().round(2).to_string())
P("\n== Run setup")
d = json.load(open(ROOT / f"{RESULTS}/qwen3.5-9b/gsm8k_cot_nsga2.json"))
P("S/T sizes:", {r.name: (len(r.data["search_sample_ids"]), len(r.data["test_sample_ids"])) for r in runs if r.model == "qwen3.5-9b"})
P("MAB config:", next(b for b in d["baselines"] if b["name"] == "multi_armed_bandit")["config"])
P("LLMLingua config:", next(b for b in d["baselines"] if b["name"] == "llmlingua2")["config"])
P("trunc/deletion:", [b["config"] for b in d["baselines"] if b["name"] in ("truncated_prompt", "random_deletion")])
(NUM / "paper_numbers.txt").write_text("\n".join(out))
print("\n".join(out))
