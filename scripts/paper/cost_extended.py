"""Break-even including output tokens.  For each run and optimizer, with the method's
best compromise (selected on S) as the deployed prompt:

    break_even(rho) = (I_search + rho * O_search) / ((I_nc - I_c) + rho * (O_nc - O_c))

I = input tokens, O = output tokens (objective tokenizer / provider completion tokens),
per request on T for the uncompressed prompt (nc) and the compromise (c); rho = price
(or cost) of an output token relative to an input token.  rho = 0 is the input-only
figure reported in rq3_cost.csv.  Writes $PP_PAPER_DIR/numbers/cost_extended.txt and
$PP_PAPER_DIR/figures/rq3_cost_extended.csv."""
from paths import ROOT, RESULTS, PAPER, FIG, TAB, NUM  # noqa: E402,F401  (chdir to repo root)
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from prompt_piper.analysis.results import load_runs, select_compromise, per_sample

OPT = ["nsga2", "random_search", "multi_armed_bandit"]
rows = []
for run in load_runs(RESULTS):
    evals = {e["id"]: e for e in run.stream("evaluations")}
    def out_tokens(df):
        return float(np.mean([evals[i]["output_tokens"] for i in df.eval_id]))
    nc = per_sample(run, "no_compression", 0, "test")
    I_nc, O_nc = nc.tokens.mean(), out_tokens(nc)
    for m in OPT:
        idx, _ = select_compromise(run, m, "search")
        c = per_sample(run, m, idx, "test")
        s_ev = [e for e in evals.values() if e["method"] == m and e["split"] == "search"]
        I_s = sum(e["input_tokens"] for e in s_ev); O_s = sum(e["output_tokens"] for e in s_ev)
        dI, dO = I_nc - c.tokens.mean(), O_nc - out_tokens(c)
        row = {"model": run.model, "task": run.name, "method": m, "I_search": I_s, "O_search": O_s,
               "saved_input": dI, "saved_output": dO, "out_nc": O_nc, "out_c": O_nc - dO}
        for rho in (0, 1, 4):
            den = dI + rho * dO
            row[f"be_rho{rho}"] = (I_s + rho * O_s) / den if den > 0 else np.inf
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(f"{FIG}/rq3_cost_extended.csv", index=False)
lines = []
for m, g in df.groupby("method", sort=False):
    lines.append(f"{m}: output tokens/request nc {g.out_nc.mean():.1f} vs compromise {g.out_c.mean():.1f}; "
                 f"saved output mean {g.saved_output.mean():.1f} (cells with longer outputs: {(g.saved_output < 0).sum()}/16); "
                 f"search O/I ratio {g.O_search.sum()/g.I_search.sum():.3f}")
    for rho in (0, 1, 4):
        v = g[f"be_rho{rho}"]
        fin = v[np.isfinite(v)]
        lines.append(f"   rho={rho}: median {v.median():.0f} q1 {v.quantile(.25):.0f} q3 {v.quantile(.75):.0f}; "
                     f"never pays back in {int((~np.isfinite(v)).sum())}/16 cells")
(NUM / "cost_extended.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
