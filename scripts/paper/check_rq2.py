"""Step 1.3 check: recompute RQ2 numbers independently from the CSVs and raw logs."""
from paths import ROOT, RESULTS, PAPER, FIG, TAB, NUM  # noqa: E402,F401  (chdir to repo root)
import json, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
from pymoo.indicators.hv import HV

F = f"{FIG}/"
pts = pd.read_csv(F + "rq2_compromise_points.csv")
summ = pd.read_csv(F + "rq2_compromise_summary.csv", header=[0, 1], index_col=0)
post = pd.read_csv(F + "rq2_posthoc.csv"); omni = pd.read_csv(F + "rq2_friedman.csv")
per = pd.read_csv(F + "rq2_per_sample.csv")
hv = pd.read_csv(F + "rq2_hypervolume.csv")
ok = True
# 1. summary = mean of per-cell points
m = pts.groupby("method").test_performance.mean()
ok &= np.allclose(m.sort_index(), summ[("test_performance", "mean")].sort_index())
# 2. per-sample means reproduce every cell's point
g = per.groupby(["task", "method"]).score.mean()
cell = pts.groupby(["task", "method"]).test_performance.mean()
ok &= np.allclose(g.sort_index(), cell.sort_index())
# 3. Friedman on performance recomputed
w = pts.pivot_table(index=["model", "task"], columns="method", values="test_performance")
chi2 = friedmanchisquare(*[w[c] for c in w.columns]).statistic
ok &= abs(chi2 - omni[omni.metric == "test_performance"].chi2.iloc[0]) < 1e-9
# 4. raw Wilcoxon p for NSGA-II vs LLMLingua-2 and NSGA-II vs RS (compression)
p1 = wilcoxon(w["nsga2"], w["llmlingua2"], alternative="greater").pvalue
r1 = post[(post.metric == "test_performance") & (post.a == "nsga2") & (post.b == "llmlingua2")].p_raw.iloc[0]
wc = pts.pivot_table(index=["model", "task"], columns="method", values="compression")
p2 = wilcoxon(wc["nsga2"], wc["random_search"]).pvalue
r2 = post[(post.metric == "compression") & (post.family == "B") & (post.a == "nsga2") & (post.b == "random_search")].p_raw.iloc[0]
ok &= abs(p1 - r1) < 1e-12 and abs(p2 - r2) < 1e-12
# 5. hypervolume recomputed from the raw result JSON with pymoo
ind = HV(ref_point=np.zeros(2)); bad = 0
for path in glob.glob(f"{RESULTS}/*/*_nsga2.json"):
    if "/transfer_" in path: continue
    d = json.load(open(path)); model = path.split("/")[-2]
    nc = next(b for b in d["baselines"] if b["name"] == "no_compression")["test_avg_tokens"]
    fronts = {"nsga2": d["pareto_solutions"], **{b["name"]: b["solutions"] for b in d["baselines"]
              if b["name"] in ("random_search", "multi_armed_bandit")}}
    for meth, sols in fronts.items():
        P = np.clip(np.array([(1 - s["test_avg_tokens"] / nc, s["test_avg_performance"]) for s in sols]), 0, 1)
        v = ind(-P)
        row = hv[(hv.model == model) & (hv.task == d["run_name"]) & (hv.method == meth)].hv_test.iloc[0]
        bad += abs(v - row) > 1e-12
ok &= bad == 0
print("RQ2 numbers reproduced:", bool(ok), "| hypervolume mismatches:", bad)
