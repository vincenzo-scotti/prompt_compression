"""Step 1.1 check: independent re-implementation of the best-compromise selection."""
from paths import ROOT, RESULTS, PAPER, FIG, TAB, NUM  # noqa: E402,F401  (chdir to repo root)
import glob, json, math, sys
sys.path.insert(0, "src")
from prompt_piper import analysis as A

def dominated(p, q):  # q dominates p (min tokens, max perf)
    return q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1])

agree = total = 0; issues = []
for path in sorted(p for p in glob.glob(f"{RESULTS}/*/*_nsga2.json") if "/transfer_" not in p):
    d = json.load(open(path))
    nc = next(b for b in d["baselines"] if b["name"] == "no_compression")
    ref = nc["search_avg_tokens"]
    fronts = {"nsga2": d["pareto_solutions"]}
    for b in d["baselines"]:
        if b["name"] in ("random_search", "multi_armed_bandit"):
            fronts[b["name"]] = b["solutions"]
    run = next(r for r in A.load_runs(RESULTS) if str(r.path) == path)
    for m, sols in fronts.items():
        pts = [(s["search_avg_tokens"], s["search_avg_performance"]) for s in sols]
        for i, p in enumerate(pts):
            if any(dominated(p, q) for j, q in enumerate(pts) if j != i):
                issues.append(f"{path} {m}: solution {i} is dominated on S")
        dist = [math.hypot(1 - perf, tok / ref) for tok, perf in pts]
        best = min(range(len(dist)), key=dist.__getitem__)
        ties = [i for i, x in enumerate(dist) if abs(x - dist[best]) < 1e-12 and pts[i] != pts[best]]
        if ties:
            issues.append(f"{path} {m}: tie between distinct points {best} and {ties}")
        i_lib, _ = A.select_compromise(run, m, "search")
        total += 1; agree += (pts[i_lib] == pts[best])
print(f"agreement {agree}/{total}")
print("issues:", issues or "none")
