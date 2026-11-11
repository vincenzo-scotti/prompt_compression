"""Paired bootstrap of per-cell test-performance differences at the best compromise.

For each of the 16 cells (model x benchmark), the test samples are resampled with
replacement (the same resample for both methods, so the comparison is paired) and the
difference of mean test performance is recomputed.  Comparisons:
  * each optimizer vs No compression;
  * the three optimizers pairwise.
The interval is the 95% percentile interval of B resamples.  It reflects the sampling
variability of the test set given the selected strategies, not the variability of the
search itself (one run per optimizer and cell).

Inputs:  $PP_RESULTS_DIR (run logs), $PP_PAPER_DIR/figures/rq2_compromise_points.csv
Outputs: $PP_PAPER_DIR/figures/rq2_bootstrap.csv, $PP_PAPER_DIR/numbers/bootstrap.txt
"""
from paths import ROOT, RESULTS, PAPER, FIG, TAB, NUM  # noqa: E402,F401  (chdir to repo root)

import numpy as np
import pandas as pd

from prompt_piper import analysis as A

B = 10_000
SEED = 0
OPT = ["nsga2", "random_search", "multi_armed_bandit"]
PAIRS = ([(o, "no_compression") for o in OPT]
         + [("nsga2", "random_search"), ("nsga2", "multi_armed_bandit"),
            ("random_search", "multi_armed_bandit")])


def main():
    runs = {(r.model, r.name): r for r in A.load_runs(RESULTS)}
    pts = pd.read_csv(FIG / "rq2_compromise_points.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for (model, task), run in sorted(runs.items()):
        cell = pts[(pts.model == model) & (pts.task == task)].set_index("method")
        scores = {}
        for m in OPT + ["no_compression"]:
            d = A.per_sample(run, m, int(cell.loc[m, "solution_index"]))
            scores[m] = d.set_index("sample_id").score
        ids = scores["no_compression"].index
        for m in scores:
            assert set(scores[m].index) == set(ids), (model, task, m)
        mat = {m: scores[m].loc[ids].to_numpy(float) for m in scores}
        idx = rng.integers(0, len(ids), size=(B, len(ids)))
        for a, b in PAIRS:
            diff = mat[a] - mat[b]
            boot = diff[idx].mean(axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            rows.append({"model": model, "task": task, "a": a, "b": b, "n": len(ids),
                         "diff": diff.mean(), "ci_low": lo, "ci_high": hi,
                         "verdict": "a>b" if lo > 0 else ("a<b" if hi < 0 else "n.s.")})
    df = pd.DataFrame(rows)
    df.to_csv(FIG / "rq2_bootstrap.csv", index=False)

    lines = [f"Paired bootstrap (B={B}) of per-cell test-performance differences, 95% percentile CI"]
    for (a, b), g in df.groupby(["a", "b"], sort=False):
        c = g.verdict.value_counts()
        lines.append(f"{a} vs {b}: cells={len(g)} a>b={c.get('a>b', 0)} a<b={c.get('a<b', 0)} "
                     f"ns={c.get('n.s.', 0)} mean_diff={g['diff'].mean():.3f} "
                     f"min_diff={g['diff'].min():.2f} max_diff={g['diff'].max():.2f}")
        for _, r in g[g.verdict != "n.s."].iterrows():
            lines.append(f"    {r.model} {r.task}: diff={r['diff']:.2f} "
                         f"CI=[{r.ci_low:.2f}, {r.ci_high:.2f}]")
    text = "\n".join(lines) + "\n"
    (NUM / "bootstrap.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
