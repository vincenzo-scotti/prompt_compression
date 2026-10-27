"""Search-to-test shift of the best-compromise points (selection optimism).
Writes $PP_PAPER_DIR/numbers/generalization.txt."""
from paths import ROOT, RESULTS, PAPER, FIG, TAB, NUM  # noqa: E402,F401  (chdir to repo root)
from pathlib import Path
import pandas as pd
p = pd.read_csv(f"{FIG}/rq2_compromise_points.csv")
o = p[p.method.isin(["nsga2", "random_search", "multi_armed_bandit"])].copy()
o["d"] = o.test_performance - o.search_performance
lines = ["test - search performance of the compromise point (mean, mean |.|)",
         o.groupby("method").d.agg(["mean", lambda v: v.abs().mean()]).round(3).to_string(),
         o.groupby("task").d.agg(["mean", lambda v: v.abs().mean()]).round(3).to_string(),
         f"overall mean {o.d.mean():.3f} mean abs {o.d.abs().mean():.3f}"]
(NUM / "generalization.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
