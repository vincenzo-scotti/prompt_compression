"""LaTeX tables for the TSE paper, generated from the CSVs written by make_figures.py and
cost_extended.py (no number is typed by hand).  Output: $PP_PAPER_DIR/tables/*.tex"""
from paths import ROOT, RESULTS, PAPER, FIG, TAB, NUM  # noqa: E402,F401  (chdir to repo root)
from pathlib import Path
import numpy as np, pandas as pd

FIG = FIG; OUT = TAB
OUT.mkdir(exist_ok=True)
TASKS = [("mbpp_instruct_nsga2", "MBPP"), ("mmlu_pro_cs_nsga2", "MMLU-Pro CS"),
         ("gsm8k_cot_nsga2", "GSM8K"), ("bbh_logical_deduction_nsga2", "BBH-Logic")]
OPT = [("nsga2", "NSGA-II"), ("random_search", "Random Search"), ("multi_armed_bandit", "MAB")]
BASE = [("no_compression", "No Compression"), ("all_transformations", "All Transf."),
        ("summarization", "Summarization"), ("truncated_prompt", "Truncated Prompt"),
        ("random_deletion", "Random Deletion"), ("llmlingua2", "LLMLingua-2")]
f2 = lambda v: f"{v:.2f}"


def compromise():
    p = pd.read_csv(FIG / "rq2_compromise_points.csv")
    lines = []
    for key, lab in OPT + BASE:
        cells = []
        for t, _ in TASKS:
            s = p[(p.method == key) & (p.task == t)]
            cells += [f2(s.test_performance.mean()), f2(s.compression.mean())]
        s = p[p.method == key]
        cells += [f2(s.test_performance.mean()), f2(s.compression.mean())]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if key == "multi_armed_bandit":
            lines.append(r"\midrule")
    (OUT / "compromise.tex").write_text("\n".join(lines) + "\n")


def hv():
    h = pd.read_csv(FIG / "rq2_hypervolume.csv")
    lines = []
    for key, lab in OPT:
        cells = []
        for t, _ in TASKS + [(None, "All")]:
            s = h[(h.method == key) & ((h.task == t) if t else True)]
            cells += [f"{s.hv_search.mean():.3f}", f"{s.hv_test.mean():.3f}", f"{s.front_size.median():.0f}"]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    (OUT / "hypervolume.tex").write_text("\n".join(lines) + "\n")


def posthoc():
    d = pd.read_csv(FIG / "rq2_posthoc.csv")
    fmt = lambda r: (r"\textbf{" if r.p_holm < 0.05 else "{") + f"{r.p_holm:.3f}" + "}" if r.p_holm >= 0.001 \
        else r"\textbf{$<$0.001}"
    lines = []
    for key, lab in BASE:
        cells = []
        for metric in ("test_performance", "compression"):
            for o, _ in OPT:
                r = d[(d.metric == metric) & (d.a == o) & (d.b == key)].iloc[0]
                cells += [f"{r.rank_biserial:+.2f}", fmt(r)]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    (OUT / "posthoc_a.tex").write_text("\n".join(lines) + "\n")
    lines = []
    pairs = [("nsga2", "random_search", "NSGA-II vs RS"), ("nsga2", "multi_armed_bandit", "NSGA-II vs MAB"),
             ("random_search", "multi_armed_bandit", "RS vs MAB")]
    for a, b, lab in pairs:
        cells = []
        for metric in ("test_performance", "compression", "hv_search", "hv_test"):
            r = d[(d.metric == metric) & (d.a == a) & (d.b == b) & (d.family == "B")].iloc[0]
            cells += [f"{r.rank_biserial:+.2f}", fmt(r)]
        lines.append(f"{lab} & " + " & ".join(cells) + r" & & & & \\")
    (OUT / "posthoc_b.tex").write_text("\n".join(lines) + "\n")


def cost():
    c = pd.read_csv(FIG / "rq3_cost.csv"); e = pd.read_csv(FIG / "rq3_cost_extended.csv")
    conv = pd.read_csv(FIG / "rq3_convergence_95.csv")
    lines = []
    q = lambda v: f"{np.median(v)/1e3:.1f}k ({np.quantile(v,.25)/1e3:.1f}--{np.quantile(v,.75)/1e3:.1f}k)"
    for key, lab in OPT:
        s = c[c.method == key]; x = e[e.method == key]; v = conv[conv.method == key]
        be1 = x.be_rho1.replace(np.inf, np.nan)
        never = int(np.isinf(x.be_rho1).sum())
        lines.append(" & ".join([
            lab, f"{s.search_calls.mean()/1e3:.1f}k", f"{s.search_input_tokens.mean()/1e6:.1f}M",
            f"{s.search_output_tokens.mean()/1e6:.1f}M",
            f"{s.search_hours.mean():.1f} ({s.search_hours.min():.1f}--{s.search_hours.max():.1f})",
            f"{x.saved_input.mean():.0f}", f"{-x.saved_output.mean():+.0f}",
            q(x.be_rho0), f"{np.nanmedian(be1.fillna(np.inf))/1e3:.1f}k ({never}/16)",
            f"{v.strategies_to_95pct_hv.median():.0f}"]) + r" \\")
    (OUT / "cost.tex").write_text("\n".join(lines) + "\n")


for fn in (compromise, hv, posthoc, cost):
    fn()
for f in sorted(OUT.glob("*.tex")):
    print("==", f.name); print(f.read_text())
