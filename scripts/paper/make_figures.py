"""Figures and tables for the TSE paper, computed from the final run logs.

Run from the repository root:

    .venv/bin/python scripts/paper/make_figures.py [--only rq1,rq2,...]

Inputs:  $PP_RESULTS_DIR/<model>/  (result JSON + complete run logs)
Outputs: $PP_PAPER_DIR/figures/  (PDF figures, CSV tables)

All numbers come from ``prompt_piper.analysis`` (unit-tested).  Every
multi-solution technique (NSGA-II, Random Search, MAB) is represented by its
best compromise, selected on the *search* set with the same rule (closest to
the ideal point: performance 1, normalised tokens 0) and reported with its
*test-set* values.
"""

from __future__ import annotations

from paths import ROOT, RESULTS, PAPER, FIG, TAB, NUM  # noqa: E402,F401  (chdir to repo root)

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prompt_piper import analysis as A  # noqa: E402


# Display order and labels (rows = models, columns = benchmarks), as in the paper.
MODELS = [
    ("gpt-5.4-nano-2026-03-17", "GPT-5.4-nano"),
    ("gemini-3.8-flash", "Gemini-3.8-Flash"),
    ("gemma4-26b-a4b-it", "Gemma-4-26B"),
    ("qwen3.5-9b", "Qwen3.5-9B"),
]
TASKS = [
    ("mbpp_instruct_nsga2", "MBPP"),
    ("mmlu_pro_cs_nsga2", "MMLU-Pro CS"),
    ("gsm8k_cot_nsga2", "GSM8K"),
    ("bbh_logical_deduction_nsga2", "BBH-Logic"),
]

# Optimizers: categorical palette validated for CVD separation and contrast
# (dataviz validate_palette.js, light surface: all checks pass).
OPTIMIZERS = {
    "nsga2": ("NSGA-II", "#1f6fd1"),
    "random_search": ("Random search", "#d4692b"),
    "multi_armed_bandit": ("MAB", "#7a4fc4"),
}
# Single-point baselines keep the markers and colors of the submitted paper.
BASELINES = {
    "no_compression": ("No compression", "#555555", "D"),
    "all_transformations": ("All transforms", "#e06c75", "s"),
    "summarization": ("Summarization", "#c678dd", "P"),
    "truncated_prompt": ("Truncated", "#61afef", "v"),
    "random_deletion": ("Rand. deletion", "#56b6c2", "X"),
    "llmlingua2": ("LLMLingua-2", "#98c379", "h"),
}

# Figures are drawn at their printed size (IEEE two-column: text width 7.16 in, column
# width 3.5 in), so the font sizes below are the sizes in the paper.
TEXTW, COLW = 7.16, 3.5
plt.rcParams.update({
    "figure.dpi": 120, "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})


def load():
    runs = A.load_runs(RESULTS)
    by = {(r.model, r.name): r for r in runs}
    missing = [(m, t) for m, _ in MODELS for t, _ in TASKS if (m, t) not in by]
    if missing:
        raise SystemExit(f"missing runs: {missing}")
    return runs, by


def _front(sols, split):
    pts = sorted((s[f"{split}_avg_tokens"], s[f"{split}_avg_performance"]) for s in sols)
    return np.array(pts)


def fig_rq1(by):
    """Search (dashed) and test (solid) fronts of the three optimizers, best compromise
    highlighted (selected on S, plotted at its test values), baselines on the test set."""
    fig, axes = plt.subplots(len(MODELS), len(TASKS), figsize=(TEXTW, 6.4),
                             squeeze=False, sharey="all", sharex="col")
    handles = {}
    for ri, (mdir, mlabel) in enumerate(MODELS):
        for ci, (stem, tlabel) in enumerate(TASKS):
            ax, run = axes[ri][ci], by[(mdir, stem)]
            ax.set_ylim(-0.1, 1.1)
            for key, (label, color) in OPTIMIZERS.items():
                sols = run.solutions(key)
                s = _front(sols, "search")
                t = _front(sols, "test")
                h1, = ax.plot(s[:, 0], s[:, 1], "--", color=color, lw=0.7, alpha=0.55,
                              marker="o", ms=1.6, label=f"{label} (search)")
                h2, = ax.plot(t[:, 0], t[:, 1], "-", color=color, lw=1.1,
                              marker="o", ms=2, label=f"{label} (test)")
                _, best = A.select_compromise(run, key, "search")
                h3 = ax.scatter(best["test_avg_tokens"], best["test_avg_performance"], s=60,
                                marker="*", color=color, edgecolors="white", linewidths=0.5,
                                zorder=7, label=f"{label} best compromise")
                handles.update({h1.get_label(): h1, h2.get_label(): h2, h3.get_label(): h3})
            for key, (label, color, marker) in BASELINES.items():
                b = run.baseline(key)
                if b is None:
                    continue
                h = ax.scatter(b["test_avg_tokens"], b["test_avg_performance"], s=26, marker=marker,
                               color=color, edgecolors="white", linewidths=0.4, zorder=6, label=label)
                handles[label] = h
            if ri == 0:
                ax.set_title(tlabel, fontweight="bold")
            if ci == 0:
                ax.set_ylabel("Avg performance")
                ax.annotate(mlabel, xy=(-0.42, 0.5), xycoords="axes fraction", fontsize=8,
                            fontweight="bold", rotation=90, va="center")
            if ri == len(MODELS) - 1:
                ax.set_xlabel("Avg tokens")
    # Legend fills column by column: one column per optimizer (search, test, best
    # compromise), then the six single-point baselines in two columns.
    order = [f"{OPTIMIZERS[k][0]} {suffix}" for k in OPTIMIZERS
             for suffix in ("(search)", "(test)", "best compromise")]
    order += [BASELINES[k][0] for k in BASELINES]
    fig.legend([handles[k] for k in order if k in handles], [k for k in order if k in handles],
               loc="lower center", ncol=5, frameon=False, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=0.6, w_pad=0.8)
    fig.savefig(FIG / "rq1_fronts.pdf", bbox_inches="tight")
    fig.savefig(FIG / "rq1_fronts.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# RQ2 — best compromise, hypervolume, statistics, distributions
# ---------------------------------------------------------------------------

def _wilcoxon(x, y, alternative):
    from scipy.stats import wilcoxon
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.allclose(x, y):
        return 1.0
    return float(wilcoxon(x, y, alternative=alternative).pvalue)


def compromise_points(runs):
    """One row per (model, task, method): best compromise (selected on S), test values."""
    pts = A.method_points(runs, selection="search")
    if pts.legacy_front_mean.any():
        raise SystemExit("legacy results: RS/MAB solutions missing")
    return pts


def hypervolumes(runs):
    rows = []
    for r in runs:
        for key in OPTIMIZERS:
            rows.append({"model": r.model, "task": r.name, "method": key,
                         "hv_search": A.front_hypervolume(r, key, "search"),
                         "hv_test": A.front_hypervolume(r, key, "test"),
                         "front_size": len(r.solutions(key))})
    return pd.DataFrame(rows)


def rq2_statistics(pts, hv):
    """Friedman (all methods) + Holm-corrected Wilcoxon with rank-biserial effect sizes.

    Family A (per optimizer): optimizer vs each single-point baseline — one-sided in
    favour of the optimizer, two-sided against No compression.
    Family B: the three optimizers pairwise, two-sided.  Hypervolume: Family B only.
    """
    from scipy.stats import friedmanchisquare

    omni, post = [], []
    metrics = [("test_performance", True), ("compression", True)]
    for metric, higher in metrics:
        wide = pts.pivot_table(index=["model", "task"], columns="method", values=metric)
        assert not wide.isna().any().any()
        chi2, p = friedmanchisquare(*[wide[c] for c in wide.columns])
        n, k = wide.shape
        omni.append({"metric": metric, "methods": k, "cells": n, "chi2": chi2, "p": p,
                     "kendall_w": chi2 / (n * (k - 1))})
        for opt in OPTIMIZERS:
            fam = []
            for b in BASELINES:
                alt = "two-sided" if b == "no_compression" else ("greater" if higher else "less")
                fam.append((opt, b, alt))
            post += _family(wide, metric, "A:" + opt, fam)
        post += _family(wide, metric, "B", [(a, b, "two-sided") for a, b in
                                             [("nsga2", "random_search"), ("nsga2", "multi_armed_bandit"),
                                              ("random_search", "multi_armed_bandit")]])
    for metric in ("hv_search", "hv_test"):
        wide = hv.pivot_table(index=["model", "task"], columns="method", values=metric)
        chi2, p = friedmanchisquare(*[wide[c] for c in wide.columns])
        n, k = wide.shape
        omni.append({"metric": metric, "methods": k, "cells": n, "chi2": chi2, "p": p,
                     "kendall_w": chi2 / (n * (k - 1))})
        post += _family(wide, metric, "B", [(a, b, "two-sided") for a, b in
                                             [("nsga2", "random_search"), ("nsga2", "multi_armed_bandit"),
                                              ("random_search", "multi_armed_bandit")]])
    return pd.DataFrame(omni), pd.DataFrame(post)


def _family(wide, metric, family, comparisons):
    rows = []
    for a, b, alt in comparisons:
        x, y = wide[a].to_numpy(), wide[b].to_numpy()
        rows.append({"metric": metric, "family": family, "a": a, "b": b, "alternative": alt,
                     "median_a": float(np.median(x)), "median_b": float(np.median(y)),
                     "mean_a": float(np.mean(x)), "mean_b": float(np.mean(y)),
                     "wins_a": int((x > y + 1e-12).sum()), "wins_b": int((y > x + 1e-12).sum()),
                     "rank_biserial": A.rank_biserial(x, y), "p_raw": _wilcoxon(x, y, alt)})
    adj = A.holm([r["p_raw"] for r in rows])
    for r, p in zip(rows, adj):
        r["p_holm"] = float(p)
    return rows


def fig_rq2_distributions(runs, pts):
    """Previous style: per benchmark, per-sample performance (mean, 95% CI) and token
    distribution of each method's point, pooled across models."""
    by = {(r.model, r.name): r for r in runs}
    methods = list(OPTIMIZERS) + list(BASELINES)
    label = {**{k: v[0] for k, v in OPTIMIZERS.items()}, **{k: v[0] for k, v in BASELINES.items()}}
    color = {**{k: v[1] for k, v in OPTIMIZERS.items()}, **{k: v[1] for k, v in BASELINES.items()}}
    per = []
    for _, row in pts.iterrows():
        d = A.per_sample(by[(row.model, row.task)], row.method, int(row.solution_index))
        d["task"], d["method"] = row.task, row.method
        per.append(d)
    per = pd.concat(per, ignore_index=True)
    # Columns = benchmarks; top row performance, bottom row token distribution.
    fig, axes = plt.subplots(2, len(TASKS), figsize=(TEXTW, 3.7), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1.15]})
    for ti, (stem, tlabel) in enumerate(TASKS):
        sub = per[per.task == stem]
        ax_p, ax_t = axes[0][ti], axes[1][ti]
        for mi, m in enumerate(methods):
            v = sub[sub.method == m]
            ci = 1.96 * v.score.std(ddof=1) / np.sqrt(len(v))
            ax_p.bar(mi, v.score.mean(), yerr=ci, width=0.7, color=color[m], alpha=0.85,
                     capsize=1.5, error_kw={"lw": 0.7, "capthick": 0.7})
        parts = ax_t.violinplot([sub[sub.method == m].tokens for m in methods],
                                positions=range(len(methods)), showmedians=True, widths=0.8)
        for body, m in zip(parts["bodies"], methods):
            body.set_facecolor(color[m]); body.set_alpha(0.6)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[key].set_color("#444444"); parts[key].set_linewidth(0.6)
        ax_p.set_ylim(0, 1.05)
        ax_p.set_title(tlabel, fontweight="bold")
        ax_t.set_xticks(range(len(methods)))
        ax_t.set_xticklabels([label[m] for m in methods], rotation=60, ha="right",
                             rotation_mode="anchor")
    axes[0][0].set_ylabel("Performance (95% CI)")
    axes[1][0].set_ylabel("Compressed tokens")
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    fig.savefig(FIG / "rq2_distributions.pdf", bbox_inches="tight")
    fig.savefig(FIG / "rq2_distributions.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    per.to_csv(FIG / "rq2_per_sample.csv", index=False)


def rq2(runs):
    pts = compromise_points(runs)
    pts.to_csv(FIG / "rq2_compromise_points.csv", index=False)
    summary = (pts.groupby("method")[["test_performance", "test_tokens", "compression"]]
               .agg(["mean", "median"]))
    summary.to_csv(FIG / "rq2_compromise_summary.csv")
    hv = hypervolumes(runs)
    hv.to_csv(FIG / "rq2_hypervolume.csv", index=False)
    omni, post = rq2_statistics(pts, hv)
    omni.to_csv(FIG / "rq2_friedman.csv", index=False)
    post.to_csv(FIG / "rq2_posthoc.csv", index=False)
    fig_rq2_distributions(runs, pts)
    print(summary.round(3))
    print(hv.groupby("method")[["hv_search", "hv_test"]].agg(["mean", "median"]).round(3))
    print(omni.round(4))


# ---------------------------------------------------------------------------
# RQ2 — LLMLingua-2 at equal compression
# ---------------------------------------------------------------------------

def matched_llmlingua(runs):
    """For each cell and optimizer: the front solution with at least LLMLingua-2's
    compression on S (tokens normalised by the uncompressed length) and the highest
    S performance among those; reported with test values.  ``None`` when the front
    does not reach LLMLingua-2's compression."""
    rows = []
    for r in runs:
        ll = r.baseline("llmlingua2")
        ref_s, ref_t = r.ref_tokens("search"), r.ref_tokens("test")
        comp_ll_s = 1 - ll["search_avg_tokens"] / ref_s
        row = {"model": r.model, "task": r.name,
               "llmlingua2_compression": 1 - ll["test_avg_tokens"] / ref_t,
               "llmlingua2_performance": ll["test_avg_performance"],
               "llmlingua2_tokens": ll["test_avg_tokens"]}
        for key in OPTIMIZERS:
            feasible = [s for s in r.solutions(key)
                        if 1 - s["search_avg_tokens"] / ref_s >= comp_ll_s]
            if feasible:
                best = max(feasible, key=lambda s: (s["search_avg_performance"], -s["search_avg_tokens"]))
                row[f"{key}_performance"] = best["test_avg_performance"]
                row[f"{key}_compression"] = 1 - best["test_avg_tokens"] / ref_t
                row[f"{key}_tokens"] = best["test_avg_tokens"]
            else:
                row[f"{key}_performance"] = row[f"{key}_compression"] = row[f"{key}_tokens"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fig_llmlingua(df):
    """Dot plot: one row per (model, benchmark); LLMLingua-2 vs each optimizer's best
    strategy at equal or higher compression (test performance)."""
    tl = dict(TASKS); ml = dict(MODELS)
    order = [(m, t) for t, _ in TASKS for m, _ in MODELS]
    df = df.set_index(["model", "task"]).loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(COLW, 0.26 * len(df) + 0.9))
    y = np.arange(len(df))[::-1]
    # Each series on its own sub-row so markers never hide each other (e.g. all at 0).
    ax.scatter(df.llmlingua2_performance, y + 0.27, marker=BASELINES["llmlingua2"][2], s=22,
               color=BASELINES["llmlingua2"][1], edgecolors="white", zorder=5, label="LLMLingua-2")
    offsets = {"nsga2": 0.09, "random_search": -0.09, "multi_armed_bandit": -0.27}
    for key, (label, color) in OPTIMIZERS.items():
        v = df[f"{key}_performance"]
        ax.scatter(v, y + offsets[key], marker="o", s=12, color=color, zorder=6, label=label)
        ax.hlines(y + offsets[key], df.llmlingua2_performance, v, color=color, lw=0.6, alpha=0.5)
    none = df[[f"{k}_performance" for k in OPTIMIZERS]].isna().all(axis=1).to_numpy()
    for yi, flag in zip(y, none):
        if flag:
            ax.text(1.04, yi, "none reaches it", ha="right", va="center",
                    fontsize=6, color="#666666", style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{tl[t]} · {ml[m]}" for m, t in zip(df.model, df.task)], fontsize=6.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlabel("Test performance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.07), ncol=4, frameon=False,
              handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "rq2_llmlingua_matched.pdf", bbox_inches="tight")
    fig.savefig(FIG / "rq2_llmlingua_matched.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


def rq2_llmlingua(runs):
    df = matched_llmlingua(runs)
    df.to_csv(FIG / "rq2_llmlingua_matched.csv", index=False)
    rows = []
    for key in OPTIMIZERS:
        v = df.dropna(subset=[f"{key}_performance"])
        x, yy = v[f"{key}_performance"].to_numpy(), v.llmlingua2_performance.to_numpy()
        rows.append({"method": key, "cells_reaching_llmlingua_compression": len(v),
                     "cells_total": len(df),
                     "mean_performance": x.mean(), "mean_llmlingua2_performance": yy.mean(),
                     "mean_compression": v[f"{key}_compression"].mean(),
                     "mean_llmlingua2_compression": v.llmlingua2_compression.mean(),
                     "wins": int((x > yy).sum()), "ties": int((x == yy).sum()),
                     "p_wilcoxon_greater": _wilcoxon(x, yy, "greater"),
                     "rank_biserial": A.rank_biserial(x, yy)})
    summ = pd.DataFrame(rows)
    summ["p_holm"] = A.holm(summ.p_wilcoxon_greater)
    summ.to_csv(FIG / "rq2_llmlingua_matched_summary.csv", index=False)
    fig_llmlingua(df)
    print(summ.round(4).to_string())
    print("cells where a front does not reach LLMLingua-2's compression:")
    for key in OPTIMIZERS:
        miss = df[df[f"{key}_performance"].isna()][["model", "task"]].values.tolist()
        print(f"  {key}: {miss}")


# ---------------------------------------------------------------------------
# RQ3 — cost and convergence
# ---------------------------------------------------------------------------

ALL_METHODS = list(OPTIMIZERS) + list(BASELINES)


def cost_table(runs, pts):
    """One row per (run, method): model calls, tokens, wall-clock time, break-even.

    Calls and tokens count the evaluations the method itself paid for (each method
    has its own cache).  Break-even: search-phase input tokens divided by the input
    tokens the method's point saves per request on the test set.
    """
    rows = []
    for r in runs:
        evals = r.stream("evaluations")
        phases = [p for p in r.stream("phases") if p["event"] == "end"]
        nc_tokens = r.ref_tokens("test")
        for m in ALL_METHODS:
            e_s = [e for e in evals if e.get("method") == m and e["split"] == "search"]
            e_t = [e for e in evals if e.get("method") == m and e["split"] == "test"]
            wall = {sp: sum(p.get("wall_elapsed_seconds", p["elapsed_seconds"]) for p in phases
                            if p.get("method") == m and p["phase"] in names)
                    for sp, names in (("search", {"search", "baseline_search"}),
                                      ("test", {"test", "baseline_test"}))}
            point = pts[(pts.model == r.model) & (pts.task == r.name) & (pts.method == m)].iloc[0]
            saved = nc_tokens - point.test_tokens
            in_s = sum(e["input_tokens"] for e in e_s)
            rows.append({
                "model": r.model, "task": r.name, "method": m,
                "search_calls": len(e_s), "test_calls": len(e_t),
                "search_input_tokens": in_s,
                "search_output_tokens": sum(e.get("output_tokens", 0) for e in e_s),
                "provider_prompt_tokens": sum((e.get("provider_usage") or {}).get("prompt_tokens", 0)
                                              for e in e_s + e_t),
                "provider_completion_tokens": sum((e.get("provider_usage") or {}).get("completion_tokens", 0)
                                                  for e in e_s + e_t),
                "search_hours": wall["search"] / 3600, "test_hours": wall["test"] / 3600,
                "saved_tokens_per_request": saved,
                "break_even_requests": in_s / saved if saved > 0 and m in OPTIMIZERS else np.nan,
            })
    return pd.DataFrame(rows)


def fig_rq3_timing(cost):
    """Previous style: running time (search + test) per approach, per benchmark (log scale)."""
    fig, axes = plt.subplots(1, len(TASKS), figsize=(TEXTW, 2.3), sharey=True, squeeze=False)
    label = {**{k: v[0] for k, v in OPTIMIZERS.items()}, **{k: v[0] for k, v in BASELINES.items()}}
    color = {**{k: v[1] for k, v in OPTIMIZERS.items()}, **{k: v[1] for k, v in BASELINES.items()}}
    for ci, (stem, tlabel) in enumerate(TASKS):
        ax = axes[0][ci]
        sub = cost[cost.task == stem]
        data = [(sub[sub.method == m].search_hours + sub[sub.method == m].test_hours) * 60
                for m in ALL_METHODS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops={"color": "#222222", "lw": 0.8}, boxprops={"lw": 0.6},
                        whiskerprops={"lw": 0.6}, capprops={"lw": 0.6},
                        flierprops={"ms": 2.5, "mew": 0.5})
        for patch, m in zip(bp["boxes"], ALL_METHODS):
            patch.set_facecolor(color[m]); patch.set_alpha(0.75)
        ax.set_yscale("log")
        ax.set_title(tlabel, fontweight="bold")
        ax.set_xticks(range(1, len(ALL_METHODS) + 1))
        ax.set_xticklabels([label[m] for m in ALL_METHODS], rotation=60, ha="right",
                           rotation_mode="anchor")
    axes[0][0].set_ylabel("Minutes (search + test)")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "rq3_timing.pdf", bbox_inches="tight")
    fig.savefig(FIG / "rq3_timing.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


def fig_rq3_convergence(conv):
    """Search-set hypervolume of the current front vs number of evaluated strategies,
    mean ± std across models, per benchmark."""
    fig, axes = plt.subplots(1, len(TASKS), figsize=(TEXTW, 1.9), sharey=True, squeeze=False)
    grid = np.arange(1, 301)
    for ci, (stem, tlabel) in enumerate(TASKS):
        ax = axes[0][ci]
        for key, (label, color) in OPTIMIZERS.items():
            curves = []
            for _, g in conv[(conv.task == stem) & (conv.method == key)].groupby("model"):
                g = g.sort_values("n_visited")
                # step function: the front after each evaluated strategy
                curves.append(np.interp(grid, g.n_visited, g.hypervolume, left=np.nan))
            c = np.array(curves)
            mean, std = np.nanmean(c, axis=0), np.nanstd(c, axis=0)
            ax.plot(grid, mean, color=color, lw=1.1, label=label)
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15, lw=0)
        ax.set_title(tlabel, fontweight="bold")
        ax.set_xlabel("Evaluated strategies")
    axes[0][0].set_ylabel("Hypervolume (search set)")
    axes[0][0].legend(frameon=False, loc="upper right")  # empty region (MBPP curves are low)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "rq3_convergence.pdf", bbox_inches="tight")
    fig.savefig(FIG / "rq3_convergence.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


def rq3(runs):
    pts = compromise_points(runs)
    cost = cost_table(runs, pts)
    cost.to_csv(FIG / "rq3_cost.csv", index=False)
    conv = []
    for r in runs:
        c = A.convergence(r)
        c["model"], c["task"] = r.model, r.name
        conv.append(c)
    conv = pd.concat(conv, ignore_index=True)
    conv.to_csv(FIG / "rq3_convergence.csv", index=False)
    # evaluated strategies needed to reach 95% of the final search hypervolume
    final = conv.groupby(["model", "task", "method"]).hypervolume.transform("max")
    reach = (conv[conv.hypervolume >= 0.95 * final].groupby(["model", "task", "method"])
             .n_visited.min().rename("strategies_to_95pct_hv").reset_index())
    reach.to_csv(FIG / "rq3_convergence_95.csv", index=False)
    summ = cost[cost.method.isin(OPTIMIZERS)].groupby("method").agg(
        search_calls=("search_calls", "mean"), search_hours=("search_hours", "mean"),
        search_hours_min=("search_hours", "min"), search_hours_max=("search_hours", "max"),
        search_input_Mtokens=("search_input_tokens", lambda v: v.mean() / 1e6),
        saved_tokens=("saved_tokens_per_request", "mean"),
        break_even_median=("break_even_requests", "median"),
        break_even_q1=("break_even_requests", lambda v: v.quantile(0.25)),
        break_even_q3=("break_even_requests", lambda v: v.quantile(0.75)))
    summ = summ.join(reach.groupby("method").strategies_to_95pct_hv.median())
    summ.to_csv(FIG / "rq3_cost_summary.csv")
    fig_rq3_timing(cost)
    fig_rq3_convergence(conv)
    print(summ.round(2).to_string())


# ---------------------------------------------------------------------------
# RQ4 — operators and transfer
# ---------------------------------------------------------------------------

# Prompt-section palette (validated: all checks pass on the light surface).
SECTIONS = {"I": ("Task instructions", "#c2364a"), "S": ("Few-shot examples", "#1b93a8"),
            "D": ("Data", "#8a9a1f")}


def operator_usage(runs):
    """Share of Pareto solutions (all three optimizers pooled) in which each operator is
    *effective* in each section, i.e. its removal changes at least one search prompt.
    Also reports the share in which it is merely set (effective or neutral)."""
    rows = []
    for r in runs:
        for key in OPTIMIZERS:
            sols = r.solutions(key)
            for s in sols:
                eff = s.get("effective_transformation_names")
                if eff is None:
                    raise SystemExit(f"{r.name}/{key}: effective operators not recorded")
                for name in s["transformation_names"]:
                    sec, _, op = name.partition(":")
                    rows.append({"model": r.model, "task": r.name, "method": key, "section": sec,
                                 "operator": op, "effective": name in eff, "n_solutions": len(sols)})
    df = pd.DataFrame(rows)
    # frequency within each (cell, method) front, then averaged
    g = df.groupby(["model", "task", "method", "section", "operator"]).agg(
        set_count=("effective", "size"), effective_count=("effective", "sum"),
        n=("n_solutions", "first")).reset_index()
    g["set_share"] = g.set_count / g.n
    g["effective_share"] = g.effective_count / g.n
    return g


def fig_rq4_operators(g, op_order):
    """Previous style: horizontal grouped bars per operator, one bar per prompt section."""
    cells = g[["model", "task", "method"]].drop_duplicates()
    n_fronts = len(cells)
    mean = (g.groupby(["section", "operator"]).effective_share.sum() / n_fronts).unstack(0).reindex(op_order).fillna(0)
    fig, ax = plt.subplots(figsize=(COLW, 4.6))
    y = np.arange(len(op_order))[::-1]
    h = 0.27
    for i, (sec, (label, color)) in enumerate(SECTIONS.items()):
        ax.barh(y + (1 - i) * h, mean.get(sec, pd.Series(0, index=op_order)), height=h,
                color=color, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(op_order)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlabel("Share of front strategies where effective")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False,
              handlelength=1.2, handletextpad=0.4, columnspacing=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "rq4_operators.pdf", bbox_inches="tight")
    fig.savefig(FIG / "rq4_operators.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return mean


TRANSFER = [("gsm8k_cot_nsga2", "bbh_logical_deduction_nsga2", "GSM8K→BBH"),
            ("bbh_logical_deduction_nsga2", "gsm8k_cot_nsga2", "BBH→GSM8K")]


def transfer_table(by):
    rows = []
    for mdir, mlabel in MODELS:
        for src, tgt, label in TRANSFER:
            native = by[(mdir, tgt)]
            tr = A.load_transfers(RESULTS)
            tr = tr[(tr.model == mdir) & (tr.source == src) & (tr.target == tgt)]
            nc = native.baseline("no_compression")
            for kind, perfs, toks in (
                ("native", [s["test_avg_performance"] for s in native.solutions("nsga2")],
                 [s["test_avg_tokens"] for s in native.solutions("nsga2")]),
                ("transferred", tr.test_performance.tolist(), tr.test_tokens.tolist())):
                for p, t in zip(perfs, toks):
                    rows.append({"model": mdir, "direction": label, "kind": kind,
                                 "performance": p, "tokens": t,
                                 "nc_performance": nc["test_avg_performance"],
                                 "nc_tokens": nc["test_avg_tokens"]})
    return pd.DataFrame(rows)


def fig_rq4_transfer(df):
    """Rows = transfer directions; per model a performance bar chart (95% CI over front
    solutions) and a token violin plot; dashed line = no compression."""
    colors = {"native": OPTIMIZERS["nsga2"][1], "transferred": "#d9a21b"}
    fig, axes = plt.subplots(len(TRANSFER), 2 * len(MODELS), figsize=(TEXTW, 2.9), squeeze=False)
    for di, (_, _, label) in enumerate(TRANSFER):
        for mi, (mdir, mlabel) in enumerate(MODELS):
            sub = df[(df.model == mdir) & (df.direction == label)]
            ax_p, ax_t = axes[di][2 * mi], axes[di][2 * mi + 1]
            for xi, kind in enumerate(("native", "transferred")):
                v = sub[sub.kind == kind]
                ci = 1.96 * v.performance.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
                ax_p.bar(xi, v.performance.mean(), yerr=ci, color=colors[kind], width=0.6,
                         capsize=1.5, alpha=0.85, error_kw={"lw": 0.7, "capthick": 0.7})
                parts = ax_t.violinplot([v.tokens], positions=[xi], showmedians=True, widths=0.7)
                parts["bodies"][0].set_facecolor(colors[kind]); parts["bodies"][0].set_alpha(0.6)
                for k in ("cmedians", "cmins", "cmaxes", "cbars"):
                    parts[k].set_color("#444444"); parts[k].set_linewidth(0.6)
            ax_p.axhline(sub.nc_performance.iloc[0], ls="--", color="#555555", lw=0.7)
            ax_t.axhline(sub.nc_tokens.iloc[0], ls="--", color="#555555", lw=0.7)
            ax_p.set_ylim(0, 1.05)
            for ax in (ax_p, ax_t):
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Native", "Transf."] if di == len(TRANSFER) - 1 else ["", ""],
                                   rotation=45, ha="right", rotation_mode="anchor")
                ax.set_xlim(-0.6, 1.6)
            if di == 0:
                ax_p.set_title("Performance", fontsize=7)
                ax_t.set_title("Tokens", fontsize=7)
        axes[di][0].set_ylabel(label, fontweight="bold")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend([Patch(color=colors["native"]), Patch(color=colors["transferred"]),
                Line2D([], [], ls="--", color="#555555", lw=0.7)],
               ["Native NSGA-II front", "Transferred front", "No compression"],
               loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), h_pad=0.6, w_pad=0.3)
    # One bold model title centered over each performance/tokens pair.
    for mi, (_, mlabel) in enumerate(MODELS):
        left, right = axes[0][2 * mi].get_position(), axes[0][2 * mi + 1].get_position()
        fig.text((left.x0 + right.x1) / 2, 0.975, mlabel, ha="center", va="center",
                 fontsize=8, fontweight="bold")
    fig.savefig(FIG / "rq4_transfer.pdf", bbox_inches="tight")
    fig.savefig(FIG / "rq4_transfer.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


def rq4(runs, by):
    from prompt_piper.transformations import ALL_TRANSFORMATIONS
    op_order = [T.name for T in ALL_TRANSFORMATIONS if int(T.id[1:]) <= 22]
    g = operator_usage(runs)
    g.to_csv(FIG / "rq4_operator_usage.csv", index=False)
    mean = fig_rq4_operators(g, op_order)
    mean.to_csv(FIG / "rq4_operator_effective_share.csv")
    cells = len(g[["model", "task", "method"]].drop_duplicates())
    neutral = 1 - g.effective_count.sum() / g.set_count.sum()
    print(f"fronts: {cells}; share of set operator bits that are neutral: {neutral:.3f}")
    print(mean.round(2).sort_values("I", ascending=False).head(8).to_string())
    tr = transfer_table(by)
    tr.to_csv(FIG / "rq4_transfer.csv", index=False)
    fig_rq4_transfer(tr)
    print(tr.groupby(["direction", "kind"])[["performance", "tokens"]].mean().round(3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    only = set(filter(None, args.only.split(",")))
    FIG.mkdir(parents=True, exist_ok=True)
    runs, by = load()
    for name, fn in [("rq1", lambda: fig_rq1(by)), ("rq2", lambda: rq2(runs)),
                     ("llmlingua", lambda: rq2_llmlingua(runs)),
                     ("rq3", lambda: rq3(runs)), ("rq4", lambda: rq4(runs, by))]:
        if not only or name in only:
            fn()
            print(f"{name}: done")


if __name__ == "__main__":
    main()
