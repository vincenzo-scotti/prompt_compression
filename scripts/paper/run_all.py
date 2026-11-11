"""Regenerate every figure, table and number of the paper from the run logs, then check them.

    python scripts/paper/run_all.py            # from any directory

Order matters: make_figures.py writes the CSVs that the later scripts read.
See ``paths.py`` for the input/output locations (env PP_RESULTS_DIR, PP_PAPER_DIR).
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    # (script, what it produces)
    ("make_figures.py", "RQ1-RQ4 figures and CSVs (fronts, compromises, hypervolume, statistics, cost, operators, transfer)"),
    ("bootstrap.py", "RQ2 paired bootstrap of per-cell performance differences"),
    ("cost_extended.py", "RQ3 break-even including output tokens"),
    ("make_tables.py", "LaTeX tables (compromise, post-hoc, hypervolume, cost)"),
    ("paper_numbers.py", "numbers quoted in the text"),
    ("extra_numbers.py", "setup numbers: output-cap hits, provider vs tokenizer tokens"),
    ("repeatability.py", "repeatability of identical LLM calls (threats)"),
    ("generalization.py", "search-to-test shift of the best compromises (RQ1, threats)"),
    ("check_compromise.py", "check: independent re-implementation of the compromise selection"),
    ("check_rq2.py", "check: independent recomputation of the RQ2 statistics and hypervolumes"),
    ("audit_numbers.py", "check: every number in main.tex traced to an output (needs the paper sources)"),
]

if __name__ == "__main__":
    for script, what in STEPS:
        print(f"== {script}: {what}", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], check=True)
