"""Shared paths for the paper scripts.

Every script in ``scripts/paper/`` imports this module first: it moves to the
repository root (so the scripts can be run from any directory), puts ``src/`` on
the import path, and defines where inputs are read and outputs written.

Environment variables (all optional):
  PP_RESULTS_DIR  result files and run logs   (default: results/icse_26_final)
  PP_PAPER_DIR    LaTeX sources of the paper  (default: paper/prompt_piper_paper)

Outputs: figures and CSVs in ``$PP_PAPER_DIR/figures``, LaTeX tables in
``$PP_PAPER_DIR/tables``, the numbers quoted in the text in ``$PP_PAPER_DIR/numbers``.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

RESULTS = Path(os.environ.get("PP_RESULTS_DIR", ROOT / "results" / "icse_26_final")).resolve()
PAPER = Path(os.environ.get("PP_PAPER_DIR", ROOT / "paper" / "prompt_piper_paper")).resolve()
FIG = PAPER / "figures"
TAB = PAPER / "tables"
NUM = PAPER / "numbers"
for _d in (FIG, TAB, NUM):
    _d.mkdir(parents=True, exist_ok=True)
