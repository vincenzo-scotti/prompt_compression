# Paper analysis (TSE)

Every figure, table and number of the paper is computed here from the complete run logs
(`results/icse_26_final/`). Run everything, in order, with

    python scripts/paper/run_all.py

or open `notebooks/paper.ipynb` (same steps, grouped by research question). Inputs and outputs
are set in `paths.py` (`PP_RESULTS_DIR`, `PP_PAPER_DIR`); outputs go to
`$PP_PAPER_DIR/figures` (PDF/CSV), `$PP_PAPER_DIR/tables` (LaTeX) and `$PP_PAPER_DIR/numbers`.

| Script | Content |
|---|---|
| `make_figures.py` | RQ1 fronts; RQ2 compromises, hypervolume, Friedman/Wilcoxon, LLMLingua-2 at equal compression; RQ3 cost, timing, convergence; RQ4 effective operators, transfer |
| `cost_extended.py` | RQ3 break-even including output tokens |
| `make_tables.py` | LaTeX tables of the paper |
| `paper_numbers.py`, `extra_numbers.py`, `repeatability.py`, `generalization.py` | numbers quoted in the text |
| `check_compromise.py`, `check_rq2.py` | independent re-implementations (selection, statistics, hypervolume) |
| `audit_numbers.py` | every number in `main.tex` traced to an output |
