"""Extra setup numbers for the paper: output-cap hits, provider vs tokenizer token counts,
reasoning tokens.  Writes $PP_PAPER_DIR/numbers/extra_numbers.txt."""
from paths import ROOT, RESULTS, PAPER, FIG, TAB, NUM  # noqa: E402,F401  (chdir to repo root)
import glob, json, collections
from pathlib import Path
import numpy as np

CAP = 2048
out = []
cap = collections.defaultdict(lambda: [0, 0])          # (model, method) -> [hits, n]
ratio = collections.defaultdict(list)                   # model -> provider/tokenizer
diff = collections.defaultdict(list)
reason = collections.defaultdict(lambda: [0, 0])        # model -> [evals with reasoning tokens>0, n]
for f in sorted(glob.glob(f"{RESULTS}/*/*_nsga2_log/evaluations.jsonl")):
    if "/transfer_" in f:
        continue
    model = Path(f).parts[-3]
    for line in open(f):
        e = json.loads(line)
        u = e.get("provider_usage") or {}
        ot = u.get("completion_tokens")
        if ot is not None:
            c = cap[(model, e["method"])]
            c[0] += ot >= CAP; c[1] += 1
        pt = u.get("prompt_tokens")
        if pt and e.get("input_tokens"):
            ratio[model].append(pt / e["input_tokens"]); diff[model].append(pt - e["input_tokens"])
        r = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or u.get("reasoning_tokens") or 0
        rr = reason[model]; rr[0] += (r or 0) > 0 or bool(e.get("provider_reasoning")); rr[1] += 1
out.append("== Output-cap hits (completion tokens >= 2048), all evaluations, per model x method")
tot = collections.defaultdict(lambda: [0, 0])
for (m, meth), (h, n) in sorted(cap.items()):
    out.append(f"{m:26s} {meth:20s} {h:6d}/{n:6d} = {h/n:.3%}")
    tot[meth][0] += h; tot[meth][1] += n
out.append("-- per method, all models")
for meth, (h, n) in sorted(tot.items()):
    out.append(f"{meth:20s} {h:6d}/{n:6d} = {h/n:.3%}")
allh = sum(v[0] for v in tot.values()); alln = sum(v[1] for v in tot.values())
out.append(f"overall {allh}/{alln} = {allh/alln:.3%}")
out.append("== Provider prompt tokens vs objective tokenizer")
for m in sorted(ratio):
    a = np.array(ratio[m]); d = np.array(diff[m])
    out.append(f"{m:26s} ratio median {np.median(a):.3f} (q1 {np.quantile(a,.25):.3f}, q3 {np.quantile(a,.75):.3f}); diff median {np.median(d):.0f}")
out.append("== Evaluations with reasoning tokens/text")
for m, (h, n) in sorted(reason.items()):
    out.append(f"{m:26s} {h}/{n}")
(NUM / "extra_numbers.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
