"""Repeatability of the target LLMs, measured for free: methods have isolated caches, so the
same (split, sample, exact prompt) is sometimes sent independently by several methods.
For each such group, compare scores and raw outputs of the independent calls.
Writes $PP_PAPER_DIR/numbers/repeatability.txt."""
from paths import ROOT, RESULTS, PAPER, FIG, TAB, NUM  # noqa: E402,F401  (chdir to repo root)
import glob, json, collections, itertools
from pathlib import Path

out = []
for mdir in sorted(glob.glob(f"{RESULTS}/*/")):
    model = mdir.split("/")[-2]
    pairs = same_score = same_out = 0
    groups_n = 0
    for f in sorted(glob.glob(mdir + "*_nsga2_log/evaluations.jsonl")):
        if "/transfer_" in f:
            continue
        g = collections.defaultdict(list)
        for line in open(f):
            e = json.loads(line)
            g[(e["split"], e["sample_id"], e["prompt_sha"])].append((e["method"], e["score"], e["output"]))
        for k, v in g.items():
            if len({m for m, _, _ in v}) < 2:
                continue
            groups_n += 1
            for (m1, s1, o1), (m2, s2, o2) in itertools.combinations(v, 2):
                if m1 == m2:
                    continue
                pairs += 1; same_score += s1 == s2; same_out += o1 == o2
    out.append(f"{model:26s} groups {groups_n:6d} pairs {pairs:7d} same score {same_score/pairs:.3f} identical output {same_out/pairs:.3f}")
(NUM / "repeatability.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
