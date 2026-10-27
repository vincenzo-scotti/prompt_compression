"""Number audit: every decimal number in main.tex (outside the generated tables) must match,
after rounding to the same number of decimals, a number in an analysis output or CSV."""
from paths import ROOT, RESULTS, PAPER, FIG, TAB, NUM  # noqa: E402,F401  (chdir to repo root)
import re, glob
from pathlib import Path
src = ""
for f in glob.glob(f"{NUM}/*.txt") + glob.glob(f"{FIG}/rq*.csv") \
        + glob.glob(f"{TAB}/*.tex"):
    src += Path(f).read_text() + "\n"
pool = [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", src)]
tex = (PAPER / "main.tex").read_text()
body = tex[tex.index(r"\begin{abstract}"):tex.index(r"\bibliography{")]
body = re.sub(r"\\cite\{[^}]*\}|\\label\{[^}]*\}|\\ref\{[^}]*\}|\\Cref\{[^}]*\}", "", body)
missing = set()
for m in re.finditer(r"(?<![\w.])(\d+\.\d+)(?![\w])", body):
    v = m.group(1); d = len(v.split(".")[1]); x = float(v)
    ok = any(round(p, d) == x or round(p * 100, d) == x or round(p / 1e3, d) == x or round(p/3600, d) == x for p in pool)
    if not ok:
        missing.add(v)
for m in re.finditer(r"\$(\d+)\\%\$|(\d+)\\%", body):
    v = m.group(1) or m.group(2); x = int(v)
    if not any(round(p * 100) == x for p in pool if 0 <= p <= 1):
        missing.add(v + "%")
print("unmatched:", sorted(missing, key=lambda s: float(s.rstrip('%'))))
