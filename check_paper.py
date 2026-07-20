"""Structural sanity check for paper/main.tex (no LaTeX toolchain needed here)."""
import collections
import re
from pathlib import Path

t = Path(__file__).resolve().parent.joinpath("paper/main.tex").read_text(encoding="utf-8")

b = collections.Counter(re.findall(r"\\begin\{([A-Za-z*]+)\}", t))
e = collections.Counter(re.findall(r"\\end\{([A-Za-z*]+)\}", t))
bad = [(k, b[k], e[k]) for k in set(b) | set(e) if b[k] != e[k]]
print("environments:", "balanced" if not bad else f"UNBALANCED {bad}")
print("braces delta:", t.count("{") - t.count("}"))
print("bengali chars (break pdflatex):", len(re.findall(r"[ঀ-৿]", t)))
print("citations:", t.count(r"\bibitem"))
print("chars:", len(t))
for pat in ["TEAM NAME", "TBD", "⟨"]:
    if t.count(pat):
        print(f"  PLACEHOLDER {pat!r}: {t.count(pat)}")
