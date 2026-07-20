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
problems = []
if bad:
    problems.append(f"unbalanced environments: {bad}")
if t.count("{") != t.count("}"):
    problems.append("unbalanced braces")
if re.search(r"[ঀ-৿]", t):
    problems.append("bengali codepoints would break pdflatex")
if "\t" in t:
    problems.append("literal tab characters (a lost backslash, e.g. \\texttt)")
for pat in ["TEAM NAME", "TBD", "⟨"]:
    if t.count(pat):
        problems.append(f"placeholder {pat!r} x{t.count(pat)}")
if t.count(r"\bibitem") and not re.search(r"\\cite[a-z]*\{", t):
    problems.append("bibliography present but no in-text citations")

if problems:
    print("\nFAILED:")
    for x in problems:
        print("  -", x)
    print("\nNOTE: this is a structural check only. It does NOT compile the document;"
          "\n      run pdflatex with the ACL style before submitting.")
    raise SystemExit(1)
print("\nstructural checks passed (still not a compile - run pdflatex)")
