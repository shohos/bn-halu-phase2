"""Fail-closed static checks and optional real LaTeX compilation."""
import argparse
import re
import subprocess
from pathlib import Path


PLACEHOLDERS = [r"<[^>]+>", r"⟨[^⟩]+⟩", r"\bTODO\b", r"\bTBD\b",
                r"\bREPLACE[-_ ]?ME\b", r"\bTEAM NAME\b"]


def strip_comments(text):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def balanced_braces(text):
    depth = 0
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def static_checks(path: Path):
    raw = path.read_text(encoding="utf-8")
    text = strip_comments(raw)
    errors = []
    if "\t" in raw:
        errors.append("literal tab character")
    if re.search(r"[\u0980-\u09ff]", raw):
        errors.append("raw Bengali codepoint (use English/transliteration in ACL source)")
    for pattern in PLACEHOLDERS:
        if re.search(pattern, text, re.I):
            errors.append(f"placeholder matching {pattern}")
    if not balanced_braces(text):
        errors.append("unbalanced braces")
    begins = re.findall(r"\\begin\{([^}]+)\}", text)
    ends = re.findall(r"\\end\{([^}]+)\}", text)
    if sorted(begins) != sorted(ends):
        errors.append("unbalanced LaTeX environments")
    if "\\usepackage{acl}" not in text:
        errors.append("ACL style package missing")
    if re.search(r"\\usepackage(?:\[[^]]*\])?\{natbib\}", text):
        errors.append("natbib loaded twice; acl.sty already loads it")
    cites = set(re.findall(r"\\cite[pt]?\{([^}]+)\}", text))
    cite_keys = {key.strip() for group in cites for key in group.split(",")}
    bib_keys = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    if not cite_keys:
        errors.append("no in-text citations")
    if cite_keys - bib_keys:
        errors.append(f"undefined citations: {sorted(cite_keys - bib_keys)}")
    if bib_keys - cite_keys:
        errors.append(f"uncited bibliography entries: {sorted(bib_keys - cite_keys)}")
    banned_claims = ["at least 99%", "never trained on those rows", "Rule 4.b"]
    for claim in banned_claims:
        if claim.lower() in text.lower():
            errors.append(f"obsolete/unsupported claim: {claim}")
    if errors:
        raise SystemExit("paper checks failed:\n- " + "\n- ".join(errors))
    print(f"STAGE:PAPER_STATIC_OK {path}")


def compile_paper(path: Path):
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
               "-file-line-error", path.name]
    result = subprocess.run(command, cwd=path.parent, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        print(result.stdout)
        raise SystemExit(f"LaTeX compilation failed with exit {result.returncode}")
    pdf = path.with_suffix(".pdf")
    if not pdf.is_file() or pdf.stat().st_size < 1000:
        raise SystemExit("LaTeX reported success but no usable PDF exists")
    print(f"STAGE:PAPER_COMPILE_OK {pdf} ({pdf.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tex", type=Path, nargs="?", default=Path("paper/main.tex"))
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    path = args.tex.resolve()
    static_checks(path)
    if args.compile:
        compile_paper(path)
    else:
        print("NOTE: static checks only; pass --compile for a real LaTeX build")


if __name__ == "__main__":
    main()
