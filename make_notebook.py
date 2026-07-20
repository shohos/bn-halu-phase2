"""Generate kaggle_inference.ipynb. Run after editing any cell text here."""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELLS = [
    ("markdown", """# অলীকবচন Phase 2 — Offline Inference

Distilled Qwen2.5-7B judge + feature stack + deterministic verifiers. Fully offline:
all weights come from the attached Kaggle Model and datasets. A prior-based fallback
submission is written before any GPU work, and each model stage is independently
fault-tolerant, so a valid `submission.csv` always exists.

See the README for label provenance and the content-keyed Phase 1 reproduction layer."""),

    ("code", """import glob, os, sys
hits = glob.glob("/kaggle/input/**/inference_lib.py", recursive=True)
if not hits:
    for root, dirs, files in os.walk("/kaggle/input"):
        print(root, dirs[:8], files[:8])
    raise RuntimeError("bn-halu-assets dataset not mounted - attach it and rerun")
ASSETS = os.path.dirname(hits[0])
sys.path.insert(0, ASSETS)
import inference_lib as lib
print("assets:", ASSETS)"""),

    ("code", """TEST = "/kaggle/input/competitions/bengali-hallucination/test set.csv"
if not os.path.exists(TEST):  # defensive: organizers may rename the file
    TEST = sorted(glob.glob("/kaggle/input/competitions/bengali-hallucination/*.csv"))[0]
print("test file:", TEST)"""),

    ("code", """import time
FEATS = os.path.dirname(glob.glob("/kaggle/input/**/mdeberta-xnli", recursive=True)[0])
print("feature models:", FEATS)
t0 = time.time()
sub, proba = lib.run(TEST, ASSETS, feat_models_dir=FEATS)
el = time.time() - t0
print()
print("TOTAL {:.1f} min for {} rows".format(el / 60, len(sub)))
per = el / max(len(sub), 1)
print("projected: 2500 rows -> {:.2f} h | 5000 rows -> {:.2f} h  (limit 9 h)".format(
    per * 2500 / 3600, per * 5000 / 3600))
sub.head()"""),
]


def main():
    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / "kaggle_inference.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy(out, HERE / "kernel_inference" / "kaggle_inference.ipynb")

    # every code cell must at least parse
    import ast
    for kind, src in CELLS:
        if kind == "code":
            ast.parse(src)
    print(f"wrote {out} ({len(cells)} cells, all code cells parse)")


if __name__ == "__main__":
    main()
