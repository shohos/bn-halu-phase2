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

    ("code", """# STAGE 0 - unconditional fallback. Nothing above this line may fail, so it does
# no discovery and imports nothing but pandas. A valid submission.csv exists from
# this point on, whatever happens later.
import pandas as pd

TEST = "/kaggle/input/competitions/bengali-hallucination/test set.csv"
OUT = "/kaggle/working/submission.csv"

raw = pd.read_csv(TEST)
ids = raw["id"] if "id" in raw.columns else range(1, len(raw) + 1)
pd.DataFrame({"id": ids, "label": 1}).to_csv(OUT, index=False)
print("stage 0: baseline submission written for", len(raw), "rows")"""),

    ("code", """# STAGE 1 - everything else, inside one exception boundary. Any failure leaves
# the stage-0 file (or whatever the pipeline last wrote) in place and lets the
# notebook run to completion, which is what the rules require.
import glob, os, sys, time, traceback

try:
    hits = glob.glob("/kaggle/input/**/inference_lib.py", recursive=True)
    if not hits:
        for root, dirs, files in os.walk("/kaggle/input"):
            print(root, dirs[:8], files[:8])
        raise RuntimeError("bn-halu-assets not mounted")
    ASSETS = os.path.dirname(hits[0])
    sys.path.insert(0, ASSETS)
    import inference_lib as lib
    print("assets:", ASSETS)

    feat_hits = glob.glob("/kaggle/input/**/mdeberta-xnli", recursive=True)
    FEATS = os.path.dirname(feat_hits[0]) if feat_hits else None
    print("feature models:", FEATS)

    t0 = time.time()
    sub, proba = lib.run(TEST, ASSETS, feat_models_dir=FEATS, out_path=OUT)
    el = time.time() - t0
    per = el / max(len(sub), 1)
    print()
    print("TOTAL {:.1f} min for {} rows".format(el / 60, len(sub)))
    print("projected: 2500 rows -> {:.2f} h | 5000 rows -> {:.2f} h  (limit 9 h)".format(
        per * 2500 / 3600, per * 5000 / 3600))
except Exception:
    traceback.print_exc()
    print("\\nPIPELINE FAILED - keeping the last valid submission.csv")"""),

    ("code", """# STAGE 2 - assert the artefact the organizers will score is well formed.
final = pd.read_csv(OUT)
assert len(final) == len(raw), (len(final), len(raw))
assert list(final.columns) == ["id", "label"], final.columns.tolist()
assert final["label"].isin([0, 1]).all() and not final.isna().any().any()
assert final["id"].is_unique
print("submission.csv OK:", len(final), "rows,", final["label"].value_counts().to_dict())"""),
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
