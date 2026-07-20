"""Honest local evaluation of the Phase 2 ensemble on the 299 organizer labels.

Stack trains on the 1,608 pseudo-labeled rows only; the 299 are never trained on
by either component (the judge held them out too), so blend weight and thresholds
calibrated here are honest.

Run:  python eval_stack.py            (caches features to data/feats_corpus.parquet)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features_lib as fl  # noqa: E402
from inference_lib import f1_class0, flat_threshold  # noqa: E402

hub = Path.home() / ".cache/huggingface/hub"
snap = lambda r: sorted((hub / r / "snapshots").iterdir())[-1]
MD = snap("models--MoritzLaurer--mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
XL = snap("models--deepset--xlm-roberta-base-squad2")
E5 = snap("models--intfloat--multilingual-e5-base")

CACHE = HERE / "data" / "feats_corpus.parquet"


def main():
    corpus = pd.read_parquet(HERE / "data" / "corpus.parquet")
    if CACHE.exists():
        feats = pd.read_parquet(CACHE)
        print(f"loaded cached features {feats.shape}")
    else:
        t = time.time()
        feats = fl.extract_all(corpus, MD, XL, E5)
        feats.to_parquet(CACHE)
        print(f"extracted features {feats.shape} in {time.time() - t:.0f}s")

    tr = (corpus["split"] == "train").values
    y_ho = corpus.loc[~tr, "label"].values
    ctx_ho = corpus.loc[~tr, "has_ctx"].values
    ids_ho = corpus.loc[~tr, "id"].values

    _, ho_stack = fl.stack_fit_predict(
        feats[tr], corpus.loc[tr, "label"].values, corpus.loc[tr, "has_ctx"].values,
        feats[~tr], feats[~tr])

    jp = json.loads((HERE / "upload_bn_halu_assets" / "holdout_probs.json").read_text())
    ho_judge = np.array([(jp[str(i)]["p_bn"] + jp[str(i)]["p_en"]) / 2 for i in ids_ho])

    def score(p, tag):
        tc, _ = flat_threshold(y_ho[ctx_ho], p[ctx_ho])
        tn, _ = flat_threshold(y_ho[~ctx_ho], p[~ctx_ho])
        pred = np.where(ctx_ho, p >= tc, p >= tn).astype(int)
        f1 = f1_class0(y_ho, pred)
        print(f"{tag:28s} overall {f1:.4f} | ctx {f1_class0(y_ho[ctx_ho], pred[ctx_ho]):.4f} "
              f"| noctx {f1_class0(y_ho[~ctx_ho], pred[~ctx_ho]):.4f} "
              f"| th {tc:.2f}/{tn:.2f}")
        return f1, pred

    print()
    score(ho_judge, "judge alone")
    score(ho_stack, "stack alone")
    best = None
    for w in np.arange(0, 1.01, 0.1):
        f1, _ = score(w * ho_judge + (1 - w) * ho_stack, f"blend w_judge={w:.1f}")
        if best is None or f1 > best[0]:
            best = (f1, w)
    print(f"\nBEST BLEND: w_judge={best[1]:.1f} -> F1_0 {best[0]:.4f}")

    # per-qtype for the best blend
    p = best[1] * ho_judge + (1 - best[1]) * ho_stack
    _, pred = score(p, "best blend")
    qt = corpus.loc[~tr, "qtype"].values
    print("\nper-qtype (best blend):")
    for q in sorted(set(qt)):
        m = qt == q
        if m.sum() >= 5:
            print(f"  {q:12s} n={m.sum():3d} F1_0={f1_class0(y_ho[m], pred[m]):.3f}")


if __name__ == "__main__":
    main()
