"""Does adding the 299 organizer labels to the stack's training set help?

Honest estimate via 5-fold CV over the 299: each fold trains on
(all 1,608 pseudo rows + 4/5 of the 299) and predicts the held-out 1/5.
That is exactly the model shipped at inference (trained on all 1,907), scored
without ever predicting a row it trained on.

Run:  python eval_stack2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features_lib as fl  # noqa: E402
from inference_lib import f1_class0, flat_threshold  # noqa: E402


def main():
    corpus = pd.read_parquet(HERE / "data" / "corpus.parquet")
    feats = pd.read_parquet(HERE / "data" / "feats_corpus.parquet")
    tr = (corpus["split"] == "train").values
    X = feats.values.astype(np.float32)
    y = corpus["label"].values
    ho_idx = np.where(~tr)[0]
    y_ho = y[ho_idx]
    ctx_ho = corpus.loc[~tr, "has_ctx"].values
    ids_ho = corpus.loc[~tr, "id"].values

    oof = np.zeros(len(ho_idx))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for f, (a, b) in enumerate(skf.split(ho_idx, y_ho)):
        train_rows = np.concatenate([np.where(tr)[0], ho_idx[a]])
        xgb, lr = fl.make_models()
        xgb.fit(X[train_rows], y[train_rows])
        sc = StandardScaler().fit(X[train_rows])
        lr.fit(sc.transform(X[train_rows]), y[train_rows])
        Xb = X[ho_idx[b]]
        oof[b] = 0.5 * xgb.predict_proba(Xb)[:, 1] + \
            0.5 * lr.predict_proba(sc.transform(Xb))[:, 1]

    jp = json.loads((HERE / "upload_bn_halu_assets" / "holdout_probs.json").read_text())
    ho_judge = np.array([(jp[str(i)]["p_bn"] + jp[str(i)]["p_en"]) / 2 for i in ids_ho])

    def score(p, tag):
        tc, _ = flat_threshold(y_ho[ctx_ho], p[ctx_ho])
        tn, _ = flat_threshold(y_ho[~ctx_ho], p[~ctx_ho])
        pred = np.where(ctx_ho, p >= tc, p >= tn).astype(int)
        f1 = f1_class0(y_ho, pred)
        print(f"{tag:34s} overall {f1:.4f} | ctx {f1_class0(y_ho[ctx_ho], pred[ctx_ho]):.4f} "
              f"| noctx {f1_class0(y_ho[~ctx_ho], pred[~ctx_ho]):.4f} | th {tc:.2f}/{tn:.2f}")
        return f1, pred, (tc, tn)

    print()
    score(ho_judge, "judge alone")
    score(oof, "stack (1608+299 CV)")
    best = None
    for w in np.arange(0, 1.01, 0.1):
        f1, _, th = score(w * ho_judge + (1 - w) * oof, f"blend w_judge={w:.1f}")
        if best is None or f1 > best[0]:
            best = (f1, w, th)
    print(f"\nBEST: w_judge={best[1]:.1f} th={best[2]} -> F1_0 {best[0]:.4f}")
    json.dump({"w_judge": float(best[1]), "th_ctx": float(best[2][0]),
               "th_noctx": float(best[2][1]), "cv_f1_0": float(best[0])},
              open(HERE / "data" / "blend_cv.json", "w"), indent=2)

    p = best[1] * ho_judge + (1 - best[1]) * oof
    _, pred, _ = score(p, "best blend")
    qt = corpus.loc[~tr, "qtype"].values
    print("\nper-qtype:")
    for q in sorted(set(qt)):
        m = qt == q
        if m.sum() >= 5:
            print(f"  {q:12s} n={m.sum():3d} F1_0={f1_class0(y_ho[m], pred[m]):.3f}")


if __name__ == "__main__":
    main()
