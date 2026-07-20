"""Build the label-free R3 stack bundle and fixed calibration artifacts.

This script runs offline before publishing Kaggle assets. It uses all 1,608
pseudo-labeled rows plus four-fifths of the 299 development rows in each fold to
produce honest five-fold OOF probabilities for the held-out fifth. The final
stack is then fitted once on all 1,907 rows and serialized without training labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import features_lib as fl


META_COLUMNS = ["label", "has_ctx", "split", "row_id"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def f1_class0(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 1) & (pred == 0)).sum())
    fn = int(((y == 0) & (pred == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def flat_threshold(y, proba):
    grid = np.arange(0.02, 0.99, 0.01)
    scores = np.asarray([f1_class0(y, np.asarray(proba) >= t) for t in grid])
    good = grid[scores >= scores.max() - 0.005]
    threshold = float(np.median(good))
    return threshold, f1_class0(y, np.asarray(proba) >= threshold)


def validate_judge_probs(path: Path, ids: np.ndarray) -> np.ndarray:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {str(int(i)) for i in ids}
    if set(raw) != expected:
        missing, extra = expected - set(raw), set(raw) - expected
        raise ValueError(f"holdout probability ids differ: missing={len(missing)}, extra={len(extra)}")
    values = []
    for row_id in ids:
        rec = raw[str(int(row_id))]
        if not isinstance(rec, dict) or set(rec) != {"p_bn", "p_en"}:
            raise ValueError(f"malformed judge probability record for id {row_id}")
        p_bn, p_en = float(rec["p_bn"]), float(rec["p_en"])
        if not np.isfinite([p_bn, p_en]).all() or not (0 <= p_bn <= 1 and 0 <= p_en <= 1):
            raise ValueError(f"invalid judge probability for id {row_id}")
        values.append((p_bn + p_en) / 2)
    return np.asarray(values, dtype=np.float64)


def side_calibration(y, ctx, p_stack, p_judge):
    config = {"blend": {}, "judge_only": {}, "stack_only": {}}
    for side, mask in (("ctx", ctx), ("noctx", ~ctx)):
        if not mask.any():
            raise ValueError(f"no development rows for side {side}")
        best = None
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            p = weight * p_judge[mask] + (1 - weight) * p_stack[mask]
            threshold, score = flat_threshold(y[mask], p)
            candidate = (score, -abs(weight - 0.5), weight, threshold)
            if best is None or candidate > best:
                best = candidate
        config["blend"][side] = {
            "judge_weight": best[2], "threshold": best[3], "oof_f1_0": best[0],
            "n": int(mask.sum()),
        }
        for mode, p, weight in (
            ("judge_only", p_judge[mask], 1.0),
            ("stack_only", p_stack[mask], 0.0),
        ):
            threshold, score = flat_threshold(y[mask], p)
            config[mode][side] = {
                "judge_weight": weight, "threshold": threshold,
                "oof_f1_0": score, "n": int(mask.sum()),
            }
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True,
                        help="private labeled corpus_features.parquet")
    parser.add_argument("--judge-probs", type=Path, required=True,
                        help="private holdout_probs.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory receiving stack_bundle and blend_config.json")
    parser.add_argument("--stack-params", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    frame = pd.read_parquet(args.features)
    missing = [c for c in META_COLUMNS if c not in frame]
    if missing:
        raise ValueError(f"feature file missing metadata: {missing}")
    if set(frame["split"].unique()) != {"train", "holdout"}:
        raise ValueError("split must contain exactly train and holdout")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("labels must be binary")
    feature_columns = [c for c in frame.columns if c not in META_COLUMNS]
    if not feature_columns or len(feature_columns) != len(set(feature_columns)):
        raise ValueError("invalid feature column contract")
    X = frame[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("training features contain NaN/inf")

    train_mask = frame["split"].eq("train").to_numpy()
    dev_mask = ~train_mask
    X_base, y_base = X[train_mask], frame.loc[train_mask, "label"].to_numpy(dtype=int)
    X_dev, y_dev = X[dev_mask], frame.loc[dev_mask, "label"].to_numpy(dtype=int)
    ctx_dev = frame.loc[dev_mask, "has_ctx"].to_numpy(dtype=bool)
    ids_dev = frame.loc[dev_mask, "row_id"].to_numpy()
    if len(X_base) != 1608 or len(X_dev) != 299:
        raise ValueError(f"unexpected corpus partition: {len(X_base)} + {len(X_dev)}")
    if pd.Series(ids_dev).isna().any() or pd.Series(ids_dev).duplicated().any():
        raise ValueError("development row ids must be complete and unique")
    p_judge = validate_judge_probs(args.judge_probs, ids_dev)

    params = {}
    if args.stack_params:
        raw = json.loads(args.stack_params.read_text(encoding="utf-8"))
        allowed = {"n_estimators", "max_depth", "learning_rate", "subsample",
                   "colsample_bytree", "reg_lambda", "min_child_weight"}
        params = {k: v for k, v in raw.items() if k in allowed}
    strat = y_dev * 2 + ctx_dev.astype(int)
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=42)
    oof = np.full(len(X_dev), np.nan, dtype=np.float64)
    for fold, (fit_dev, val_dev) in enumerate(splitter.split(X_dev, strat), start=1):
        X_fit = np.vstack([X_base, X_dev[fit_dev]])
        y_fit = np.concatenate([y_base, y_dev[fit_dev]])
        models = fl.fit_stack_models(X_fit, y_fit, params=params)
        oof[val_dev] = fl.predict_stack_models(models, X_dev[val_dev])
        print(f"fold {fold}/{args.folds}: fit={len(X_fit)}, validation={len(val_dev)}")
    if not np.isfinite(oof).all() or ((oof < 0) | (oof > 1)).any():
        raise ValueError("OOF stack probabilities are incomplete or invalid")

    output = args.output
    bundle = output / "stack_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    output.mkdir(parents=True, exist_ok=True)
    final_models = fl.fit_stack_models(X, frame["label"].to_numpy(dtype=int), params=params)
    import sklearn
    import xgboost
    fl.save_stack_bundle(
        final_models, feature_columns, bundle,
        metadata={
            "base_rows": len(X_base), "development_rows": len(X_dev),
            "final_fit_rows": len(X), "calibration_folds": args.folds,
            "feature_source_sha256": sha256_file(args.features),
            "features_lib_sha256": sha256_file(Path(fl.__file__).resolve()),
            "xgboost_params": params,
            "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                         "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
        },
    )

    calibration = side_calibration(y_dev, ctx_dev, oof, p_judge)
    calibration.update({
        "schema_version": 1,
        "method": "5-fold OOF on 299; each fold also includes all 1608 base rows",
        "development_rows": len(X_dev),
        "judge_probs_sha256": sha256_file(args.judge_probs),
    })
    (output / "blend_config.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    oof_records = {
        str(int(row_id)): {"stack": float(ps), "judge": float(pj), "label": int(label),
                           "has_ctx": bool(ctx)}
        for row_id, ps, pj, label, ctx in zip(ids_dev, oof, p_judge, y_dev, ctx_dev)
    }
    # Private audit artifact only; final_push.py intentionally does not publish it.
    (output / "stack_oof_audit.json").write_text(
        json.dumps(oof_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"built {bundle} and {output / 'blend_config.json'}")


if __name__ == "__main__":
    main()
