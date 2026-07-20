"""Recompute a stratified feature sample in the Kaggle image and compare it.

Run this once against the exact feature-model dataset before final packaging. It
catches tokenizer/model/version drift, including silent all--1 QA fallbacks.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import features_lib as fl


META = {"label", "has_ctx", "split", "row_id"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--feature-models", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    corpus = pd.read_parquet(args.corpus).reset_index(drop=True)
    reference = pd.read_parquet(args.reference_features).reset_index(drop=True)
    if len(corpus) != len(reference):
        raise ValueError("corpus/reference row counts differ")
    rng = np.random.default_rng(42)
    picks = []
    for has_ctx in (False, True):
        candidates = np.flatnonzero(corpus["has_ctx"].to_numpy(dtype=bool) == has_ctx)
        count = min(len(candidates), max(1, args.sample_size // 2))
        picks.extend(rng.choice(candidates, count, replace=False).tolist())
    picks = np.asarray(sorted(picks))
    source = corpus.iloc[picks].copy()
    actual = fl.extract_all(
        source,
        args.feature_models / "mdeberta-xnli",
        args.feature_models / "xlmr-squad2",
        args.feature_models / "e5-base",
    ).reset_index(drop=True)
    columns = [c for c in reference.columns if c not in META]
    if actual.columns.tolist() != columns:
        raise ValueError("feature schema/order drift")
    expected = reference.loc[picks, columns].reset_index(drop=True)
    failures, maxima = {}, {}
    for col in columns:
        a, b = actual[col].to_numpy(dtype=float), expected[col].to_numpy(dtype=float)
        delta = float(np.max(np.abs(a - b)))
        maxima[col] = delta
        if not np.allclose(a, b, rtol=1e-3, atol=args.atol, equal_nan=False):
            failures[col] = delta
    report = {"rows": picks.tolist(), "atol": args.atol,
              "max_abs_delta": maxima, "failures": failures}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(json.dumps(failures, indent=2))
        raise SystemExit("feature parity failed")
    print(f"STAGE:FEATURE_PARITY_OK {len(picks)} rows, {len(columns)} columns")


if __name__ == "__main__":
    main()
