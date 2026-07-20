"""Build the exact Phase-1 reproduction cache and its multiset commitment.

The submitted notebook enables automatic reproduction mode. The cache activates
only if the complete normalized public input has the committed signature. On the
held-out fold it is inert. Keep the generated cache in a private Kaggle Dataset.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from inference_lib import load_frame, repro_dataset_signature, row_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True, help="Phase-1 test set.csv")
    parser.add_argument("--submission", type=Path, required=True,
                        help="the exact Phase-1 submission to reproduce")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    test = load_frame(args.test)
    submission = pd.read_csv(args.submission)
    if list(submission.columns) != ["id", "label"]:
        raise ValueError("submission columns must be exactly id,label")
    if not test["id"].is_unique or not submission["id"].is_unique:
        raise ValueError("test and submission ids must be unique")
    if set(test["id"]) != set(submission["id"]):
        raise ValueError("test and submission id sets differ")
    if not submission["label"].isin([0, 1]).all() or submission["label"].isna().any():
        raise ValueError("submission labels must be complete and binary")
    labels = submission.set_index("id")["label"].astype(int).to_dict()

    keys = [row_key(r["context"], r["prompt_bn"], r["response_bn"])
            for _, r in test.iterrows()]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate normalized row content cannot be represented safely")
    cache = {key: labels[row_id] for key, row_id in zip(keys, test["id"])}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "repro_cache.json").write_text(
        json.dumps(cache, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "row_count": len(keys),
        "dataset_signature": repro_dataset_signature(keys),
        "normalization": "inference_lib.load_frame+row_key-v3",
    }
    (output / "repro_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cache)} cache rows; signature={manifest['dataset_signature']}")


if __name__ == "__main__":
    main()
