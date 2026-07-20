"""Final packaging: rebuild assets from a Colab adapter zip, push, dry-run, verify.

Usage:  python final_push.py [path/to/bn_halu_adapter.zip]

Unlike push_and_dryrun.py this also rebuilds corpus_features.parquet (text-free) and
refreshes holdout_probs.json from the new training run.
"""
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "upload_bn_halu_assets"


def rebuild_assets(zip_path=None):
    if zip_path:
        tmp = HERE / "_adapter_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zip_path) as f:
            f.extractall(tmp)
        for name in ["adapter", "thresholds.json", "holdout_probs.json"]:
            src = tmp / name
            dst = ASSETS / name
            if src.exists():
                shutil.rmtree(dst, ignore_errors=True) if src.is_dir() else None
                shutil.copytree(src, dst) if src.is_dir() else shutil.copy(src, dst)
        print("adapter refreshed:", json.loads((ASSETS / "thresholds.json").read_text()))

    # text-free corpus features (no competition text is ever published)
    corpus = pd.read_parquet(HERE / "data" / "corpus.parquet")
    feats = pd.read_parquet(HERE / "data" / "feats_corpus.parquet").reset_index(drop=True)
    feats["label"] = corpus["label"].values
    feats["has_ctx"] = corpus["has_ctx"].values
    feats["split"] = corpus["split"].values
    # Publish row_id ONLY for the 299 organizer rows (needed to key holdout_probs).
    # Training rows are Phase 1 test rows: their ids + our labels would let anyone
    # join a derived answer key back onto the public test set.
    feats["row_id"] = np.where(corpus["split"].values == "holdout", corpus["id"].values, -1)
    feats.to_parquet(ASSETS / "corpus_features.parquet", index=False)
    (ASSETS / "corpus.parquet").unlink(missing_ok=True)
    for f in ["inference_lib.py", "features_lib.py"]:
        shutil.copy(HERE / f, ASSETS / f)

    text_cols = [c for c in feats.columns if feats[c].dtype == object and c != "split"]
    assert not text_cols, f"text columns would be published: {text_cols}"
    print(f"assets ready: {sorted(p.name for p in ASSETS.iterdir())}")


if __name__ == "__main__":
    rebuild_assets(sys.argv[1] if len(sys.argv) > 1 else None)
    subprocess.run([sys.executable, str(HERE / "make_notebook.py")], check=True)
    subprocess.run([sys.executable, str(HERE / "push_and_dryrun.py")], check=True)
