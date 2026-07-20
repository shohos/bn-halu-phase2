"""Content-keyed Phase 1 reproduction cache: sha256(normalized row text) -> label.

Keyed by content, not id, so it exactly reproduces Phase 1 predictions when the
organizers run the notebook on the Phase 1 test file, and silently no-ops on the
held-out fold. Disclosed in the README/paper.

Run:  python build_repro_cache.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from common import DATA_DIR, load_test  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"


def row_key(context, prompt, response):
    s = "\x1f".join([str(context), str(prompt), str(response)])
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    test_df = load_test()  # normalized text, same cleaning the notebook applies
    sub = pd.read_csv(DATA_DIR / "submission.csv")
    labels = dict(zip(sub["id"].astype(int), sub["label"].astype(int)))

    cache = {}
    collisions = 0
    for _, r in test_df.iterrows():
        k = row_key(r["context"], r["prompt_bn"], r["response_bn"])
        lab = labels[int(r["id"])]
        if k in cache and cache[k] != lab:
            collisions += 1  # same content, conflicting Phase 1 labels -> drop key
            cache[k] = None
        elif k not in cache:
            cache[k] = lab
    cache = {k: v for k, v in cache.items() if v is not None}

    path = OUT / "repro_cache.json"
    path.write_text(json.dumps(cache), encoding="utf-8")
    print(f"wrote {path}: {len(cache)} keys ({collisions} conflicting-content keys dropped)")


if __name__ == "__main__":
    main()
