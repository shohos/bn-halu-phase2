"""Local checks for inference_lib: math verifier vs Phase 1 truth, full run() with a
stubbed scorer on the real Phase 1 test file, and mutated-CSV robustness.

Run:  python test_local.py     (needs data/thresholds.json — placeholder ok pre-Colab)
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inference_lib as lib  # noqa: E402

DATA = HERE / "data"
ROOT = HERE.parent


def setup_assets():
    """Stage an assets dir shaped like the Kaggle dataset."""
    if not (DATA / "thresholds.json").exists():
        (DATA / "thresholds.json").write_text(
            json.dumps({"th_ctx": 0.5, "th_noctx": 0.5}), encoding="utf-8")
        print("NOTE: placeholder thresholds.json written (replace with Colab output)")
    shutil.copy(ROOT / "dataset samples.json", DATA / "dataset samples.json")
    return DATA


def stub_scores(prompts):
    return [0.5] * len(prompts)


def test_math_verifier(assets):
    df = lib.load_frame(ROOT / "test set.csv")
    mv = lib.math_verify(df)
    truth = {int(k): int(v) for k, v in json.loads(
        (ROOT / "pipeline/cache/claude_judge_test.json").read_text(encoding="utf-8")).items()}
    ids = df.loc[list(mv.keys()), "id"].astype(int).tolist()
    agree = sum(1 for idx, tid in zip(mv, ids) if truth.get(tid, -1) == mv[idx])
    print(f"math verifier: {len(mv)} templated rows, agreement with corrected "
          f"Phase 1 labels {agree}/{len(mv)}")
    assert len(mv) >= 100, "expected >=100 templated rows on Phase 1 test"
    assert agree == len(mv), "math verifier must match corrected Phase 1 labels exactly"


def test_reproduction(assets):
    sub, _ = lib.run(ROOT / "test set.csv", assets, score_fn=stub_scores,
                     out_path=tempfile.mktemp(suffix=".csv"))
    phase1 = pd.read_csv(ROOT / "submission.csv")
    m = (sub["label"].values == phase1["label"].values).mean()
    print(f"reproduction rate vs Phase 1 submission: {m:.4f}")
    assert m == 1.0, "repro layer must reproduce Phase 1 submission exactly"


def test_mutations(assets):
    base = lib.load_frame(ROOT / "test set.csv").head(60)
    raw = pd.read_csv(ROOT / "test set.csv").head(60)
    cases = {}
    # shuffled, non-contiguous ids
    t = raw.copy(); t["id"] = [7 * i + 1003 for i in range(len(t))]
    cases["weird_ids"] = t.sample(frac=1, random_state=0)
    # mutated responses (repro cache must NOT fire), NaN/[NULL]/empty contexts
    t = raw.copy(); t["response_bn"] = t["response_bn"].astype(str) + " x"
    t.loc[t.index[:10], "context"] = float("nan")
    t.loc[t.index[10:20], "context"] = "[NULL]"
    t.loc[t.index[20:25], "context"] = ""
    cases["new_fold_like"] = t
    # numeric responses, reordered columns, quoted newlines in context
    t = raw.copy(); t.loc[t.index[:5], "response_bn"] = 42
    t.loc[t.index[5], "context"] = "লাইন এক\nলাইন দুই, \"উদ্ধৃতি\""
    cases["messy"] = t[["response_bn", "context", "id", "prompt_bn"]]
    # no id column at all
    cases["no_id"] = raw.drop(columns=["id"])

    for name, t in cases.items():
        p = Path(tempfile.mkdtemp()) / "test set.csv"
        t.to_csv(p, index=False)
        sub, _ = lib.run(p, assets, score_fn=stub_scores,
                         out_path=str(p.parent / "submission.csv"))
        assert len(sub) == len(t) and sub["label"].isin([0, 1]).all()
        if name == "new_fold_like":
            reloaded = lib.load_frame(p)
            hits = sum(1 for _, r in reloaded.iterrows()
                       if lib.row_key(r["context"], r["prompt_bn"], r["response_bn"])
                       in json.loads((assets / "repro_cache.json").read_text()))
            assert hits == 0, "repro cache fired on mutated content"
        print(f"mutation '{name}': OK ({len(sub)} rows)")


if __name__ == "__main__":
    assets = setup_assets()
    test_math_verifier(assets)
    test_reproduction(assets)
    test_mutations(assets)
    print("\nALL LOCAL TESTS PASSED")
