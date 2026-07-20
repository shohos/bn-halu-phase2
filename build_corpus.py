"""Phase 2 corpus: merge label sources -> corpus.parquet + train.jsonl + holdout_299.jsonl.

Sources (decreasing trust):
  - dataset samples.json          299 official labels  -> holdout (never trained on)
  - claude_judge_test.json        frontier labels, no-context Phase 1 test rows (-1 dropped)
  - claude_ctx_review_golden910   verified labels, context Phase 1 test rows

Consistency filter: drop a pseudo-labeled row only when BOTH 32B judge caches exist for
it and both confidently contradict the stored label.

Run:  python build_corpus.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from common import CACHE_DIR, load_sample, load_test  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

CTX_CLIP = 3000  # same clip as Phase 1 llm_judge.py

TMPL_BN_CTX = (
    "প্রসঙ্গ: {c}\n\nপ্রশ্ন: {p}\nপ্রস্তাবিত উত্তর: {r}\n"
    "উপরের প্রসঙ্গ অনুযায়ী প্রস্তাবিত উত্তরটি কি সঠিক? শুধুমাত্র 'হ্যাঁ' অথবা 'না' লিখুন।"
)
TMPL_BN_NOCTX = (
    "প্রশ্ন: {p}\nপ্রস্তাবিত উত্তর: {r}\n"
    "প্রস্তাবিত উত্তরটি কি সঠিক ও তথ্যগতভাবে নির্ভুল? শুধুমাত্র 'হ্যাঁ' অথবা 'না' লিখুন।"
)
TMPL_EN_CTX = (
    "You are a careful fact-checker. Based ONLY on the passage below, decide if the "
    "proposed answer to the question is correct.\nPassage: {c}\nQuestion: {p}\n"
    "Proposed answer: {r}\nReply with only Yes or No."
)
TMPL_EN_NOCTX = (
    "You are a careful fact-checker. A question was asked in Bengali and an answer was "
    "proposed.\nQuestion: {p}\nProposed answer: {r}\n"
    "Is the proposed answer factually correct? Reply with only Yes or No."
)


def build_prompt(row, tmpl):
    c = str(row["context"])[:CTX_CLIP]
    if row["has_ctx"]:
        t = TMPL_BN_CTX if tmpl == "bn" else TMPL_EN_CTX
        return t.format(c=c, p=row["prompt_bn"], r=row["response_bn"])
    t = TMPL_BN_NOCTX if tmpl == "bn" else TMPL_EN_NOCTX
    return t.format(p=row["prompt_bn"], r=row["response_bn"])


def load_json(name):
    p = CACHE_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    sample_df = load_sample()
    test_df = load_test().set_index("id", drop=False)

    frontier = {int(k): int(v) for k, v in load_json("claude_judge_test.json").items()}
    golden = {int(k): int(v) for k, v in load_json("claude_ctx_review_golden910.json").items()}
    colab = load_json("judge_ctx_colab.json")
    qwen3 = load_json("judge_ctx_qwen3.json")

    overlap = set(frontier) & set(golden)
    if overlap:
        print(f"WARNING: {len(overlap)} ids in both caches; golden wins: {sorted(overlap)[:10]}")

    rows, dropped = [], []
    seen = set()
    for src, labels in [("golden_ctx", golden), ("frontier_noctx", frontier)]:
        for tid, lab in labels.items():
            if lab == -1 or tid in seen or tid not in test_df.index:
                continue
            seen.add(tid)
            r = test_df.loc[tid]
            # consistency check vs both 32B caches (test rows are keyed by plain id)
            pc = colab.get(str(tid))
            pq = qwen3.get(str(tid))
            if pc is not None and pq is not None:
                p_c = 0.5 * (pc["p_yes_bn"] + pc["p_yes_en"])
                p_q = pq["p_mean"]
                if (lab == 1 and p_c < 0.1 and p_q < 0.1) or (lab == 0 and p_c > 0.9 and p_q > 0.9):
                    dropped.append((tid, src, lab, round(p_c, 3), round(p_q, 3)))
                    continue
            rows.append(dict(
                id=int(tid), source=src, label=int(lab),
                context=r["context"], prompt_bn=r["prompt_bn"], response_bn=r["response_bn"],
                has_ctx=bool(r["has_ctx"]), qtype=r["qtype"], split="train",
            ))

    for i, r in sample_df.iterrows():
        rows.append(dict(
            id=int(i), source="official_sample", label=int(r["label"]),
            context=r["context"], prompt_bn=r["prompt_bn"], response_bn=r["response_bn"],
            has_ctx=bool(r["has_ctx"]), qtype=r["qtype"], split="holdout",
        ))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "corpus.parquet", index=False)

    print(f"corpus: {len(df)} rows "
          f"(train {sum(df.split == 'train')}, holdout {sum(df.split == 'holdout')})")
    print(df[df.split == "train"].groupby(["source", "label"]).size())
    print(f"consistency-dropped {len(dropped)} rows:")
    for d in dropped:
        print("  ", d)

    # train.jsonl: bn + en template variants as augmentation, answer token as target
    with open(OUT / "train.jsonl", "w", encoding="utf-8") as f:
        for _, r in df[df.split == "train"].iterrows():
            for tmpl in ("bn", "en"):
                ans = ("হ্যাঁ" if r.label == 1 else "না") if tmpl == "bn" else \
                      ("Yes" if r.label == 1 else "No")
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": build_prompt(r, tmpl)},
                        {"role": "assistant", "content": ans},
                    ],
                    "id": r.id, "source": r.source, "tmpl": tmpl,
                    "label": r.label, "has_ctx": r.has_ctx, "qtype": r.qtype,
                }, ensure_ascii=False) + "\n")

    # holdout: prompts only + label, scored (not trained) in the Colab notebook
    with open(OUT / "holdout_299.jsonl", "w", encoding="utf-8") as f:
        for _, r in df[df.split == "holdout"].iterrows():
            f.write(json.dumps({
                "prompt_bn_tmpl": build_prompt(r, "bn"),
                "prompt_en_tmpl": build_prompt(r, "en"),
                "id": r.id, "label": r.label, "has_ctx": r.has_ctx, "qtype": r.qtype,
            }, ensure_ascii=False) + "\n")

    print(f"wrote {OUT / 'train.jsonl'} and {OUT / 'holdout_299.jsonl'}")


if __name__ == "__main__":
    main()
