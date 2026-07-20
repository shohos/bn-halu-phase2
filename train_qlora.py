"""QLoRA fine-tune of Qwen2.5-7B-Instruct as a Bengali hallucination judge.

Runs on Colab A100/L4. Expects train.jsonl + holdout_299.jsonl in the working dir.
Outputs: adapter/ (LoRA + tokenizer), thresholds.json, holdout_probs.json.

Plain transformers Trainer (no TRL) to avoid API churn. Loss only on the answer tokens.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_LEN = 1536
SEED = 42
OUT = Path("out")
OUT.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------ tokenization
def encode_example(tok, ex):
    """input_ids for the full chat, labels = -100 on everything but the answer.

    Tokenizes template STRINGS (tokenize=False) — apply_chat_template(tokenize=True)
    returns a tokenizers.Encoding on transformers v5, which breaks torch.tensor()."""
    msgs = ex["messages"]
    prompt_text = tok.apply_chat_template(msgs[:1], add_generation_prompt=True, tokenize=False)
    full_text = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
    prompt_ids = list(tok(prompt_text, add_special_tokens=False)["input_ids"])
    full_ids = list(tok(full_text, add_special_tokens=False)["input_ids"])
    # mask up to the longest common prefix (junction tokens can merge across the boundary)
    k = 0
    while k < min(len(prompt_ids), len(full_ids)) and prompt_ids[k] == full_ids[k]:
        k += 1
    if len(full_ids) > MAX_LEN:  # truncate from the left; answer sits at the end
        cut = len(full_ids) - MAX_LEN
        full_ids = full_ids[cut:]
        k = max(0, k - cut)
    labels = [-100] * k + full_ids[k:]
    return {"input_ids": full_ids, "labels": labels}


class Collator:
    def __init__(self, tok):
        self.pad = tok.pad_token_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


# ------------------------------------------------------------------ P(yes) scoring
def first_token_ids(tok, yes_words, no_words):
    """Disjoint first-token id sets. Bengali words share a byte-fragment first token in
    their space-prefixed forms (' হ্যাঁ' and ' না' both start with id 35178 on Qwen), so
    any id landing in both sets is dropped — it carries no yes/no signal."""
    def collect(words):
        ids = set()
        for w in words:
            for form in [w, " " + w]:
                t = tok.encode(form, add_special_tokens=False)
                if t:
                    ids.add(t[0])
        return ids
    y, n = collect(yes_words), collect(no_words)
    shared = y & n
    return sorted(y - shared), sorted(n - shared)


@torch.no_grad()
def yes_prob(model, tok, prompts, yes_ids, no_ids, batch_size=16):
    out = []
    tok.padding_side = "left"
    tok.truncation_side = "left"  # keep the question+answer tail, drop passage head
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p_yes = probs[:, yes_ids].sum(-1)
        p_no = probs[:, no_ids].sum(-1)
        out.extend((p_yes / (p_yes + p_no + 1e-9)).cpu().tolist())
    tok.padding_side = "right"
    return out


def f1_class0(y, pred):
    tp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def flat_region_threshold(y, proba):
    """Median of thresholds whose F1_0 is within 0.005 of the max (robust, not knife-edge)."""
    grid = np.arange(0.10, 0.91, 0.01)
    f1s = [f1_class0(y, [int(p >= t) for p in proba]) for t in grid]
    best = max(f1s)
    good = [t for t, f in zip(grid, f1s) if f >= best - 0.005]
    return float(np.median(good)), best


def eval_holdout(model, tok, holdout, tag):
    """Score both templates on the holdout; returns (overall F1_0, per-side dict, probs)."""
    yes_ids, no_ids = first_token_ids(tok, ["Yes", "হ্যাঁ"], ["No", "না"])

    def chat(user):
        return tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)

    p_bn = yes_prob(model, tok, [chat(h["prompt_bn_tmpl"]) for h in holdout], yes_ids, no_ids)
    p_en = yes_prob(model, tok, [chat(h["prompt_en_tmpl"]) for h in holdout], yes_ids, no_ids)
    proba = [(a + b) / 2 for a, b in zip(p_bn, p_en)]

    y = [h["label"] for h in holdout]
    ctx = [h["has_ctx"] for h in holdout]
    res = {}
    for side, mask in [("ctx", ctx), ("noctx", [not c for c in ctx])]:
        ys = [a for a, m in zip(y, mask) if m]
        ps = [a for a, m in zip(proba, mask) if m]
        th, f1 = flat_region_threshold(ys, ps)
        res[side] = {"threshold": th, "f1_0": round(f1, 4), "n": len(ys)}
    pred = [int(p >= (res["ctx"]["threshold"] if c else res["noctx"]["threshold"]))
            for p, c in zip(proba, ctx)]
    overall = f1_class0(y, pred)
    print(f"[{tag}] ctx {res['ctx']['f1_0']:.4f}@{res['ctx']['threshold']:.2f} | "
          f"noctx {res['noctx']['f1_0']:.4f}@{res['noctx']['threshold']:.2f} | "
          f"OVERALL F1_0 {overall:.4f}", flush=True)
    from collections import defaultdict
    byq = defaultdict(list)
    for h, pr in zip(holdout, pred):
        byq[h["qtype"]].append((h["label"], pr))
    for qt, pairs in sorted(byq.items()):
        if len(pairs) >= 5:
            print(f"    {qt:12s} n={len(pairs):3d} F1_0={f1_class0(*zip(*pairs)):.3f}")
    return overall, res, (p_bn, p_en)


def main():
    import argparse
    import glob as globmod

    from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training,
                      set_peft_model_state_dict)
    from safetensors.torch import load_file
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments)

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--side", choices=["all", "ctx", "noctx"], default="all",
                    help="restrict training rows to one evidence regime")
    ap.add_argument("--include-holdout", action="store_true",
                    help="final run: also train on the 299 (holdout metrics become biased)")
    args_cli = ap.parse_args()

    train = load_jsonl("train.jsonl")
    holdout = load_jsonl("holdout_299.jsonl")
    if args_cli.side != "all":
        want = args_cli.side == "ctx"
        train = [t for t in train if t.get("has_ctx") == want]
    if args_cli.include_holdout:
        for h in holdout:  # bn + en variants, mirroring build_corpus
            for tmpl, yes, no in [("bn", "হ্যাঁ", "না"), ("en", "Yes", "No")]:
                train.append({"messages": [
                    {"role": "user", "content": h[f"prompt_{tmpl}_tmpl"]},
                    {"role": "assistant", "content": yes if h["label"] == 1 else no}]})
    print(f"train {len(train)} examples, holdout {len(holdout)} "
          f"(include_holdout={args_cli.include_holdout})")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                 device_map="auto", torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=args_cli.dropout,
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # zero-shot reference before any training
    model.eval()
    eval_holdout(model, tok, holdout, "epoch 0 / zero-shot")
    model.train()

    ds = [encode_example(tok, ex) for ex in train]
    random.shuffle(ds)

    args = TrainingArguments(
        output_dir=str(OUT / "ckpt"), num_train_epochs=args_cli.epochs,
        per_device_train_batch_size=8, gradient_accumulation_steps=2,
        learning_rate=args_cli.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=20, save_strategy="epoch", save_total_limit=None,
        report_to=[], seed=SEED, gradient_checkpointing=True,
    )
    Trainer(model=model, args=args, train_dataset=ds, data_collator=Collator(tok)).train()

    # ---------------- evaluate every epoch checkpoint on the holdout, keep the best
    model.eval()
    model.config.use_cache = True
    ckpts = sorted(globmod.glob(str(OUT / "ckpt" / "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    best = None
    all_probs = {}
    for ep, ck in enumerate(ckpts, start=1):
        sd = load_file(f"{ck}/adapter_model.safetensors")
        set_peft_model_state_dict(model, sd)
        overall, res, probs = eval_holdout(model, tok, holdout, f"epoch {ep}")
        all_probs[f"epoch{ep}"] = {str(h["id"]): {"p_bn": a, "p_en": b}
                                   for h, a, b in zip(holdout, probs[0], probs[1])}
        if best is None or overall > best[0]:
            best = (overall, res, probs, ck, ep)
    json.dump(all_probs, open(OUT / "holdout_probs_all_epochs.json", "w"))

    overall, res, (p_bn, p_en), ck, ep = best
    print(f"\nBEST = epoch {ep} ({ck}) OVERALL holdout F1_0 = {overall:.4f}")
    print(f"GATE: overall>=0.80 {'PASS' if overall >= 0.80 else 'FAIL'}; "
          f"sides>=0.75 {'PASS' if min(res['ctx']['f1_0'], res['noctx']['f1_0']) >= 0.75 else 'FAIL'}")

    sd = load_file(f"{ck}/adapter_model.safetensors")
    set_peft_model_state_dict(model, sd)
    json.dump({"th_ctx": res["ctx"]["threshold"], "th_noctx": res["noctx"]["threshold"],
               "holdout_f1_0": round(overall, 4), "per_side": res, "best_epoch": ep,
               "include_holdout": args_cli.include_holdout},
              open(OUT / "thresholds.json", "w"), indent=2)
    json.dump({str(h["id"]): {"p_bn": a, "p_en": b} for h, a, b in zip(holdout, p_bn, p_en)},
              open(OUT / "holdout_probs.json", "w"))
    model.save_pretrained(OUT / "adapter")
    tok.save_pretrained(OUT / "adapter")
    print(f"saved adapter + thresholds to {OUT}/")


if __name__ == "__main__":
    main()
