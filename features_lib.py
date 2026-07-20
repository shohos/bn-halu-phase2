"""Feature extraction + stacked classifier for Phase 2 (ported from Phase 1).

Features: lexical/overlap, e5 embedding cosines (plain transformers, mean pooling),
mDeBERTa-XNLI entailment (context rows), XLM-R-SQuAD2 extractive-QA agreement
(context rows). Stack: XGBoost + standardized LogisticRegression, 50/50 blend,
Phase 1 hyperparameters, trained in-kernel on shipped features for the 1,907
labeled rows, with the distilled judge's P(yes) as two extra features.
"""
import os

# transformers pulls in TF for some image utils; on a machine with a numpy-1.x-built
# TF that import explodes. Harmless on Kaggle, required locally (same as Phase 1).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from inference_lib import bn_to_en_digits

BN_STOP = set(
    "এর ও এবং যে যা কি কী তা সে এই ওই তার কে থেকে করে হয় হন ছিল ছিলেন একটি এক দুই "
    "জন্য সালে সাল করা হয়েছে হয়েছিল আছে নেই না হ্যাঁ প্রায় মোট বেশি কম".split()
)


def tokens(s):
    return [t for t in re.findall(r"[\wঀ-৿]+", str(s)) if t]


def content_tokens(s):
    return [t for t in tokens(s) if t not in BN_STOP and len(t) > 1]


def char_ratio(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()


# ---------------------------------------------------------------- lexical (verbatim)
def lexical_features(df):
    feats = pd.DataFrame(index=df.index)
    feats["has_ctx"] = df["has_ctx"].astype(int)
    for col in ["prompt_bn", "response_bn", "context"]:
        feats[f"{col}_nchar"] = df[col].str.len()
        feats[f"{col}_nword"] = df[col].map(lambda s: len(tokens(s)))
    feats["resp_prompt_len_ratio"] = feats["response_bn_nchar"] / (feats["prompt_bn_nchar"] + 1)

    resp_en = df["response_bn"].map(bn_to_en_digits)
    feats["resp_has_digit"] = resp_en.str.contains(r"\d").astype(int)
    feats["resp_has_year"] = resp_en.str.contains(r"\b1[0-9]{3}\b|\b20[0-9]{2}\b").astype(int)
    feats["prompt_asks_when"] = df["prompt_bn"].str.contains("কবে|কত সালে|কোন সালে|তারিখ|সময়ে").astype(int)
    feats["prompt_recency"] = df["prompt_bn"].str.contains("বর্তমান|এখনকার|সাম্প্রতিক|এ বছর|চলতি").astype(int)
    feats["prompt_asks_howmuch"] = df["prompt_bn"].str.contains("কত").astype(int)
    feats["when_but_no_year"] = ((feats["prompt_asks_when"] == 1) & (feats["resp_has_year"] == 0)).astype(int)
    feats["howmuch_but_no_digit"] = ((feats["prompt_asks_howmuch"] == 1) & (feats["resp_has_digit"] == 0)).astype(int)

    in_ctx, tok_overlap, ctok_overlap = [], [], []
    for ctx, resp in zip(df["context"], df["response_bn"]):
        if not ctx:
            in_ctx.append(-1); tok_overlap.append(-1.0); ctok_overlap.append(-1.0)
            continue
        r = str(resp).strip().rstrip("।.")
        in_ctx.append(int(r in ctx))
        rt = tokens(resp)
        ct = set(tokens(ctx))
        tok_overlap.append(sum(t in ct for t in rt) / max(len(rt), 1))
        rct = content_tokens(resp)
        cct = set(content_tokens(ctx))
        ctok_overlap.append(sum(t in cct for t in rct) / max(len(rct), 1))
    feats["resp_in_ctx"] = in_ctx
    feats["resp_ctx_tok_overlap"] = tok_overlap
    feats["resp_ctx_ctok_overlap"] = ctok_overlap
    feats["resp_prompt_tok_overlap"] = [
        sum(t in set(tokens(p)) for t in tokens(r)) / max(len(tokens(r)), 1)
        for p, r in zip(df["prompt_bn"], df["response_bn"])
    ]
    for qt in ["date", "math", "mcq", "meaning", "other", "spelling", "translation", "who"]:
        feats[f"qt_{qt}"] = (df["qtype"] == qt).astype(int)
    return feats


# ---------------------------------------------------------------- NLI
def nli_features(df, model_dir, batch_size=16):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, torch_dtype=torch.float16).to("cuda").eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    @torch.no_grad()
    def run(premises, hypotheses):
        probs = np.zeros((len(premises), 3), dtype=np.float32)
        for i in range(0, len(premises), batch_size):
            enc = tok(premises[i:i + batch_size], hypotheses[i:i + batch_size],
                      truncation=True, max_length=512, padding=True,
                      return_tensors="pt").to("cuda")
            logits = model(**enc).logits.float()
            probs[i:i + batch_size] = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    feats = pd.DataFrame(index=df.index)
    for name in ["ent", "neu", "con"]:
        for variant in ["r", "pr"]:
            feats[f"nli_{name}_{variant}"] = -1.0
    sub = df[df["has_ctx"]]
    if len(sub):
        order = [None, None, None]
        for idx, lab in id2label.items():
            if "entail" in lab:
                order[0] = idx
            elif "neutral" in lab:
                order[1] = idx
            else:
                order[2] = idx
        for variant, hyps in [("r", sub["response_bn"].tolist()),
                              ("pr", (sub["prompt_bn"] + " " + sub["response_bn"]).tolist())]:
            probs = run(sub["context"].tolist(), hyps)
            feats.loc[sub.index, f"nli_ent_{variant}"] = probs[:, order[0]]
            feats.loc[sub.index, f"nli_neu_{variant}"] = probs[:, order[1]]
            feats.loc[sub.index, f"nli_con_{variant}"] = probs[:, order[2]]
    del model
    gc.collect()
    import torch as _t
    _t.cuda.empty_cache()
    return feats


# ---------------------------------------------------------------- extractive QA
def qa_features(df, model_dir, batch_size=16):
    import torch
    feats = pd.DataFrame(index=df.index)
    feats["qa_score"] = -1.0
    feats["qa_ans_char_sim"] = -1.0
    feats["qa_ans_tok_overlap"] = -1.0
    feats["qa_ans_in_resp"] = -1
    feats["qa_resp_in_ans"] = -1
    sub = df[df["has_ctx"]]
    if not len(sub):
        return feats
    try:
        from transformers import pipeline as hf_pipeline
        qa = hf_pipeline("question-answering", model=model_dir, device=0,
                         batch_size=batch_size, torch_dtype=torch.float16)
        qs = sub["prompt_bn"].tolist()
        cs = sub["context"].tolist()
        # transformers v5 made __call__ keyword-only; v4 accepted a list of dicts.
        try:
            outputs = qa(question=qs, context=cs, max_answer_len=64,
                         handle_impossible_answer=False)
        except TypeError:
            outputs = qa([{"question": q, "context": c} for q, c in zip(qs, cs)],
                         max_answer_len=64, handle_impossible_answer=False)
        if isinstance(outputs, dict):
            outputs = [outputs]
        char_sims, tok_ovs, ans_in, resp_in, scores = [], [], [], [], []
        for o, resp in zip(outputs, sub["response_bn"]):
            a, r = str(o["answer"]).strip(), str(resp).strip()
            scores.append(o["score"])
            char_sims.append(char_ratio(a, r))
            at = tokens(a)
            tok_ovs.append(sum(t in set(tokens(r)) for t in at) / max(len(at), 1))
            ans_in.append(int(a in r) if a else 0)
            resp_in.append(int(r.rstrip("।.") in a) if r else 0)
        feats.loc[sub.index, "qa_score"] = scores
        feats.loc[sub.index, "qa_ans_char_sim"] = char_sims
        feats.loc[sub.index, "qa_ans_tok_overlap"] = tok_ovs
        feats.loc[sub.index, "qa_ans_in_resp"] = ans_in
        feats.loc[sub.index, "qa_resp_in_ans"] = resp_in
        del qa
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"QA features failed ({type(e).__name__}: {e}); keeping -1 fillers")
    return feats


# ---------------------------------------------------------------- e5 embeddings
def embedding_features(df, model_dir, batch_size=32):
    """e5 mean pooling + L2 normalize via plain transformers (matches the
    sentence-transformers Pooling(mean)+Normalize pipeline used in Phase 1)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16).to("cuda").eval()

    @torch.no_grad()
    def encode(texts, prefix):
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = [f"{prefix}: {t}" if t else f"{prefix}: " for t in texts[i:i + batch_size]]
            enc = tok(batch, truncation=True, max_length=512, padding=True,
                      return_tensors="pt").to("cuda")
            out = model(**enc).last_hidden_state.float()
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            vecs.append(emb.cpu().numpy())
        return np.vstack(vecs)

    emb_p = encode(df["prompt_bn"].tolist(), "query")
    emb_r = encode(df["response_bn"].tolist(), "passage")
    emb_c = encode(df["context"].tolist(), "passage")
    feats = pd.DataFrame(index=df.index)
    feats["cos_pr"] = (emb_p * emb_r).sum(axis=1)
    has = df["has_ctx"].values
    feats["cos_cr"] = np.where(has, (emb_c * emb_r).sum(axis=1), -1.0)
    feats["cos_cp"] = np.where(has, (emb_c * emb_p).sum(axis=1), -1.0)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return feats


def extract_all(df, mdeberta_dir, xlmr_dir, e5_dir):
    parts = [lexical_features(df), embedding_features(df, e5_dir),
             nli_features(df, mdeberta_dir), qa_features(df, xlmr_dir)]
    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------- stack
def add_judge_cols(feats, df, p_bn, p_en):
    out = feats.copy()
    out["judge_p_bn"] = p_bn
    out["judge_p_en"] = p_en
    mean = (out["judge_p_bn"] + out["judge_p_en"]) / 2
    for qt in ["math", "meaning", "mcq"]:
        out[f"judge_x_{qt}"] = np.where((df["qtype"] == qt).values, mean, -1.0)
    return out


def make_models():
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.06,
                        subsample=0.9, colsample_bytree=0.8, reg_lambda=2.0,
                        min_child_weight=3, eval_metric="logloss",
                        random_state=42, n_jobs=4)
    lr = LogisticRegression(max_iter=2000, C=0.3, class_weight="balanced",
                            random_state=42)
    return xgb, lr


def stack_fit_predict(X_train, y_train, ctx_train, X_new, X_holdout):
    """Fit XGB+LR (50/50) on the training rows; return probabilities for X_new and
    for X_holdout (the 299 organizer rows, never trained on — used to calibrate the
    blend and thresholds downstream)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    cols = X_train.columns
    Xt = X_train.values.astype(np.float32)
    Xn = X_new[cols].values.astype(np.float32)
    Xh = X_holdout[cols].values.astype(np.float32)

    oof = np.zeros(len(y_train))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(Xt, y_train):
        xgb, lr = make_models()
        xgb.fit(Xt[tr], y_train[tr])
        sc = StandardScaler().fit(Xt[tr])
        lr.fit(sc.transform(Xt[tr]), y_train[tr])
        oof[va] = 0.5 * xgb.predict_proba(Xt[va])[:, 1] + \
            0.5 * lr.predict_proba(sc.transform(Xt[va]))[:, 1]
    print(f"stack train-OOF acc {(np.round(oof) == y_train).mean():.4f}")

    xgb, lr = make_models()
    xgb.fit(Xt, y_train)
    sc = StandardScaler().fit(Xt)
    lr.fit(sc.transform(Xt), y_train)

    def predict(X):
        return 0.5 * xgb.predict_proba(X)[:, 1] + 0.5 * lr.predict_proba(sc.transform(X))[:, 1]

    return predict(Xn), predict(Xh)
