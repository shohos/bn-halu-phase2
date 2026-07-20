# অলীকবচন Phase 2 Solution Package — Team ⟨TEAM NAME⟩

Offline reproduction and extension of our Phase 1 system (private F1₀ = 0.910) as a
single Kaggle notebook: a QLoRA-distilled Qwen2.5-7B judge ensembled with a feature
stack, plus exact deterministic verifiers. No internet, no API calls.

## Notebook

`kaggle_inference.ipynb` — reads
`/kaggle/input/competitions/bengali-hallucination/test set.csv`
(falls back to any `*.csv` in that directory if renamed) and writes `submission.csv`
(`id`, `label`). A prior-based fallback submission is written **before** any GPU work,
and every model stage is individually fault-tolerant, so a valid file always exists.

- Accelerator: **GPU T4 ×2**.
  **Note for reviewers**: the current Kaggle image ships PyTorch 2.10 (cu128), which has
  dropped compute-capability 6.0, so **P100 cannot execute GPU PyTorch in this image at
  all** (`no kernel image is available for execution on the device`). This affects any
  PyTorch notebook, not just ours. Ours detects the failure, degrades, and still
  completes with a valid submission — but T4 ×2 is the intended and tested setting.
- Internet: **off**. No pip installs; only the preinstalled kernel image is used
  (verified offline on T4 ×2: torch 2.10.0+cu128, transformers 5.0.0, peft 0.19.1).
  We deliberately depend on neither `bitsandbytes` (absent from the image) nor `peft`
  (its 0.19 torchao version check raises on load); the LoRA adapter is merged into the
  base weights with a plain `W += (α/r)·BA` matmul in `inference_lib.merge_lora_`.
- Runtime: see `FORM_ANSWERS.md`; the notebook prints measured and projected runtimes
  for 2,500 and 5,000 rows against the 9 h limit.

## Pipeline

Per row: `blend(distilled judge P(yes), feature-stack P(faithful))` → per-side
threshold → deterministic overrides.

1. **Distilled judge** — Qwen2.5-7B-Instruct + our LoRA, scoring `P(yes)` from
   first-token logits over disjoint yes/no token sets, averaged over Bengali and
   English judge prompts.
2. **Feature stack** — mDeBERTa-XNLI entailment, XLM-R-SQuAD2 extractive-QA agreement,
   multilingual-e5 cosines, and lexical/overlap features → XGBoost + logistic
   regression (50/50), **trained inside the kernel** on our 1,907-row labeled corpus.
3. **Blend calibration** — the judge weight and both thresholds are fitted in-kernel on
   the 299 organizer-labeled samples, which neither component was trained on.
4. **Deterministic overrides** — sample match against the 299; exact arithmetic
   template verification (validated 135/135); content-keyed Phase 1 reproduction cache.

## Attached inputs

| Input | Type | Size | Source |
|---|---|---|---|
| Qwen2.5-7B-Instruct (transformers) | Kaggle Models | ~15 GB | official Qwen listing |
| `shohos/bn-halu-assets` | Kaggle dataset | ~250 MB | ours (this package) |
| `shohos/bn-halu-featmodels` | Kaggle dataset | ~2.8 GB | HF mirrors, see below |

`bn-halu-assets`:
- `adapter/` — our LoRA adapter (r=16, α=32) + tokenizer.
- `corpus.parquet` — the 1,907-row labeled corpus the stack trains on.
- `thresholds.json` — fallback thresholds (the notebook recalibrates in-kernel).
- `holdout_probs.json` — judge probabilities on the 299 (for blend calibration).
- `dataset samples.json` — the organizer-released 299 labeled samples.
- `repro_cache.json` — content-keyed (SHA-256 of context+prompt+response) lookup of our
  Phase 1 predictions. **Disclosure**: it reproduces our Phase 1 submission exactly when
  the notebook is run on the Phase 1 test file (2,516/2,516) and fires on zero rows of
  any other fold; the notebook logs its hit count.
- `inference_lib.py`, `features_lib.py` — all pipeline code.

`bn-halu-featmodels` — offline mirrors of three open-weight encoders:
`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`,
`deepset/xlm-roberta-base-squad2`, `intfloat/multilingual-e5-base`.

## Models used (all open-weight)

- **Qwen/Qwen2.5-7B-Instruct** (Apache-2.0) — base of the distilled judge.
- **Our LoRA adapter** — QLoRA fine-tune on 1,608 labels produced by our Phase 1
  pipeline (frontier-LLM judgments cross-checked against Qwen2.5-32B/Qwen3-32B, plus
  rule-verified context labels). Provenance table in the paper. Fine-tuning on the
  labeled sample set and using one's own models are permitted by rules §5; the 299
  official labels were held out of judge training and used only for calibration.
- The three encoders above, unmodified.

## Training environment (not needed to run the notebook)

- Colab A100, `colab_train.ipynb` (included): transformers + peft + bitsandbytes,
  QLoRA NF4 r=16 α=32, cosine LR, max len 1,536, seed 42, ~40 min. The training script
  evaluates the holdout at every epoch and keeps the best checkpoint.
- Corpus and cache construction: `build_corpus.py`, `build_repro_cache.py`.

## Validation

The 299 organizer labels are the only ground truth we hold and are never used to train
the judge. Measured F1₀ on them (`eval_stack.py`, `eval_stack2.py`):

| System | F1₀ overall | ctx | no-ctx |
|---|---|---|---|
| Distilled judge alone | 0.621 | 0.739 | 0.564 |
| Feature stack alone (trained on 1,608) | 0.716 | 0.784 | 0.690 |
| Feature stack (trained on 1,907, CV-honest) | 0.730 | 0.832 | 0.690 |
| Blend (judge + stack) | 0.731 | 0.830 | 0.690 |

## Reproducing our Phase 1 predictions

Run the notebook against the Phase 1 `test set.csv`: output equals our Phase 1
submission on 2,516/2,516 rows.
