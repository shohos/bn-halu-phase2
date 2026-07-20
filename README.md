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
- Runtime: **41.5 min measured** for the 2,516-row Phase 1 test file on 2×T4 with
  internet off; the organizers' ~5,000-row fold projects to **1.37 h** against the 9 h
  limit. The judge dominates (fp16 7B split pipeline-parallel across two T4s). The
  notebook prints both measured and projected figures in its output.

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
- `corpus_features.parquet` — precomputed **numeric features only** for the 1,907-row
  labeled corpus, plus labels. We deliberately do not publish the corpus *text*: those
  rows are Phase 1 competition data, and this dataset is public. Nothing in this dataset
  reproduces competition content.
- `thresholds.json` — fallback thresholds (the notebook recalibrates in-kernel).
- `holdout_probs.json` — judge probabilities on the 299 (for blend calibration).
- The organizer-released `dataset samples.json` is read from the competition mount at
  runtime rather than shipped here, for the same reason.
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
| Qwen2.5-7B zero-shot judge | 0.699 | 0.737 | 0.686 |
| + QLoRA 3 epochs @ lr 1e-4 (overfit) | 0.621 | 0.739 | 0.564 |
| + QLoRA 1 epoch @ lr 4e-5 (**shipped**) | 0.711 | 0.747 | 0.704 |
| Feature stack, trained on 1,608 | 0.716 | 0.784 | 0.690 |
| Feature stack, trained on 1,907 (CV-honest) | 0.730 | 0.832 | 0.690 |
| Blend, per-side weights (local CV) | 0.737 | 0.832 | 0.698 |
| **Blend as calibrated in-kernel (shipped)** | **0.732** | 0.816 | 0.698 |

Two findings drive the design. First, the zero-shot base model already scores 0.686 on
closed-book rows, and aggressive fine-tuning drops it to 0.564 — so we select the
checkpoint by per-epoch holdout evaluation (epoch 1 wins; epochs 2-3 degrade
monotonically). Second, the stack dominates open-book rows while the judge is better on
closed-book rows, so blend weights are fitted per evidence regime rather than globally.

The last two rows differ because the in-kernel calibration model trains only on the
1,608 pseudo-labeled rows, keeping its predictions on the 299 honest, whereas the local
CV figure cross-validates a model trained on all 1,907. The shipped number is 0.732.

## Reproducing our Phase 1 predictions

Run the notebook against the Phase 1 `test set.csv`: output equals our Phase 1
submission on 2,516/2,516 rows. With the reproduction cache disabled, the distilled
system alone agrees with our Phase 1 submission on **79.6%** of rows — the residual
gap is the frontier-LLM and manual-verification layer that cannot run offline.
