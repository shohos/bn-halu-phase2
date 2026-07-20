# Bengali Hallucination Phase 2 - R3

This directory is a merge-ready hardening pass for the Phase-2 submission. It is
designed around the organizers' actual two-run procedure:

1. approximately 2,516 public rows must reproduce the Phase-1 result;
2. the same notebook then predicts approximately 5,000 unseen rows within 9 hours.

R3 has not yet been run on Kaggle. The last verified public run is the older v8
pipeline (2xT4, 2,516 rows, 43.1 minutes). Do not submit R3 until both dry-run gates
below pass.

## Execution policy

The production notebook always calls `run(..., repro_mode=True)`, but this is an
automatic exact gate, not per-row lookup:

- On the public Phase-1 file, the complete multiset of normalized row hashes must
  equal the committed multiset and its SHA-256 signature. Only then are all 2,516
  Phase-1 predictions replayed.
- On any changed, partial, duplicated, or unseen input, the cache applies to zero
  rows. The judge and pre-fitted stack produce every prediction.
- Sample-label and arithmetic overrides are disabled in the production notebook.
  They remain in `inference_lib.py` only for explicit ablations/tests.

Keep `bn-halu-assets-r3-private` private. It contains the Phase-1 prediction map.
The staged dataset metadata enforces `isPrivate: true`. It intentionally excludes
organizer sample rows, labeled corpus features, holdout probabilities, and OOF
audit labels.

## R3 architecture

Per row:

```text
Qwen2.5-7B judge P(faithful) + pre-fitted XGBoost/logistic stack
  -> fixed per-context blend calibrated from 5-fold OOF development predictions
  -> fixed threshold
  -> exact public-file replay only if the full dataset signature matches
```

The feature stack no longer trains inside the Kaggle inference kernel. The private
builder makes OOF predictions for the 299 development rows; every fold trains on
all 1,608 pseudo-labeled rows plus four-fifths of the development rows. It then
fits one final model on all 1,907 rows and ships only model parameters, feature
schema, and calibration. This removes the old calibration/test probability-scale
mismatch and removes labeled feature rows from inference assets.

## Build order

All paths below are examples; use the corresponding paths in your merged project.

### 1. Build the final stack and calibration privately

```bash
python build_stack_bundle.py \
  --features data/corpus_features.parquet \
  --judge-probs data/holdout_probs.json \
  --output build/stack
```

This requires `xgboost`, `scikit-learn`, `numpy`, `pandas`, and a parquet engine.
The output contains `stack_bundle/`, `blend_config.json`, and a private
`stack_oof_audit.json`. The audit file is never copied into Kaggle assets.

### 2. Build the public-file replay commitment

```bash
python build_repro_cache.py \
  --test "data/test set.csv" \
  --submission data/phase1_submission.csv \
  --output build/repro
```

The command rejects ID mismatches, non-binary labels, content duplicates, and
conflicting structure.

### 3. Bind the attached feature-model dataset

```bash
python build_feature_models_manifest.py \
  --root /path/to/bn-halu-featmodels \
  --output build/feature_models_manifest.json
```

Before release, run feature parity in the exact Kaggle image:

```bash
python check_feature_parity.py \
  --corpus data/corpus.parquet \
  --reference-features data/corpus_features.parquet \
  --feature-models /path/to/bn-halu-featmodels \
  --report build/feature_parity_report.json
```

This catches tokenizer/model/version drift and silent QA fallback behavior.

### 4. Stage the private Kaggle assets

```bash
python final_push.py \
  --adapter-zip bn_halu_adapter.zip \
  --stack-build build/stack \
  --repro-build build/repro \
  --thresholds build/adapter/thresholds.json \
  --feature-models-manifest build/feature_models_manifest.json \
  --output upload_bn_halu_assets_r3
```

`final_push.py` performs safe ZIP extraction, copies an allowlist, rejects known
label-bearing files, writes private dataset metadata, and hashes every runtime
artifact into `asset_manifest.json`. It also writes `r3_release_lock.json` outside
the dataset; `make_notebook.py` embeds that hash so a notebook cannot silently run
against a different asset version.

### 5. Generate both notebooks

```bash
python make_notebook.py
```

Outputs:

- `kaggle_inference.ipynb`: production two-pass notebook;
- `kaggle_runtime_probe.ipynb`: duplicates the public file to about 5,032 rows,
  assigns fresh IDs, and exercises the held-out path with replay disabled.

Both notebooks have stable cell IDs, exact input/output ID checks, finite
probability checks, positive completion markers, and an outer fallback boundary.
The copies included in this source bundle are intentionally unbound templates;
the release gate rejects them until `final_push.py` writes the external lock and
`make_notebook.py` is run again.

### 6. Run local tests

```bash
python -m unittest discover -s tests -v
python check_paper.py paper/main.tex
```

### 7. Run both Kaggle gates

Production/public reproduction:

```bash
python push_and_dryrun.py \
  --mode production \
  --test "data/test set.csv" \
  --phase1-submission data/phase1_submission.csv
```

Required result: all model stages complete, `STAGE:REPRO_ACTIVE` appears, output
IDs are exact, and agreement with the Phase-1 submission is 1.000000.

Held-out-path runtime probe:

```bash
python push_and_dryrun.py --mode runtime-probe --skip-dataset-upload
```

Required result: at least 5,000 rows, `STAGE:REPRO_INACTIVE`, both model stages,
finite probabilities, `STAGE:RUNTIME_PROBE_OK`, and elapsed time below 9 hours.

## Safe-degradation behavior

The notebook writes a valid prior submission before asset discovery. Failures do
not leave a missing or malformed CSV. However, a structurally valid fallback is
not considered a successful dry run: the release script rejects missing positive
markers, any model degradation, invalid probabilities, wrong IDs, and cache-mode
mismatches.

## Files that must remain private

- Phase-1 `repro_cache.json` and its source submission;
- `corpus_features.parquet` with labels;
- `dataset samples.json`;
- `holdout_probs.json`;
- `stack_oof_audit.json`;
- raw/pseudo-labeled training corpus.

The pre-fitted stack bundle and adapter may be attached to the private inference
notebook. Preserve the original training notebook and data lineage: organizers may
request the notebook used to produce the checkpoint after top-15 selection.

## Verified and unverified status

Verified locally in this R3 tree:

- Python syntax and notebook-cell parsing;
- exact/99%/duplicate-substitution cache boundaries;
- strict context-aware sample matching and conflict rejection;
- negation/correction decline behavior;
- probability, manifest, ID, and LR-bundle contracts;
- paper compilation and visual checks (when `paper/main.pdf` is present).

Not yet verified until you run the commands above:

- final XGBoost bundle and OOF calibration values;
- feature parity in the current Kaggle image;
- exact Phase-1 reproduction with the rebuilt assets;
- the 2x held-out-path runtime;
- held-out semantic performance.

Do not treat an old v8 Kaggle log as verification of R3.
