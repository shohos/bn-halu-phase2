# R3 merge and run instructions

## 1. Merge source, not generated private data

Copy these R3 files over their project equivalents:

- `inference_lib.py`
- `features_lib.py`
- `build_repro_cache.py`
- `final_push.py`
- `make_notebook.py`
- `push_and_dryrun.py`
- `check_paper.py`
- `paper/main.tex` and `paper/acl.sty`

Add these new files/directories:

- `build_stack_bundle.py`
- `build_feature_models_manifest.py`
- `check_feature_parity.py`
- `build_training_manifest.py`
- `tests/test_r3.py`
- `kernel_inference/kernel-metadata.json`
- `kernel_runtime_probe/kernel-metadata.json`

Keep your real adapter, Phase-1 test/submission, corpus, feature parquet, holdout
probabilities, and feature-model directory private. Do not overwrite them with any
example or generated file in this bundle.

## 2. Check identifiers

The supplied metadata assumes:

- Kaggle user: `shohos`
- private asset slug: `bn-halu-assets-r3-private`
- feature-model slug: `bn-halu-featmodels`
- production kernel: `bn-halu-inference`
- runtime probe kernel: `bn-halu-runtime-probe`
- base model: `qwen-lm/qwen2.5/Transformers/7b-instruct/1`

If any identifier differs, update it consistently in both kernel metadata files,
`final_push.py` arguments, and `push_and_dryrun.py` arguments.

## 3. Build in this order

```bash
python -m unittest discover -s tests -v

python build_stack_bundle.py \
  --features data/corpus_features.parquet \
  --judge-probs data/holdout_probs.json \
  --output build/stack

python build_repro_cache.py \
  --test "data/test set.csv" \
  --submission data/phase1_submission.csv \
  --output build/repro

python build_feature_models_manifest.py \
  --root /path/to/bn-halu-featmodels \
  --output build/feature_models_manifest.json

python final_push.py \
  --adapter-zip bn_halu_adapter.zip \
  --stack-build build/stack \
  --repro-build build/repro \
  --thresholds build/adapter/thresholds.json \
  --feature-models-manifest build/feature_models_manifest.json \
  --output upload_bn_halu_assets_r3

python make_notebook.py
python check_paper.py paper/main.tex --compile
```

Use the same XGBoost major version for bundle construction and Kaggle inference when
possible. The builder records all relevant package versions in the stack manifest.

## 4. Run feature parity before uploading the final release

Run in the exact Kaggle image/feature-model mount:

```bash
python check_feature_parity.py \
  --corpus data/corpus.parquet \
  --reference-features data/corpus_features.parquet \
  --feature-models /kaggle/input/bn-halu-featmodels \
  --report build/feature_parity_report.json
```

Do not continue unless it prints `STAGE:FEATURE_PARITY_OK`.

## 5. Run production gate

```bash
python push_and_dryrun.py \
  --mode production \
  --assets upload_bn_halu_assets_r3 \
  --dataset-ref shohos/bn-halu-assets-r3-private \
  --test "data/test set.csv" \
  --phase1-submission data/phase1_submission.csv
```

Required:

- dataset reaches READY;
- judge, feature-model, stack, calibration, pipeline, final, and notebook markers;
- `STAGE:REPRO_ACTIVE`;
- exact ID values/order;
- agreement 1.000000 with the Phase-1 submission;
- no failure/degradation markers.

## 6. Run the held-out-path 2x gate

```bash
python push_and_dryrun.py \
  --mode runtime-probe \
  --assets upload_bn_halu_assets_r3 \
  --dataset-ref shohos/bn-halu-assets-r3-private \
  --skip-dataset-upload
```

Required:

- at least 5,000 rows;
- `STAGE:REPRO_INACTIVE`;
- both judge templates unless the measured runtime strategy intentionally changes;
- both model components and fixed calibration;
- `STAGE:RUNTIME_PROBE_OK` below nine hours;
- no failure/degradation markers.

## 7. Final paper and training lineage

After `build_stack_bundle.py`, compare final OOF metrics in `blend_config.json` with
the rounded values in `paper/main.tex`. If different, update the paper and re-run:

```bash
python check_paper.py paper/main.tex --compile
```

Generate a private training-lineage record using the exact command that produced the
adapter:

```bash
python build_training_manifest.py \
  --training-notebook colab_train_original.ipynb \
  --train-jsonl data/train.jsonl \
  --holdout-jsonl data/holdout_299.jsonl \
  --adapter build/adapter \
  --command "python train_qlora.py --epochs 3 --lr 4e-5 --dropout 0.1" \
  --output build/training_manifest.json
```

Keep that manifest, notebook, exact data, and checkpoint private but ready for the
organizers' top-15 verification request.

## 8. Release decision

Submit only after both Kaggle gates pass on the exact staged assets. The older v8 log
does not validate R3. If possible, ask the organizers whether exact full-file replay is
an acceptable interpretation of the public Phase-1 reproduction gate.
