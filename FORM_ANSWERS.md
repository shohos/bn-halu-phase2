# Phase 2 Submission Form — answers

| Field | Answer |
|---|---|
| Inference Notebook Link (Github) | `<FILL: https://github.com/<user>/bn-halu-phase2/blob/main/kaggle_inference.ipynb>` |
| Inference Notebook Link (Kaggle) | https://www.kaggle.com/code/shohos/bn-halu-inference |
| Runtime on test set of 2500 rows (In Hours) | `<FILL from final dry run>` — measured 0.17 h with the judge partially skipped; expect ~0.5 h with the full judge, ~1.0 h projected at 5000 rows (limit 9 h) |
| GPU Used | **GPU T4*2** (P100 cannot run PyTorch in the current Kaggle image — see README) |
| Kaggle link of external dataset used as knowledge base | https://www.kaggle.com/datasets/shohos/bn-halu-featmodels (+ https://www.kaggle.com/datasets/shohos/bn-halu-assets) |
| Uploaded the model checkpoint on | **Kaggle** |

## MUST DO BEFORE SUBMITTING — everything has to be public
- [ ] Dataset `shohos/bn-halu-assets` -> Settings -> Visibility: **Public**
- [ ] Dataset `shohos/bn-halu-featmodels` -> Settings -> Visibility: **Public**
- [ ] Notebook `shohos/bn-halu-inference` -> Share -> **Public**
- [ ] GitHub repo public, containing: kaggle_inference.ipynb, inference_lib.py,
      features_lib.py, colab_train.ipynb, train_qlora.py, build_corpus.py,
      build_repro_cache.py, README.md, paper.pdf
- [ ] Attach the Qwen2.5-7B-Instruct **Kaggle Model** to the notebook (already in metadata)

## Optional extra notebooks (encouraged for reproducibility)
- `colab_train.ipynb` — QLoRA training that produced the checkpoint (organizers may
  request this after top-15 selection to verify the checkpoint provenance).
