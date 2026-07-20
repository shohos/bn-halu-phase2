# Training-checkpoint reproducibility status

`colab_train_original.ipynb` is preserved byte-for-byte because it is the available
notebook that produced the adapter. It is important evidence, but it is not a complete
environment lock:

- its install cell used unpinned `-U` packages;
- `Qwen/Qwen2.5-7B-Instruct` was not pinned to a Hugging Face commit;
- the notebook did not save `pip freeze`, CUDA/driver details, or training-data hashes;
- checkpoint selection used the 299 development labels, so those rows are not an
  untouched evaluation set.

Do not rewrite history by inserting guessed versions into the original notebook.
Instead:

1. preserve the original file and current adapter exactly;
2. run `build_training_manifest.py` to bind the notebook, JSONL files, exact command,
   and every adapter/tokenizer file;
3. save the current adapter's `adapter_config.json` (it records PEFT 0.19.1);
4. retain the exact Kaggle base-model source used at inference:
   `qwen-lm/qwen2.5/Transformers/7b-instruct/1`;
5. if retraining, create a new experiment rather than replacing the original, pin
   all package/model revisions, and save `pip freeze`, GPU/CUDA details, random seeds,
   logs, checkpoints, and hashes.

This is the most accurate lineage that can be produced from the available evidence.
It should accompany any organizer request for the training notebook.
