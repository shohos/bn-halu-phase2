"""One command from adapter zip -> Kaggle dry run verdict.

Usage:  python push_and_dryrun.py path/to/bn_halu_adapter.zip
  1. unzips adapter/thresholds/holdout_probs into upload_bn_halu_assets/
  2. creates or versions the shohos/bn-halu-assets dataset
  3. pushes shohos/bn-halu-inference (T4 x2, internet off, Qwen model attached)
  4. polls to completion, downloads submission.csv, diffs vs Phase 1 submission
"""
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "upload_bn_halu_assets"
KERNEL = HERE / "kernel_inference"


def main():
    import ast
    from kaggle.api.kaggle_api_extended import KaggleApi

    # a syntax error costs a full kernel round trip; catch it here instead
    nb = json.loads((KERNEL / "kaggle_inference.ipynb").read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse(c["source"] if isinstance(c["source"], str) else "".join(c["source"]))
    print("notebook cells parse OK")

    api = KaggleApi()
    api.authenticate()

    if len(sys.argv) > 1:
        z = Path(sys.argv[1])
        with zipfile.ZipFile(z) as f:
            f.extractall(ASSETS)
        assert (ASSETS / "adapter" / "adapter_model.safetensors").exists()
        assert (ASSETS / "thresholds.json").exists()
        print("unzipped:", json.loads((ASSETS / "thresholds.json").read_text()))

    # dataset: create if missing, else new version
    try:
        api.dataset_status("shohos/bn-halu-assets")
        r = api.dataset_create_version(str(ASSETS), version_notes="update", dir_mode="zip")
    except Exception:
        r = api.dataset_create_new(str(ASSETS), dir_mode="zip")
    print("dataset push:", r)
    for _ in range(60):  # wait until Kaggle finishes processing the new version
        try:
            if str(api.dataset_status("shohos/bn-halu-assets")).lower() == "ready":
                break
        except Exception:
            pass
        time.sleep(15)
    print("dataset ready; settling 30s"); time.sleep(30)

    print("kernel push:", api.kernels_push(str(KERNEL)))
    for i in range(400):  # judge on 2xT4 is pipeline-parallel and slow; allow >3 h
        st = str(api.kernels_status("shohos/bn-halu-inference").status)
        if i % 10 == 0 or "RUNNING" not in st.upper():
            print(f"[{i * 30}s] {st}", flush=True)
        if any(k in st.upper() for k in ["COMPLETE", "ERROR", "CANCEL"]):
            break
        time.sleep(30)

    out = HERE / "dryrun_out"
    shutil.rmtree(out, ignore_errors=True)
    api.kernels_output("shohos/bn-halu-inference", path=str(out), force=True)
    print("output files:", [p.name for p in out.iterdir()])

    log = next(out.glob("*.log"), None)
    if log:
        for r_ in json.load(open(log, encoding="utf-8")):
            if r_["stream_name"] == "stdout":
                print(r_["data"], end="")

    sub_path = out / "submission.csv"
    if sub_path.exists():
        sub = pd.read_csv(sub_path)
        phase1 = pd.read_csv(HERE.parent / "submission.csv")
        m = sub.merge(phase1, on="id", suffixes=("_kaggle", "_phase1"))
        agree = (m.label_kaggle == m.label_phase1).mean()
        print(f"\nDRY RUN VERDICT: {len(sub)} rows | agreement vs Phase 1: {agree:.4f}")
        assert agree == 1.0, "dry run must reproduce Phase 1 exactly"
        print("REPRODUCTION EXACT — dry run PASSED")
    else:
        print("NO submission.csv in output — dry run FAILED, read the log above")


if __name__ == "__main__":
    main()
