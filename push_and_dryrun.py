"""Upload R3 assets/kernel, wait, and enforce positive execution gates.

Run production first, then run the separate 2x probe kernel. External writes occur
only when this script is explicitly invoked by the team.
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def log_text(output: Path) -> str:
    text = ""
    for path in output.glob("*.log"):
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            text += "".join(str(r.get("data", "")) for r in records)
        except Exception:
            text += path.read_text(encoding="utf-8", errors="replace")
    return text


def validate_notebook(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    ids = []
    all_source = ""
    for cell in nb["cells"]:
        if not cell.get("id"):
            raise ValueError("every notebook cell requires an id")
        ids.append(cell["id"])
        if cell["cell_type"] == "code":
            source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            all_source += source
            ast.parse(source)
    if len(ids) != len(set(ids)):
        raise ValueError("notebook cell ids are not unique")
    if "UNBOUND_REGENERATE_AFTER_FINAL_PUSH" in all_source:
        raise ValueError("notebook is not bound to staged assets; rerun make_notebook.py")


def wait_dataset(api, ref, timeout=1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = str(api.dataset_status(ref)).lower()
        print("dataset state:", state)
        if state.rsplit(".", 1)[-1] == "ready":
            print("STAGE:DATASET_READY")
            return
        if "error" in state or "failed" in state:
            raise RuntimeError(f"dataset processing failed: {state}")
        time.sleep(15)
    raise TimeoutError("dataset did not become READY within 30 minutes")


def wait_kernel(api, ref, timeout=9 * 3600 + 1800):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        status = str(api.kernels_status(ref).status)
        if status != last:
            print("kernel state:", status, flush=True)
            last = status
        upper = status.upper()
        if "COMPLETE" in upper:
            return status
        if "ERROR" in upper or "CANCEL" in upper or "FAILED" in upper:
            raise RuntimeError(f"kernel failed: {status}")
        time.sleep(30)
    raise TimeoutError("kernel exceeded the 9-hour evaluation budget")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["production", "runtime-probe"], required=True)
    parser.add_argument("--assets", type=Path, default=HERE / "upload_bn_halu_assets_r3")
    parser.add_argument("--dataset-ref", default="shohos/bn-halu-assets-r3-private")
    parser.add_argument("--kernel-dir", type=Path)
    parser.add_argument("--kernel-ref")
    parser.add_argument("--test", type=Path, help="local Phase-1 test CSV (production)")
    parser.add_argument("--phase1-submission", type=Path, help="exact Phase-1 CSV (production)")
    parser.add_argument("--skip-dataset-upload", action="store_true")
    args = parser.parse_args()

    if args.kernel_dir is None:
        args.kernel_dir = HERE / ("kernel_inference" if args.mode == "production"
                                  else "kernel_runtime_probe")
    if args.kernel_ref is None:
        args.kernel_ref = ("shohos/bn-halu-inference" if args.mode == "production"
                           else "shohos/bn-halu-runtime-probe")
    notebook_name = ("kaggle_inference.ipynb" if args.mode == "production"
                     else "kaggle_runtime_probe.ipynb")
    validate_notebook(args.kernel_dir / notebook_name)
    print("STAGE:NOTEBOOK_STATIC_OK")

    metadata = json.loads((args.assets / "dataset-metadata.json").read_text(encoding="utf-8"))
    if metadata.get("id") != args.dataset_ref or metadata.get("isPrivate") is not True:
        raise ValueError("asset metadata must match dataset-ref and set isPrivate=true")
    sys.path.insert(0, str(args.assets))
    import inference_lib as lib
    lib.validate_asset_manifest(args.assets, required=True)
    print("STAGE:LOCAL_ASSETS_OK")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    if not args.skip_dataset_upload:
        try:
            api.dataset_status(args.dataset_ref)
            result = api.dataset_create_version(
                str(args.assets), version_notes="R3 manifest-bound private assets", dir_mode="zip")
        except Exception:
            result = api.dataset_create_new(str(args.assets), dir_mode="zip")
        print("dataset push:", result)
    wait_dataset(api, args.dataset_ref)

    print("kernel push:", api.kernels_push(str(args.kernel_dir)))
    wait_kernel(api, args.kernel_ref)
    output = HERE / ("dryrun_production" if args.mode == "production" else "dryrun_runtime_probe")
    if output.exists():
        shutil.rmtree(output)
    api.kernels_output(args.kernel_ref, path=str(output), force=True)
    logs = log_text(output)
    print(logs)

    forbidden = ["ASSETS FAILED", "JUDGE FAILED", "STACK FAILED", "PIPELINE FAILED",
                 "MODELS DEGRADED"]
    present_bad = [marker for marker in forbidden if marker in logs]
    if present_bad:
        raise RuntimeError(f"degraded/failure markers present: {present_bad}")
    required = ["STAGE:ASSETS_OK", "STAGE:JUDGE_OK", "STAGE:FEATURE_MODELS_OK", "STAGE:STACK_OK",
                "STAGE:CALIBRATION_OK", "STAGE:FINAL_OK", "STAGE:PIPELINE_OK",
                "STAGE:NOTEBOOK_OK"]
    required.append("STAGE:REPRO_ACTIVE" if args.mode == "production"
                    else "STAGE:REPRO_INACTIVE")
    if args.mode == "runtime-probe":
        required.append("STAGE:RUNTIME_PROBE_OK")
    missing = [marker for marker in required if marker not in logs]
    if missing:
        raise RuntimeError(f"positive execution markers missing: {missing}")

    filename = "submission.csv" if args.mode == "production" else "probe_submission.csv"
    submission = pd.read_csv(output / filename)
    if list(submission.columns) != ["id", "label"] or not submission["id"].is_unique:
        raise ValueError("output schema or ids invalid")
    if not submission["label"].isin([0, 1]).all() or submission.isna().any().any():
        raise ValueError("output labels invalid")

    if args.mode == "production":
        if not args.test or not args.phase1_submission:
            raise ValueError("production requires --test and --phase1-submission")
        test = pd.read_csv(args.test)
        expected = pd.read_csv(args.phase1_submission)
        if submission["id"].tolist() != test["id"].tolist():
            raise ValueError("production id values/order differ from public test")
        joined = submission.merge(expected, on="id", suffixes=("_r3", "_phase1"), validate="one_to_one")
        agreement = (joined["label_r3"] == joined["label_phase1"]).mean()
        if agreement != 1.0:
            raise ValueError(f"Phase-1 reproduction is not exact: {agreement:.6f}")
        print("STAGE:PRODUCTION_DRYRUN_OK exact Phase-1 reproduction")
    else:
        if len(submission) < 5000:
            raise ValueError(f"runtime probe too small: {len(submission)}")
        print(f"STAGE:RUNTIME_DRYRUN_OK {len(submission)} held-out-path rows")


if __name__ == "__main__":
    main()
