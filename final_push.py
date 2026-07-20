"""Stage a private, manifest-bound Kaggle asset dataset for R3.

This command only builds the directory. It never uploads anything. The output
contains a pre-fitted stack, adapter, fixed calibration, and exact Phase-1 replay
cache. It intentionally excludes labeled features, organizer samples, holdout
probabilities, and OOF audit labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_extract(zip_path: Path, destination: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        root = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe zip member: {member.filename}")
        archive.extractall(destination)
    configs = list(destination.rglob("adapter_config.json"))
    if len(configs) != 1:
        raise ValueError(f"expected one adapter_config.json in zip, found {len(configs)}")
    return configs[0].parent


def copy_adapter(source: Path, destination: Path):
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"adapter missing {missing}")
    shutil.copytree(source, destination)


def build_manifest(root: Path):
    files = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()
                       and p.name not in {"asset_manifest.json", "dataset-metadata.json"}):
        rel = path.relative_to(root).as_posix()
        files[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "phase1-exact-auto_then_heldout-model",
        "privacy": "private Kaggle Dataset required",
        "files": files,
    }


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter-dir", type=Path)
    source.add_argument("--adapter-zip", type=Path)
    parser.add_argument("--stack-build", type=Path, required=True,
                        help="output directory from build_stack_bundle.py")
    parser.add_argument("--repro-build", type=Path, required=True,
                        help="output directory from build_repro_cache.py")
    parser.add_argument("--thresholds", type=Path, required=True,
                        help="judge-only fallback thresholds.json")
    parser.add_argument("--feature-models-manifest", type=Path, required=True,
                        help="output from build_feature_models_manifest.py")
    parser.add_argument("--output", type=Path, default=HERE / "upload_bn_halu_assets_r3")
    parser.add_argument("--dataset-id", default="shohos/bn-halu-assets-r3-private")
    parser.add_argument("--release-lock", type=Path, default=HERE / "r3_release_lock.json",
                        help="external lock embedded into generated notebooks")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r3-stage-", dir=args.output.parent) as temp:
        stage = Path(temp) / "assets"
        stage.mkdir()
        if args.adapter_zip:
            extracted = Path(temp) / "adapter_zip"
            extracted.mkdir()
            adapter = safe_extract(args.adapter_zip, extracted)
        else:
            adapter = args.adapter_dir
        copy_adapter(adapter, stage / "adapter")

        required_sources = {
            HERE / "inference_lib.py": stage / "inference_lib.py",
            HERE / "features_lib.py": stage / "features_lib.py",
            args.stack_build / "blend_config.json": stage / "blend_config.json",
            args.repro_build / "repro_cache.json": stage / "repro_cache.json",
            args.repro_build / "repro_manifest.json": stage / "repro_manifest.json",
            args.thresholds: stage / "thresholds.json",
            args.feature_models_manifest: stage / "feature_models_manifest.json",
        }
        for src, dst in required_sources.items():
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)
        bundle = args.stack_build / "stack_bundle"
        if not (bundle / "stack_manifest.json").is_file():
            raise FileNotFoundError(bundle / "stack_manifest.json")
        shutil.copytree(bundle, stage / "stack_bundle")

        metadata = {
            "title": "BN Halu R3 Private Assets",
            "id": args.dataset_id,
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        }
        (stage / "dataset-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        manifest = build_manifest(stage)
        (stage / "asset_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        banned = {"corpus_features.parquet", "dataset samples.json",
                  "holdout_probs.json", "stack_oof_audit.json"}
        leaked = [p for p in stage.rglob("*") if p.name in banned]
        if leaked:
            raise AssertionError(f"private training artifacts entered release: {leaked}")
        if not json.loads((stage / "dataset-metadata.json").read_text())["isPrivate"]:
            raise AssertionError("release dataset must be private")

        if args.output.exists():
            shutil.rmtree(args.output)
        shutil.move(stage, args.output)
    release_lock = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "asset_manifest_sha256": sha256_file(args.output / "asset_manifest.json"),
    }
    args.release_lock.parent.mkdir(parents=True, exist_ok=True)
    args.release_lock.write_text(
        json.dumps(release_lock, indent=2) + "\n", encoding="utf-8")
    print(f"R3 assets staged at {args.output}")
    print(f"manifest files: {len(manifest['files'])}; dataset id: {args.dataset_id}")
    print(f"release lock: {args.release_lock}")


if __name__ == "__main__":
    main()
