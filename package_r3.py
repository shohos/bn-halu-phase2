"""Create a clean source archive and per-file SHA-256 manifest."""
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / "BENGALI_HALLUCINATION_R3_MERGE_BUNDLE.zip"
MANIFEST = ROOT / "R3_SOURCE_MANIFEST.json"
EXCLUDED_DIRS = {"__pycache__", "tmp", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".pyc"}


def included(path: Path):
    rel = path.relative_to(ROOT)
    return not any(part in EXCLUDED_DIRS for part in rel.parts) and \
        path.suffix not in EXCLUDED_SUFFIXES and path != MANIFEST


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and included(p))
    banned = {"corpus_features.parquet", "dataset samples.json", "holdout_probs.json",
              "repro_cache.json", "stack_oof_audit.json", "adapter_model.safetensors"}
    leaked = [p for p in files if p.name in banned]
    if leaked:
        raise SystemExit(f"private artifact in source bundle: {leaked}")
    manifest = {
        "schema_version": 1,
        "bundle": OUTPUT.name,
        "note": "Source-only R3 bundle; generated notebooks are unbound templates.",
        "files": {p.relative_to(ROOT).as_posix(): {
            "size": p.stat().st_size, "sha256": digest(p)} for p in files},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files.append(MANIFEST)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in sorted(files):
            archive.write(path, Path("project_fixed_r3") / path.relative_to(ROOT))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    print(f"sha256 {digest(OUTPUT)}")


if __name__ == "__main__":
    main()
