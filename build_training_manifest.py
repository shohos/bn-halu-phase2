"""Create a private checkpoint-lineage record for organizer verification."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return {"size": path.stat().st_size, "sha256": h.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-notebook", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--holdout-jsonl", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--command", required=True,
                        help="exact training command, including flags")
    args = parser.parse_args()
    inputs = [args.training_notebook, args.train_jsonl, args.holdout_jsonl]
    adapter_files = sorted(p for p in args.adapter.rglob("*") if p.is_file())
    if not adapter_files:
        raise FileNotFoundError("adapter directory is empty")
    files = {}
    for path in inputs + adapter_files:
        files[str(path)] = digest(path)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "training_command": args.command,
        "python": platform.python_version(),
        "files": files,
        "note": "Private lineage record; not an inference-time asset.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {len(files)} committed files")


if __name__ == "__main__":
    main()
