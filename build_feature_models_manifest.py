"""Commit to every file in the separately attached feature-model dataset."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = ["mdeberta-xnli", "xlmr-squad2", "e5-base"]
    for name in required:
        if not (args.root / name / "config.json").is_file():
            raise FileNotFoundError(args.root / name / "config.json")
    # Kaggle consumes dataset-metadata.json and does not publish it, and never ships
    # bytecode/OS cruft. Hashing them makes the mount validation fail on files that
    # cannot be present.
    skip_names = {"dataset-metadata.json", ".DS_Store"}
    def excluded(rel):
        return (rel in skip_names or "__pycache__" in rel or rel.endswith(".pyc"))
    files = {}
    for path in sorted(p for p in args.root.rglob("*") if p.is_file()):
        rel = path.relative_to(args.root).as_posix()
        if excluded(rel):
            print("skip", rel)
            continue
        files[rel] = {"size": path.stat().st_size, "sha256": digest(path)}
        print(rel, files[rel]["sha256"][:12])
    manifest = {"schema_version": 1, "root_contract": required, "files": files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {len(files)} files")


if __name__ == "__main__":
    main()
