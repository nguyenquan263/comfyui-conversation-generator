#!/home/qnguyen3/.venvs/comfyui/bin/python
"""
Download the model files a pipeline needs (resumable, verified by size and SHA-256).

    ./setup_models.py                 # show what is present / missing
    ./setup_models.py wan ltx25       # download everything those pipelines need
    ./setup_models.py all
    ./setup_models.py --verify     # re-hash every present file against its known SHA-256

All sources are public HuggingFace repos (no token needed). Files download to a .part
file and are renamed only when complete, so an interrupted download never looks finished.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys

import pipelines


def download(key: str) -> None:
    rel, url = pipelines.FILES[key]
    dst = pipelines.MODELS / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if pipelines.file_ok(key):
        print(f"  ok       {rel}")
        return
    part = dst.with_suffix(dst.suffix + ".part")
    print(f"  download {rel}  ({pipelines.SIZES[key] / 1e9:.2f} GB)")
    # curl -C - resumes a partial .part file; --fail turns HTTP errors into non-zero exits.
    rc = subprocess.call(["curl", "-L", "--fail", "-C", "-", "--retry", "5", "--progress-bar", "-o", str(part), url])
    if rc != 0 or part.stat().st_size < pipelines.SIZES[key]:
        sys.exit(f"download of {rel} failed (curl exit {rc}); re-run to resume")
    print(f"  verify   {rel}", end="", flush=True)
    if sha256_of(part) != pipelines.SHA256[key]:
        part.unlink()
        sys.exit(f"\nchecksum mismatch for {rel}; the partial file was deleted, re-run to download it again")
    print("  ok")
    part.replace(dst)


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_all() -> None:
    bad = 0
    for key, (rel, _) in pipelines.FILES.items():
        path = pipelines.MODELS / rel
        if not path.exists():
            continue
        ok = sha256_of(path) == pipelines.SHA256[key]
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'}  {rel}")
    print("all good" if not bad else f"{bad} corrupt file(s); delete them and re-run setup_models.py")


def main(argv: list[str]) -> None:
    if argv == ["--verify"]:
        return verify_all()
    if not argv:
        for name, st in pipelines.pipeline_status().items():
            print(f"{name:12} {'ready' if st['available'] else 'missing: ' + ', '.join(st['missing_files'])}")
        print("\nusage: setup_models.py <pipeline>... | all")
        return
    wanted = list(pipelines.PIPELINE_FILES) if argv == ["all"] else argv
    keys: list[str] = []
    for p in wanted:
        if p not in pipelines.PIPELINE_FILES:
            sys.exit(f"unknown pipeline {p!r}; choose from {list(pipelines.PIPELINE_FILES)} or 'all'")
        keys += [k for k in pipelines.PIPELINE_FILES[p] if k not in keys]
    total = sum(pipelines.SIZES[k] for k in keys if not pipelines.file_ok(k)) / 1e9
    print(f"{len(keys)} files, {total:.1f} GB still to download")
    for k in keys:
        download(k)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:])
