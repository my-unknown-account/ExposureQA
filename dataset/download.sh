#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${SCRIPT_DIR}"
REPO_ID="JamshidJDMY/ExposureQA"
MODE="${1:-}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"
DECOMPRESS_WORKERS="${DECOMPRESS_WORKERS:-0}"

mkdir -p "${DATASET_DIR}"

require_python_pkg() {
  local pkg="$1"
  python - "$pkg" <<'PY'
import importlib, sys
pkg = sys.argv[1]
try:
    importlib.import_module(pkg)
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

ensure_dependencies() {
  local need_full="${1:-0}"
  local missing=()
  if [ "${need_full}" = "1" ]; then
    require_python_pkg "huggingface_hub" || missing+=("huggingface_hub[hf_transfer]")
  fi
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "Missing Python packages: ${missing[*]}"
    echo "Install with: pip install ${missing[*]}"
    exit 1
  fi
}

download_simple() {
  python - "${DATASET_DIR}" <<'PY'
from pathlib import Path
from urllib.request import urlretrieve
import sys

dataset_dir = Path(sys.argv[1])
out = dataset_dir / "exposureQA.json"
if not out.exists():
    urlretrieve(
        "https://huggingface.co/datasets/JamshidJDMY/ExposureQA/resolve/main/exposureQA.json",
        str(out),
    )
print(f"Ready: {out}")
PY
}

download_full() {
  HF_HUB_ENABLE_HF_TRANSFER=1 python - "${DATASET_DIR}" "${REPO_ID}" "${DOWNLOAD_WORKERS}" "${DECOMPRESS_WORKERS}" <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download
import copy
import os
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed

dataset_dir = Path(sys.argv[1])
repo_id = sys.argv[2]
download_workers = int(sys.argv[3])
decompress_workers = int(sys.argv[4])
if decompress_workers <= 0:
    decompress_workers = max(4, min(32, (os.cpu_count() or 8) * 2))
print(f"Using decompress workers: {decompress_workers}")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["runs/*.tar.gz"],
    local_dir=str(dataset_dir),
    local_dir_use_symlinks=False,
    max_workers=download_workers,
)

hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename="exposureQA.json",
    local_dir=str(dataset_dir),
    local_dir_use_symlinks=False,
)

repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset")

passage_files = sorted(
    p for p in repo_files
    if p.startswith("passages/") and p.count("/") >= 2 and p.endswith(".tar.gz")
)
if passage_files:
    print(f"Downloading passages: {len(passage_files)} files ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=passage_files,
        local_dir=str(dataset_dir),
        local_dir_use_symlinks=False,
        max_workers=download_workers,
    )
else:
    print("No passage files found.")

def _is_within_directory(base_dir: Path, target: Path) -> bool:
    base = base_dir.resolve()
    dest = target.resolve()
    return str(dest).startswith(str(base))

def _split_parts(name: str):
    return [p for p in name.split("/") if p not in ("", ".")]

def extract_tar_gz_and_remove(archive_path: Path) -> None:
    extract_to = archive_path.parent
    with tarfile.open(archive_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name and m.name != "./"]
        top_levels = set()
        for m in members:
            parts = _split_parts(m.name)
            if parts:
                top_levels.add(parts[0])

        strip_prefix = None
        if len(top_levels) == 1:
            only = next(iter(top_levels))
            # If the archive is wrapped in a single root folder (e.g. "run/"),
            # strip it so files land directly in extract_to.
            strip_prefix = only

        prepared = []
        for orig in members:
            parts = _split_parts(orig.name)
            if strip_prefix and parts and parts[0] == strip_prefix:
                parts = parts[1:]
            if not parts:
                continue
            member = copy.copy(orig)
            member.name = "/".join(parts)
            prepared.append((orig, member))

        for _, member in prepared:
            member_target = extract_to / member.name
            if not _is_within_directory(extract_to, member_target):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")

        for orig, member in prepared:
            if member.isdir():
                (extract_to / member.name).mkdir(parents=True, exist_ok=True)
                continue
            if member.issym() or member.islnk():
                # Skip links for safety/portability in this dataset flow.
                continue
            target = extract_to / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(orig)
            if src is None:
                continue
            with src, open(target, "wb") as dst:
                dst.write(src.read())
    archive_path.unlink()

runs_dir = dataset_dir / "runs"
if runs_dir.exists():
    run_jobs = []
    for archive_file in runs_dir.glob("*.tar.gz"):
        run_jobs.append(archive_file)
    if run_jobs:
        total_runs = len(run_jobs)
        print(f"Decompressing runs: 0/{total_runs}")
        with ThreadPoolExecutor(max_workers=decompress_workers) as ex:
            futures = [ex.submit(extract_tar_gz_and_remove, archive) for archive in run_jobs]
            done = 0
            for f in as_completed(futures):
                f.result()
                done += 1
                print(f"Decompressing runs: {done}/{total_runs}")

passages_dir = dataset_dir / "passages"
if passages_dir.exists():
    passage_jobs = []
    for archive_file in passages_dir.rglob("*.tar.gz"):
        passage_jobs.append(archive_file)
    if passage_jobs:
        total_passages = len(passage_jobs)
        print(f"Decompressing passages: 0/{total_passages}")
        with ThreadPoolExecutor(max_workers=decompress_workers) as ex:
            futures = [ex.submit(extract_tar_gz_and_remove, archive) for archive in passage_jobs]
            done = 0
            for f in as_completed(futures):
                f.result()
                done += 1
                print(f"Decompressing passages: {done}/{total_passages}")

print("Full dataset ready.")
PY
}

if [ -z "${MODE}" ]; then
  if [ -t 0 ]; then
    echo "Choose dataset mode:"
    echo "1) ExposureQA-Simple (only exposureQA.json)"
    echo "2) ExposureQA-Full (exposureQA.json + runs + passages)"
    read -r -p "Enter 1 or 2: " MODE
  else
    echo "No TTY detected. Use: $0 1  (simple) or $0 2  (full)"
    exit 1
  fi
fi

case "${MODE}" in
  1)
    ensure_dependencies 0
    download_simple
    ;;
  2)
    ensure_dependencies 1
    download_full
    ;;
  *)
    echo "Invalid choice: ${MODE}"
    echo "Use: $0 1  (simple) or $0 2  (full)"
    exit 1
    ;;
esac

echo "Done."
