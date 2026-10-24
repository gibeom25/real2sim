#!/usr/bin/env bash
# Downloads gated SAM 3 and SAM 3D Objects checkpoints using Python huggingface_hub API.
# Requires approved access on:
#   https://huggingface.co/facebook/sam3
#   https://huggingface.co/facebook/sam-3d-objects
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for CONDA_SH in ~/anaconda3/etc/profile.d/conda.sh ~/miniforge3/etc/profile.d/conda.sh ~/miniconda3/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh; do
  if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    break
  fi
done
conda activate real2sim
unset PYTHONPATH  # ROS Humble python3.10 paths break this py3.11 env

python -c '
import getpass
import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, login, snapshot_download
from huggingface_hub.errors import GatedRepoError

def prompt_token():
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        tok = getpass.getpass("Hugging Face Token (for authorized account, e.g. gibeom25): ").strip()
    if tok:
        login(token=tok)

api = HfApi()
try:
    user = api.whoami()
    print(f"[hf] currently logged in as: {user.get(\"name\")}")
except Exception:
    print("[hf] not logged in yet.")
    prompt_token()

for attempt in range(2):
    try:
        print("[hf] downloading facebook/sam3 ...")
        hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt")
        hf_hub_download(repo_id="facebook/sam3", filename="config.json")
        break
    except GatedRepoError:
        if attempt == 0:
            print("\n[hf] ERROR: Current account does not have approved access to facebook/sam3.")
            print("[hf] Please enter a token for an authorized account (e.g. gibeom25):")
            prompt_token()
        else:
            raise

sam3d_dir = Path(sys.argv[1])
target_dir = sam3d_dir / "checkpoints" / "hf"
if not (target_dir / "pipeline.yaml").exists():
    print("[hf] downloading facebook/sam-3d-objects ...")
    dl_dir = sam3d_dir / "checkpoints" / "hf-download"
    snapshot_download(repo_id="facebook/sam-3d-objects", local_dir=str(dl_dir), max_workers=2)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(dl_dir / "checkpoints"), str(target_dir))
    shutil.rmtree(dl_dir, ignore_errors=True)

print("[hf] checkpoints ready at:", target_dir)
for p in sorted(target_dir.iterdir()):
    print("  ", p.name)
' "$ROOT/third_party/sam-3d-objects"
