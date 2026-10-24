#!/usr/bin/env bash
# Downloads gated SAM 3 and SAM 3D Objects checkpoints. Requires approved access on
# https://huggingface.co/facebook/sam3 and https://huggingface.co/facebook/sam-3d-objects and `hf auth login`.
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

hf auth whoami
# SAM 3: cached in ~/.cache/huggingface, loaded by build_sam3_image_model(load_from_HF=True)
hf download facebook/sam3 sam3.pt config.json

cd "$ROOT/third_party/sam-3d-objects"
if [ ! -f checkpoints/hf/pipeline.yaml ]; then
  hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects
  mv checkpoints/hf-download/checkpoints checkpoints/hf
  rm -rf checkpoints/hf-download
fi
ls -la checkpoints/hf
