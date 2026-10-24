#!/usr/bin/env bash
# Builds the `real2sim` conda env: SAM 3 + SAM 3D Objects + RealSense + OpenUSD (usd-core).
# Supports both x86_64 workstations and aarch64 (NVIDIA Jetson Thor) on CUDA 12.8+.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

# Locate conda (Anaconda or Miniforge/Miniconda on x86_64 / aarch64)
for CONDA_SH in ~/anaconda3/etc/profile.d/conda.sh ~/miniforge3/etc/profile.d/conda.sh ~/miniconda3/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh; do
  if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    break
  fi
done

conda env list | grep -q "^real2sim " || conda create -y -n real2sim python=3.11
conda activate real2sim
unset PYTHONPATH  # ROS Humble python3.10 paths break this py3.11 env

# Auto-detect CUDA_HOME
if [ -z "${CUDA_HOME:-}" ]; then
  for d in /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda; do
    if [ -d "$d" ]; then
      export CUDA_HOME="$d"
      break
    fi
  done
fi
export PATH="$CUDA_HOME/bin:$PATH"

# Auto-detect GPU Compute Capability (e.g., 12.0 on RTX PRO 6000, 10.1/11.0 on Jetson Thor)
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || true)"
  export TORCH_CUDA_ARCH_LIST="${CAP:-12.0}"
fi
ARCH="$(uname -m)"
JOBS="$(nproc 2>/dev/null || echo 8)"
export MAX_JOBS="${MAX_JOBS:-$(( JOBS > 16 ? 16 : JOBS ))}"
echo "[setup] arch=$ARCH CUDA_HOME=$CUDA_HOME TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS"

# Clone third_party repositories if not already present
if [ ! -d "$TP/sam-3d-objects/.git" ]; then
  git clone --depth 1 https://github.com/facebookresearch/sam-3d-objects.git "$TP/sam-3d-objects"
fi
if [ ! -d "$TP/sam3/.git" ]; then
  git clone --depth 1 https://github.com/facebookresearch/sam3.git "$TP/sam3"
fi

pip install -U pip wheel ninja "setuptools<81"  # pkg_resources is still imported by SAM 3
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# SAM 3D Objects core deps (official list minus cu121 pins / unused heavy packages)
grep -vE "^(torchaudio|spconv-cu121|nvidia-cuda-nvcc-cu12|nvidia-pyindex|cuda-python|xformers|bpy|sagemaker|auto_gptq|bitsandbytes|mosaicml-streaming|optimum|peft|sentence-transformers|librosa|pip-system-certs|timm|hdfs|pymongo|smplx|decord|av|wandb|jupyter|black|flake8|usort|autoflake|pdoc3|conda-pack|polyscope|pyrender|panda3d-gltf)==" \
  "$TP/sam-3d-objects/requirements.txt" > "$TP/.req_sam3d.txt" || true
pip install -r "$TP/.req_sam3d.txt"
pip install spconv-cu126 "timm>=1.0.17" "numpy<2" || {
  echo "[setup] prebuilt spconv wheel unavailable on $ARCH; building cumm + spconv from source"
  export CUMM_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"
  export SPCONV_DISABLE_JIT="1"
  pip install pccm ccimport
  pip install --no-build-isolation "git+https://github.com/FindDefinition/cumm.git"
  pip install --no-build-isolation "git+https://github.com/traveller59/spconv.git"
  pip install "timm>=1.0.17" "numpy<2"
}
pip install --no-deps -e "$TP/sam-3d-objects"

# pytorch3d / gsplat from source (for Blackwell / Jetson Thor)
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"
pip install "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" seaborn
pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"  # mesh hole filling
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.1_cu128.html \
  || pip install --no-build-isolation "git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.18.0"
(cd "$TP/sam-3d-objects" && ./patching/hydra) || true

# SAM 3
pip install --no-deps -e "$TP/sam3"
pip install ftfy regex iopath typing_extensions huggingface_hub einops pycocotools psutil

# Camera + geometry + standalone OpenUSD exporter
pip install open3d==0.19.0 trimesh scipy "numpy<2" "setuptools<81" fast_simplification usd-core
pip install pyrealsense2 || echo "[setup] warning: pyrealsense2 wheel not found for $ARCH (build librealsense from source if capturing directly)"

python - <<'PY'
import torch, pytorch3d, spconv, open3d
from pxr import Usd
print("torch", torch.__version__, torch.cuda.get_device_capability(0))
print("OpenUSD (pxr) OK")
PY
echo "real2sim env OK"
