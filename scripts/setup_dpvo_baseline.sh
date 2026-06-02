#!/usr/bin/env bash
# Set up DPVO (Princeton-VL, 2023-2024 deep monocular VO/SLAM) in an isolated
# venv so its pinned torch (2.3.1+cu121, matching system nvcc 12.0) does not
# collide with the system torch 2.10. Logs everything; idempotent-ish.
set -e
export CUDA_HOME=/usr
ROOT=$HOME/dpvo_battle
VENV=$ROOT/venv
mkdir -p "$ROOT"
cd "$ROOT"

echo "### [1/6] venv"
if [ ! -d "$VENV" ]; then python3 -m venv "$VENV"; fi
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel setuptools

echo "### [2/6] torch 2.3.1 cu121"
python -c "import torch" 2>/dev/null || \
  pip install -q torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"

echo "### [3/6] clone DPVO"
if [ ! -d "$ROOT/DPVO" ]; then git clone --depth 1 https://github.com/princeton-vl/DPVO.git; fi
cd "$ROOT/DPVO"

echo "### [4/6] eigen thirdparty"
if [ ! -d thirdparty/eigen-3.4.0 ]; then
  wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip -O /tmp/eigen.zip
  mkdir -p thirdparty && unzip -q -o /tmp/eigen.zip -d thirdparty
fi

echo "### [5/6] python deps + build CUDA ext"
pip install -q numpy'<2' opencv-python yacs einops kornia plyfile evo pypose tqdm matplotlib scipy
# torch-scatter wheel for torch 2.3.1 cu121
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
# build DPVO's CUDA/CPP extensions
pip install -e . 2>&1 | tail -5

echo "### [6/6] pretrained weights"
mkdir -p "$ROOT/DPVO"
if [ ! -f dpvo.pth ]; then
  pip install -q gdown
  gdown 1dRqftpImtHbbIPLkIYZGyfLO5HoFDUOj -O models_dpvo.zip 2>&1 | tail -2 || true
  [ -f models_dpvo.zip ] && unzip -q -o models_dpvo.zip || true
  ls -la dpvo.pth 2>/dev/null || echo "WEIGHTS: gdown may have failed, will retry alternate"
fi
echo "### DPVO SETUP DONE"
python -c "import dpvo; print('dpvo import OK')" 2>&1 | tail -2
ls -la "$ROOT/DPVO"/*.pth 2>/dev/null || true
