#!/usr/bin/env bash
# Build DROID-SLAM (Princeton-VL, 2021) for a same-modality STEREO fight vs
# visloc-rs on KITTI. Isolated venv (torch 2.3.1+cu121, matches nvcc 12.0).
set -e
export CUDA_HOME=/usr TORCH_CUDA_ARCH_LIST="8.9"
ROOT=$HOME/droid_battle
VENV=$ROOT/venv
mkdir -p "$ROOT"; cd "$ROOT"

echo "### [1/6] venv + torch 2.3.1 cu121"
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel setuptools
python -c "import torch" 2>/dev/null || \
  pip install -q torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"

echo "### [2/6] clone DROID-SLAM (recursive: lietorch+eigen submodules)"
[ -d "$ROOT/DROID-SLAM" ] || git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
cd "$ROOT/DROID-SLAM"
git submodule update --init --recursive

echo "### [3/6] python deps"
pip install -q numpy'<2' opencv-python scipy tqdm matplotlib pyyaml einops \
  torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html || \
  pip install -q numpy'<2' opencv-python scipy tqdm matplotlib pyyaml einops
pip install -q evo open3d 2>&1 | tail -1 || true

echo "### [4/6] build CUDA ext (droid_backends + lietorch) --no-build-isolation"
python setup.py install 2>&1 | tail -8

echo "### [5/6] weights droid.pth"
if [ ! -f droid.pth ]; then
  pip install -q gdown
  gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O droid.pth 2>&1 | tail -2 || \
  wget -q "https://www.dropbox.com/s/jc8oa5x1bbk9w8x/droid.pth?dl=1" -O droid.pth 2>&1 | tail -2 || true
fi
ls -la droid.pth 2>/dev/null || echo "WEIGHTS MISSING"

echo "### [6/6] verify import"
python -c "import torch; import droid_backends, lietorch; print('DROID CUDA ext OK')" 2>&1 | tail -3
echo "### DROID SETUP DONE"
