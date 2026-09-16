#!/bin/bash
# Prepare a Colab runtime for the prior pre-training experiment. Run from the root of the
# cloned repository after Google Drive is mounted:
#
#     bash run/colab_setup.sh
#
# The repository is private; it is cloned and pulled with a read-only fine-grained token
# kept in Colab Secrets as GITHUB_TOKEN, never in the remote URL. The dumps are too large
# for git and live in the Drive folder. `.cache/tfmp` becomes a link into that folder, so
# checkpoints, work files and logs survive a disconnected runtime and a run resumes from
# its work checkpoint. Colab's own torch is kept: pip is pinned to the installed version.
set -e
DRIVE=${DRIVE:-/content/drive/MyDrive/tfmp}
TFMP_COMMIT=7b37681364101ab7167742856de9ed8b4d243b90

test -d "$DRIVE/dumps" || { echo "no $DRIVE/dumps: upload the dumps to Drive first"; exit 1; }
mkdir -p "$DRIVE/checkpoints" "$DRIVE/work" "$DRIVE/logs" .cache
[ -L .cache/tfmp ] || ln -s "$DRIVE" .cache/tfmp

if [ ! -d /content/TFM-Playground ]; then
  git clone -q https://github.com/automl/TFM-Playground /content/TFM-Playground
fi
git -C /content/TFM-Playground checkout -q "$TFMP_COMMIT"

python -c "import torch; print('torch==' + torch.__version__.split('+')[0])" \
  > /tmp/torch-constraint.txt
pip install -q -c /tmp/torch-constraint.txt "pfns==0.3.0" h5py schedulefree
pip install -q --no-deps --ignore-requires-python -e /content/TFM-Playground
pip install -q --no-deps -e .

python - <<'EOF'
import torch, pfns, tfmplayground, h5py, schedulefree
from common.adapters import nanotabpfn
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
print(f"torch {torch.__version__} | {name} | bf16 {bf16}")
if not bf16:
    print("WARNING: no bf16 on this card. Pick an L4 or A100 runtime, not a T4.")
EOF
