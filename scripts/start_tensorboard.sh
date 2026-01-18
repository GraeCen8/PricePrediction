
rm -rf runs

python tests.py

set -e
cd weights || { echo "weights directory not found"; exit 1; }
mkdir -p checkpoints
mv *checkpoint* checkpoints/ 2>/dev/null || echo "No checkpoint files found"

tensorboard --logdir=runs --port=6006 --reload_multifile=true