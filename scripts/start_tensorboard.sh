
rm -rf runs
python tests.py
tensorboard --logdir=runs --port=6006 --reload_multifile=true