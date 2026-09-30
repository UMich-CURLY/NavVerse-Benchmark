# 
# pip install rsl-rl-lib==3.0.1
# python policy/train.py --task Isaac-Velocity-Rough-Spot-v0 --num_envs 4096 --enable_cameras --livestream 2
python policy/train.py \
    --task Isaac-Velocity-Rough-Spot-v0 \
    --num_envs 8192 \
    --livestream 2 \
    --max_iterations 10000 \
    --resume \
    --load_run 2025-12-07_03-56-21 \
    --video \
    --enable_cameras \
    --device "cuda:0"
