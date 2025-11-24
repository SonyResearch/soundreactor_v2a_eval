#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python extract_video.py \
    --gt_cache /path/to/dir/of/gt_cache \
    --video_path /path/to/dir/of/target_videos \
    --syncformer_ckpt_path /path/to/synchformer_state_dict.pth \
    --gt_batch_size 4 --audio_length=7.8 --start_time 0