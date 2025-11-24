#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0



python evaluate.py \
    --gt_audio /path/to/dir/of/gt_audio \
    --gt_cache /path/to/dir/of/gt_cache   \
    --pred_audio /path/to/dir/of/pred_audio  \
    --pred_cache /path/to/dir/of/pred_cache \
    --audio_length=8 --pred_batch_size 12 --num_workers 4  \
    --gt_batch_size 12 \
    --mono_type "mean" --start_time 0.0 \
    --clap_ckpt_path /path/to/laion_clap_ckpt.pt \
    --syncformer_ckpt_path /path/to/synchformer_state_dict.pth --recompute_gt_cache --recompute_pred_cache \

# --mono_type "mean", "left", "right", "side"
# you can remove --recompute_gt_cache and/or --recompute_pred_cache if the cache files are already computed