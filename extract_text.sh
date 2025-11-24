#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python extract_text.py \
    --text_csv /path/to/csv_file.csv \
    --output_cache_path /path/to/dir/of/gt_cache \
    --clap_ckpt_path /path/to/laion_clap_ckpt.pt

