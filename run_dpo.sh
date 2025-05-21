#!/bin/bash

python3 src/train_dpo.py \
    --dataset_path /root/multimodal-garment-designer/assets/data/dresscode \
    --dataset dresscode \
    --batch_size 1 \
    --mixed_precision fp16 \
    --output_dir ./dpo_results \
    --learning_rate 1e-5 \
    --max_train_steps 1000 \
    --beta 0.1 \
    --gradient_accumulation_steps 4 \
    --save_steps 100 \
    --num_workers 4