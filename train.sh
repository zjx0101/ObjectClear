#!/bin/bash
# Train ObjectClear on the OBER dataset (multi-GPU with 🤗 accelerate).
#
# Prerequisites (see README for details):
#   1. conda activate objectclear
#   2. Download the CLIP image encoder into ./ckpts/clip-vit-large-patch14
#   3. Download the OBER dataset parquet shards into ./data/OBER/data
#
# Adjust the paths / hyper-parameters below to your setup, then run:
#   bash train.sh
set -e

# Number of GPUs to train on.
NUM_GPUS=8

# --- Paths (edit these to match your environment) ---
PRETRAINED_MODEL="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
IMAGE_ENCODER="./ckpts/clip-vit-large-patch14"
TRAIN_DATA="./data/OBER/data"                                   # OBER parquet shards
VAL_PARQUET="./data/OBER/data/test-00000-of-00001.parquet"      # OBER test parquet
OUTPUT_DIR="./runs/train_objectclear"
MODEL_CACHE="./model_cache"

accelerate launch \
    --multi_gpu \
    --num_processes ${NUM_GPUS} \
    --num_machines 1 \
    --mixed_precision fp16 \
    train_objectclear.py \
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
    --image_encoder_name_or_path "${IMAGE_ENCODER}" \
    --model_dir "${MODEL_CACHE}" \
    --output_dir "${OUTPUT_DIR}" \
    --image_dir1 "${TRAIN_DATA}" \
    --resolution 512 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 4 \
    --learning_rate 1e-05 \
    --learning_rate_attn 1e-05 \
    --lr_scheduler cosine \
    --lr_warmup_steps 500 \
    --max_train_steps 100000 \
    --checkpointing_steps 5000 \
    --checkpoints_total_limit 5 \
    --validation_parquet "${VAL_PARQUET}" \
    --validation_subset "OBER-Test" \
    --validation_num_samples 8 \
    --validate_by_iter \
    --validation_iterations 2000 \
    --seed 42 \
    --mixed_precision fp16 \
    --color_augmentation \
    --flip_augmentation \
    --random_mask_dilation \
    --random_mask_erosion \
    --object_localization \
    --object_localization_weight 0.01 \
    --background_loss_weight 1 \
    --real_only
