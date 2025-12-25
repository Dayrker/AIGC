#!/usr/bin/env bash
set -e
# Needed to assign
CUDA_DEVICES="4,5,6,7"
ARCH="NV"

# 参数解析
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      CUDA_DEVICES="$2"
      shift 2
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --precision)
      ARCH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
OUTPUT_DIR="checkpoint/$ARCH/sdxl-naruto"

# SetDown
MODEL_NAME="/mnt/zhangchen/S3Precision/models/StableDiffusion/stable-diffusion-xl-base-1.0"
# VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
DATASET_NAME="/mnt/zhangchen/S3Precision/datasets/naruto-blip-captions"
# --pretrained_vae_model_name_or_path=$VAE_NAME \
# --push_to_hub \

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
echo "Using arch=$ARCH -> Output dir=$OUTPUT_DIR"
accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=16 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="cute dragon creature" --report_to="wandb" \
  
