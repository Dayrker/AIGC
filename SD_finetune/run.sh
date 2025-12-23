export MODEL_NAME="/mnt/zhangchen/S3Precision/models/StableDiffusion/stable-diffusion-xl-base-1.0"
# export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="/mnt/zhangchen/S3Precision/datasets/naruto-blip-captions"

# --pretrained_vae_model_name_or_path=$VAE_NAME \
# --push_to_hub \
accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="checkpoint/sd-naruto-model-lora-sdxl" \
  --validation_prompt="cute dragon creature" --report_to="wandb" \