import os
import json
import torch
import argparse
from datasets import load_dataset
# Eval Metrics
from transformers import CLIPModel, CLIPProcessor
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms.functional as TF
# helper functions
from utils import same_seed

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--base_model", type=str, default="stable-diffusion-xl-base-1.0")
    parser.add_argument("--lora_model", type=str, default="sd-naruto-model-lora-sdxl")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    return args

def LoadPipe(args, device="cuda:0"):
    from diffusers import StableDiffusionXLPipeline as SDPipeline   # 放这里才不会影响args.cuda
    ### Get Parameters
    base_model = args.base_model
    lora_model = args.lora_model

    ### Get model path
    SD_dir = "/mnt/zhangchen/S3Precision/models/StableDiffusion/"   # Base model dir
    model_path = SD_dir + base_model
    LORA_dir = "/mnt/zhangchen/S3Precision/AIGC/SD/checkpoint/"     # Lora model dir
    lora_path = LORA_dir + lora_model
    lora_name = "pytorch_lora_weights.safetensors"

    ### Get pipeline
    pipe = SDPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    if lora_model != "None":
        pipe.load_lora_weights(lora_path, weight_name=lora_name)
    pipe.to(device)
    return pipe

def runSD(prompts: list,
          SDPipeline = None,
          num_inference_steps = 30, guidance_scale = 7.5):
    """
    base_model: str, lora_model: str, 
    SDPipeline = StableDiffusionXLPipeline,
    num_inference_steps = 30, guidance_scale = 7.5
    """
    pipe = SDPipeline

    # StableDiffusionXLPipelineOutput(images=[<PIL.Image.Image image mode=RGB size=1024x1024 at 0x7C0C8B59EBF0>, <>, ...])
    output = pipe(prompts, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    return output.images

### For FID metric
def pil_list_to_uint8_batch(pil_list, device):
    """
    PIL list -> uint8 tensor [B,3,H,W], range [0,255]
    """
    x = torch.stack(
        [TF.to_tensor(im.convert("RGB").resize((1024, 1024))) for im in pil_list], dim=0
    )  # float [0,1]
    x = (x * 255).clamp(0, 255).to(torch.uint8).to(device)
    return x

### For CLIP metric
def compute_CLIPScoreI2I(imgs1, imgs2, device="cuda:0"):
    """
    return: List[float], length B
    imgs1: gen  images
    imgs2: real images
    """
    assert len(imgs1) == len(imgs2)
    B = len(imgs1)

    # Get CLIP model
    model_path = "/mnt/zhangchen/S3Precision/models/StableDiffusion/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_path).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(model_path)

    # Get score 
    inputs = clip_processor(images=imgs1 + imgs2, return_tensors="pt").to(device)
    feats = clip_model.get_image_features(**inputs)          # [2, D]
    feats = feats / feats.norm(dim=-1, keepdim=True)    # L2 normalize

    gen_feats = feats[:B]      # [B, D]
    real_feats = feats[B:]     # [B, D]
    scores = (gen_feats * real_feats).sum(dim=-1)       # cosine similarity

    return scores.tolist()

def evalMetrics(dataset, start_index=0, device="cuda:0", args=None):
    """
    batch_size: int
    """
    ### Get Parameters
    batch_size = args.batch_size
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    # Get SD Pipeline
    pipe = LoadPipe(args, device)

    real_images = dataset["image"]
    texts = dataset["text"]
    Metrics = []

    from alive_progress import alive_bar
    with alive_bar((len(dataset) + batch_size - 1) // batch_size) as bar:
        for i in range(0, len(dataset), batch_size):
            gen_images_batch  = runSD(texts[i: i + batch_size], pipe)
            real_images_batch = real_images[i: i + batch_size]

            # compute CLIP
            CLIPScores = compute_CLIPScoreI2I(gen_images_batch, real_images_batch, device)
            
            # compute FID (prepare for)
            fid_metric.update(pil_list_to_uint8_batch(gen_images_batch, device), real=True,)
            fid_metric.update(pil_list_to_uint8_batch(real_images_batch, device), real=False,)

            for j in range(len(gen_images_batch)):
                metric = {
                    "index": start_index + i + j,
                    "prompt": texts[i + j],
                }
                metric["CLIPScore"] = CLIPScores[j]

                Metrics.append(metric)

            bar()   # 每个batch推进一次
    
    # ---------- summary ----------
    mean_CLIPScore = sum(metric["CLIPScore"] for metric in Metrics) / len(Metrics)
    FIDScore = fid_metric.compute().item()
    Metrics.append({
        "mean_CLIPScore": round(mean_CLIPScore, 4),
        "FIDScore": round(FIDScore, 4),
    })
    
    return Metrics


if __name__ == "__main__":
    same_seed(0)
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda


    # prompts = ["A naruto with green eyes and red legs.",
    #            "a man with dark hair and brown eyes"]
    # images = runSD(prompts, args)
    # print("images:", images)
    # image.save("images/naruto.png")
    
    dataset_path="/mnt/zhangchen/S3Precision/datasets/naruto-blip-captions"
    dataset = load_dataset(dataset_path)["train"]
    Metrics = evalMetrics(dataset, args=args)
    print("Metrics:", Metrics)
    
    LORA_dir = "/mnt/zhangchen/S3Precision/AIGC/SD/checkpoint/"     # Lora model dir
    lora_path = LORA_dir + args.lora_model
    with open(lora_path, "w", encoding="utf-8") as f:
        json.dump(Metrics, f, indent=2, ensure_ascii=False)
