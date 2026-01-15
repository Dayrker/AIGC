import wan

"""
'Config: Wan TI2V 5B', 
't5_model': 'umt5_xxl', 
't5_dtype': torch.bfloat16, 
'text_len': 512, 
'param_dtype': torch.bfloat16, 
'num_train_timesteps': 1000, 
'sample_fps': 24, 
'sample_neg_prompt': '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走', 
'frame_num': 121, 
't5_checkpoint': 'models_t5_umt5-xxl-enc-bf16.pth', 
't5_tokenizer': 'google/umt5-xxl', 
'vae_checkpoint': 'Wan2.2_VAE.pth', 
'vae_stride': (4, 16, 16), 
'patch_size': (1, 2, 2), 
'dim': 3072, 
'ffn_dim': 14336, 
'freq_dim': 256, 
'num_heads': 24, 
'num_layers': 30, 
'window_size': (-1, -1), 
'qk_norm': True, 
'cross_attn_norm': True, 
'eps': 1e-06, 
'sample_shift': 5.0, 
'sample_steps': 50, 
'sample_guide_scale': 5.0}
"""
def load_ti2v_pipeline(cfg, model, device_id=0):
    model_dir = "/mnt/zhangchen/S3Precision/models/Wan/Wan2.2-" # Need to set
    model_path = model_dir + model

    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=model_path,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )
    # print(f"Loading {pipe} over.")
    return pipe


# dict_keys(['t2v-A14B', 'i2v-A14B', 'ti2v-5B', 'animate-14B', 's2v-14B'])
def get_wan_task(model):
    if model == "T2V-A14B":
        task = "t2v-A14B"
    elif model == "I2V-A14B":
        task = "i2v-A14B"
    elif model == "TI2V-5B":
        task = "ti2v-5B"
    elif model == "S2V-14B":
        task = "s2v-14B"
    elif model == "Animate-14B":
        task = "animate-14B"
    else:
        raise ValueError(f"Not support WAN model: {model}")

    return task
