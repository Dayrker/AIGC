import os
from PIL import Image
# Wan config
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video
# process model
from .load_pipeline import load_ti2v_pipeline, get_wan_task
from .convert_model import replace_modules
from .help_funcs import same_seed, getContent
# Parallel
import torch.multiprocessing as mp


def vedio_dump(video, fps, sub_dir, index):
    OUTPUT_VIDEOS_DIR = "/mnt/zhangchen/S3Precision/AIGC/WAN/output/"    # Need to set

    out_video_path = OUTPUT_VIDEOS_DIR + sub_dir + f"/video_{index}.mp4"
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
    save_video(
        tensor=video[None],
        save_file=out_video_path,
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    return out_video_path

"""
SIZE_CONFIGS: 
{'720*1280': (720, 1280), 
'1280*720': (1280, 720), 
'480*832': (480, 832), 
'832*480': (832, 480), 
'704*1280': (704, 1280), 
'1280*704': (1280, 704), 
'1024*704': (1024, 704), 
'704*1024': (704, 1024)}
MAX_AREA_CONFIGS: 
{'720*1280': 921600, 
'1280*720': 921600, 
'480*832': 399360, 
'832*480': 399360, 
'704*1280': 901120, 
'1280*704': 901120, 
'1024*704': 720896, 
704*1024': 720896}
"""
def infer_batch(args, dataset, 
                ref_image_path = None, device_id = 0,
                return_queue = None):
    # Get Parameters
    model     = args.model
    task      = get_wan_task(model)
    size      = args.size
    arch      = args.arch
    precision = args.precision
    same_seed(42)

    # Load Pipeline & convert
    cfg = WAN_CONFIGS[task]
    pipe = load_ti2v_pipeline(cfg, model, device_id=device_id)
    replace_modules(pipe.model, arch=arch, precision=precision)
    print(f"Replace modules over.")

    img = None
    if ref_image_path:
        img = Image.open(ref_image_path).convert("RGB")

    frame_num = 25  # fps: 24 (frame_num need +1)
    sample_steps = 50
    outputs = []
    for _, data in enumerate(dataset):
        prompt   = data["caption"]
        video_id = data["video_id"]
        id       = data["id"]
        sample = {"id": id}

        SIZE = "1280*704" if size == "720p" else "832*480"

        # transformer engine
        content = getContent(args.arch, args.precision)
        with content:
            video = pipe.generate(
                prompt,
                img=img,
                size=SIZE_CONFIGS[SIZE],
                max_area=MAX_AREA_CONFIGS[SIZE],
                frame_num=frame_num,
                shift=cfg.sample_shift,
                sample_solver="unipc",
                sampling_steps=sample_steps,
                guide_scale=cfg.sample_guide_scale,
                seed=1234,
                offload_model=True,
            )

        # save info
        sub_dir = f"{model}/{size}/{arch}/{precision}"
        save_path = vedio_dump(video, cfg.sample_fps, sub_dir, id)
        sample["path"]   = save_path
        sample["prompt"] = prompt
        outputs.append(sample)

    if return_queue:    # 返回本GPU结果，用于multi-gpu
        return_queue.put(outputs)
    return outputs


def infer_batch_multiprocessing(args, dataset):
    ### parse args needed
    cuda_ids = args.cuda

    dataset_len = len(dataset)
    world_size = len(cuda_ids.split(","))

    # split dataset
    data_slices = []
    per_gpu = (dataset_len + world_size - 1) // world_size
    for i in range(world_size):
        start = i * per_gpu
        end   = min(start + per_gpu, dataset_len)
        data_slices.append(dataset[start: end])

    # === 启动多进程 ===
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_queue = manager.Queue()
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target = infer_batch,
            args = (args, data_slices[rank], None, rank, 
                    return_queue)
        )
        p.start()
        processes.append(p)
    for p in processes: p.join()    # 阻塞主进程，等待这个子进程结束 -> 等待所有 GPU 完成

    # === 收集4卡结果 ===
    all_results = []
    while not return_queue.empty():
        all_results.extend(return_queue.get())
    # === 按 index 排序 ===
    all_results = sorted(all_results, key=lambda x: x["id"])

    print(f"🔥 已完成 {world_size}-GPU 推理.")
    return all_results
